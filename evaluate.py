import json
import math
import time
import os
import torch
import datetime
import hashlib
from pathlib import Path
from eval_protocol import parse_mc, score_exact, parse_exact_answer, slice_bounds, harmonic_mean
from tqdm import tqdm
from datasets import load_dataset
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
from peft import PeftModel
import random
import warnings

ICL_PROMPT = None
model = None
tokenizer = None
# provide your own perspective API key through google cloud
PERSPECTIVE_API_KEY = None
perspective_already_warned = False

client = None


def _perspective_client():
    global client
    if client is None:
        from googleapiclient import discovery
        client = discovery.build(
            "commentanalyzer", "v1alpha1", developerKey=PERSPECTIVE_API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
    return client

multitask_domain_dataset_dict = {
    "legal": ["hearsay", "citation_prediction_classification"],
    "medical": ["medqa", "medmcqa"],
    "science": ["scifact", "stem"],
    "culture": ["normad_country", "normad_value"],
}

# setting up Vertex AI API for Gemini model access, for objective 4

# please provide your own project_id
project_id = None
location_list = ["us-east5", "us-south1", "us-central1", "us-west4", "us-east1", "us-east4", "us-west1"]
gemini_model = None


def _gemini_client():
    global gemini_model, generationConfig, safety_config
    if gemini_model is None:
        import vertexai
        from vertexai.generative_models import (
            GenerativeModel, GenerationConfig, SafetySetting, HarmCategory,
            HarmBlockThreshold,
        )
        vertexai.init(project=project_id, location=random.choice(location_list))
        gemini_model = GenerativeModel("gemini-1.5-flash-001")
        generationConfig = GenerationConfig(temperature=0, max_output_tokens=20)
        safety_config = [SafetySetting(category=category,
                                      threshold=HarmBlockThreshold.BLOCK_NONE)
                         for category in (
                             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                             HarmCategory.HARM_CATEGORY_HARASSMENT,
                             HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                             HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT)]
    return gemini_model

ONLY_ONE_OR_TWO = None

# determisnistic model behavior for reproducibility
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def lora_weight_visualize(path):
    state_dict_this = load_file(path, device="cpu")
    return [state_dict_this["base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"][0][0].item(), state_dict_this["base_model.model.model.layers.10.self_attn.k_proj.lora_B.weight"][0][0].item()]

# print(lora_weight_visualize("search/human_{asian_food}_{0.3}_{0.2}_{0.5}_{0.05}_{0.6}_a100-16-4-bk-2/particle_3/personal_best/adapter_model.safetensors"))

def parse_gemini_score(response):
    # It should be "Rating: ?/10"
    if "Rating: " not in response:
        warnings.warn("Gemini score parsing error for string: " + response)
        return 1 # output format invalid
    score = response.split("Rating: ")[1].split("/10")[0]
    return int(score)

# print(parse_gemini_score("Rating: 2/10"))

@retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(10))
def gemini_eval(prompt, response):
    eval_prompt = "For the following instruction and response, please rate the response on a scale of 1 to 10, where 1 is the worst and 10 is the best. Please respond with \"Rating: ?/10\" first and then provide your reason.\n\n + \
    Instruction: " + prompt + "\n\nResponse: " + response
    eval_response = _gemini_client().generate_content(eval_prompt, generation_config = generationConfig, safety_settings = safety_config).text

    time.sleep(1)

    return parse_gemini_score(eval_response)

# print(gemini_eval("Please tell me about your experience with the product.", "I just can't stop using this phone case! It has a very nice texture and also very light, perfect for long trips."))

def update_only_one_or_two(value):
    global ONLY_ONE_OR_TWO
    ONLY_ONE_OR_TWO = value

def curret_time_string():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

# metric for AbstainQA, check out https://arxiv.org/abs/2402.00367
def get_effective_reliability(correct_flags, abstain_flags):
    answered_correct = 0
    answered_incorrect = 0

    for i in range(len(correct_flags)):
        if abstain_flags[i] == 0:
            if correct_flags[i] == 1:
                answered_correct += 1
            else:
                answered_incorrect += 1

    return (answered_correct - answered_incorrect) / len(correct_flags)

@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=20))
def get_toxicity_score(text):
    text = text.split("\n")[0].strip()
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}}
    }

    response = _perspective_client().comments().analyze(body=analyze_request).execute()
    return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]

def multiple_choice_prompt(instance_dict, dataset):
    prompt = "Question: " + instance_dict["question"] + "\n"

    if dataset == "knowledge_crosswords":
        prompt = prompt
    elif dataset == "hellaswag":
        prompt = "Please choose an option that best completes the sentence.\n" + prompt
    else:
        prompt = "Please choose an option that best answers the question.\n" + prompt

    for key in instance_dict["choices"].keys():
        prompt += (key + ": " + instance_dict["choices"][key] + "\n")

    prompt += "The answer is"

    # the format of Knowledge Crosswords is more complex and neccesitates an in-context example
    if dataset == "knowledge_crosswords":
        prompt = ICL_PROMPT + "\n" + prompt

    # print(prompt)

    return prompt

def multiple_choice_answer_parsing(instance_dict, output_text):

    # print(output_text)
    # print("-----")

    # directly answer
    for key in instance_dict["choices"].keys():
        if key in output_text[:5]:
            return key
    # "The answer is ."
    for key in instance_dict["choices"].keys():
        if key in output_text[-5:]:
            return key
    # answer text exact match
    for key in instance_dict["choices"].keys():
        if instance_dict["choices"][key].lower() in output_text.lower():
            return key
    return "Z" # so that it is absolutely incorrect

# for objective 3: reward models and objective 4: human interests we employ chat templates for conversation-like generation
def batch_generate_chat_template(model, tokenizer, prompts, gpu_id, batch_size=10, max_new_tokens=512):
    outputs = []
    # batch_size argument is useless here, sequential generation is necessary
    for prompt in tqdm(prompts):
        chat = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output = model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens, do_sample=False)
        outputs.append(tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True).strip())
    # print(outputs[-1])
    return outputs

# for objective 1: single task and objective 2: multi-task domains we directly generate
def batch_generate(model, tokenizer, prompts, gpu_id, batch_size=10, max_new_tokens=10,
                   return_details=False, batch_journal=None):
    num_batches = math.ceil(len(prompts) / batch_size)
    outputs = []
    # left padding + explicit attention mask: with right padding and pad==eos the mask
    # cannot be inferred, and pad tokens pollute generation for shorter prompts
    tokenizer.padding_side = "left"
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(f"cuda:{gpu_id}")
        if batch_journal is not None:
            batch_journal("started", start_idx, end_idx, None)
        output = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens, do_sample = False)

        for j in range(len(output)):
            continuation = output[j][enc.input_ids.shape[1]:]
            if return_details:
                eos = getattr(model.generation_config, "eos_token_id", None)
                if eos is None:
                    eos = tokenizer.eos_token_id
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                tokens = continuation.tolist()
                eos_index = next((k for k, token in enumerate(tokens)
                                  if token in eos_ids), None)
                generated = tokens[:eos_index + 1] if eos_index is not None else tokens
                reason = ("eos" if eos_index is not None else
                          "length" if len(tokens) >= max_new_tokens else "unknown")
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                outputs.append({"prompt": batch_prompts[j], "raw": text,
                                "generated_tokens": len(generated),
                                "stop_reason": reason})
            else:
                text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
                outputs.append(text)

        if batch_journal is not None:
            batch_journal("completed", start_idx, end_idx,
                          outputs[start_idx:end_idx])
        del enc, output
        torch.cuda.empty_cache()

    return outputs

def _strict_evaluate(model_path, eval_type, dataset, gpu_id, base_model,
                     split, subset_size, subset_start, result_dir, protocol="strict_v1",
                     subset_indices=None):
    """Versioned local evaluation; legacy artifacts are never overwritten."""
    from eval_protocol import (STRICT_EXACT_DATASETS, PROTOCOL_METADATA,
                               instruction_suffix)

    if split not in ("dev", "test"):
        raise ValueError("unknown split")
    if subset_indices is not None:
        if (split != "test" or protocol != "racer_eval_v1" or
                eval_type == "multitask" or subset_size is not None or
                subset_start is not None or not isinstance(subset_indices, list) or
                not subset_indices or any(type(i) is not int for i in subset_indices) or
                any(a >= b for a, b in zip(subset_indices, subset_indices[1:]))):
            raise ValueError("indexed subset requires ascending test indices in racer_eval_v1")
    if eval_type not in ("exact_match", "multiple_choice", "multitask"):
        raise ValueError("strict_v1 supports MC, numeric/boolean exact match and domains only")
    if eval_type == "exact_match" and dataset not in STRICT_EXACT_DATASETS:
        raise ValueError("strict exact-match task has no frozen scoring rule")
    base = Path(base_model)
    adapter = Path(model_path)
    if not base.is_dir() or not adapter.is_dir():
        raise ValueError("strict_v1 requires local base and adapter directories")
    if result_dir is None:
        raise ValueError("strict_v1 requires a new result_dir")
    destination = Path(result_dir)
    if destination.exists():
        raise FileExistsError(destination)

    data_root = Path(__file__).resolve().parent / "data" / "eval"
    if eval_type == "multitask":
        children = multitask_domain_dataset_dict[dataset]
        # Validate every child before creating output or loading a model.
        for child in children:
            with (data_root / (child + ".json")).open() as handle:
                rows = json.load(handle)[split]
            slice_bounds(len(rows), subset_size, subset_start, protocol)
        destination.mkdir(parents=True, exist_ok=False)
        scores = []
        for child in children:
            kind = "exact_match" if child in STRICT_EXACT_DATASETS else "multiple_choice"
            scores.append(_strict_evaluate(
                model_path, kind, child, gpu_id, base_model, split,
                subset_size, subset_start, destination / child, protocol))
        score = harmonic_mean(scores, protocol)
        with (destination / "evaluation.json").open("x") as handle:
            json.dump({"scoring_version": protocol, "split": split,
                       "aggregation_version": "harmonic_v1",
                       "subset_start": subset_start, "subset_size": subset_size,
                       "per_dataset_scores": dict(zip(children, scores)),
                       "score": score}, handle, indent=2)
        return score

    with (data_root / (dataset + ".json")).open() as handle:
        data = json.load(handle)
    if subset_indices is None:
        start, stop = slice_bounds(len(data[split]), subset_size, subset_start, protocol)
        indices = list(range(start, stop))
    else:
        if subset_indices[0] < 0 or subset_indices[-1] >= len(data[split]):
            raise ValueError("indexed test subset is outside the split")
        indices = subset_indices
        start = stop = None
    rows = [data[split][index] for index in indices]
    global ICL_PROMPT
    ICL_PROMPT = data.get("icl_prompt")
    suffix = instruction_suffix(dataset, protocol) if eval_type == "exact_match" else ""
    prompts = [multiple_choice_prompt(row, dataset) if eval_type == "multiple_choice"
               else row["question"] + suffix for row in rows]
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {"scoring_version": protocol, "protocol_metadata": PROTOCOL_METADATA[protocol],
                "split_protocol_version": "slice_v1",
                "aggregation_version": "mean_v1", "dataset": dataset,
                "eval_type": eval_type, "split": split, "subset_start": start,
                "subset_stop": stop, "subset_indices": subset_indices,
                "base_model": str(base.resolve()),
                "adapter": str(adapter.resolve()), "n_samples": len(rows),
                "instruction_suffix": suffix,
                "do_sample": False, "max_new_tokens": 200 if dataset == "gsm8k" else 10,
                "batch_size": 5 if dataset.startswith("nlgraph") else 10}
    local_model = local_tokenizer = None
    try:
        local_model = AutoModelForCausalLM.from_pretrained(
            str(base), torch_dtype=torch.float16, local_files_only=True)
        local_model = PeftModel.from_pretrained(local_model, str(adapter), local_files_only=True)
        local_model = local_model.merge_and_unload().to(f"cuda:{gpu_id}")
        local_tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        local_tokenizer.pad_token = local_tokenizer.eos_token
        journal_path = destination / "attempt.jsonl"
        def batch_journal(state, first, last, batch_outputs):
            entry = {"state": state, "indices": indices[first:last]}
            if batch_outputs is not None:
                entry["generated_count"] = len(batch_outputs)
                entry["generated_tokens"] = [item["generated_tokens"]
                                             for item in batch_outputs]
            with journal_path.open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(entry, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())

        outputs = batch_generate(local_model, local_tokenizer, prompts, gpu_id,
                                 batch_size=metadata["batch_size"],
                                 max_new_tokens=metadata["max_new_tokens"],
                                 return_details=protocol == "racer_eval_v1",
                                 **({"batch_journal": batch_journal}
                                    if protocol == "racer_eval_v1" else {}))
        if len(outputs) != len(rows):
            raise RuntimeError("generation count does not match requested samples")
        if protocol == "racer_eval_v1" and outputs and not isinstance(outputs[0], dict):
            raise RuntimeError("racer_eval_v1 requires per-sample generation details")
        details = outputs if protocol == "racer_eval_v1" else [None] * len(outputs)
        records = []
        for index, row, prompt, output, detail in zip(indices, rows, prompts,
                                                     outputs, details):
            raw = output["raw"] if detail is not None else output
            if detail is not None:
                if detail.get("prompt") != prompt or not isinstance(raw, str):
                    raise RuntimeError("generation detail bookkeeping mismatch")
            parsed = (parse_mc(row, raw, protocol) if eval_type == "multiple_choice"
                      else parse_exact_answer(dataset, raw, protocol))
            value = (float(parsed == row["answer"]) if eval_type == "multiple_choice"
                     else score_exact(dataset, row["answer"], raw, protocol, split))
            payload = json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
            record = {"index": index, "question_sha256": hashlib.sha256(payload).hexdigest(),
                      "question_text_sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "raw_generation": raw, "parsed_answer": parsed,
                      "gold": row["answer"], "score": value}
            if detail is not None:
                record["generated_tokens"] = detail.get("generated_tokens")
                record["stop_reason"] = detail.get("stop_reason")
            records.append(record)
        score = sum(record["score"] for record in records) / len(records)
        with (destination / "evaluation.json").open("x") as handle:
            json.dump({**metadata, "records": records, "score": score}, handle,
                      indent=2, ensure_ascii=False)
        return score
    finally:
        del local_model, local_tokenizer
        torch.cuda.empty_cache()


# given a model, evaluate it on the utility function and return the scalar value
def evaluate(model_path, eval_type, dataset, gpu_id, base_model = "google/gemma-7b-it", save_dev_flag = False, only_one_or_two = None, skip_flag = False, subset_size = None, subset_start = None, protocol="legacy", result_dir=None):

    if skip_flag:
        return None
    if protocol in ("strict_v1", "racer_eval_v1"):
        if only_one_or_two is not None or ONLY_ONE_OR_TWO is not None:
            raise ValueError("strict_v1 does not support a single domain-component override")
        return _strict_evaluate(model_path, eval_type, dataset, gpu_id, base_model,
                                "dev", subset_size, subset_start, result_dir, protocol)
    if protocol != "legacy":
        raise ValueError("unknown evaluation protocol")

    global model
    global tokenizer
    only_one_or_two = ONLY_ONE_OR_TWO
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
        # transformers 4.46 load_adapter silently drops LoRA weights whose keys carry the
        # base_model.model.* prefix; load via peft and merge so the adapter is always applied
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        model.to(f"cuda:{gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except:
        try:  # 先用本地缓存离线重试基模（HF 网络抖动时 from_pretrained 会抛错, 勿跳进 adapter 回退陷阱）
            del model
            del tokenizer
        except NameError:
            pass
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16,
                                                     local_files_only=True)
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        model.to(f"cuda:{gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # prompt = "What is the capital of France? Answer:"
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # output = model.generate(input_ids, max_new_tokens=10)
    # print(tokenizer.decode(output[0], skip_special_tokens=True))

    # objective 4: human interests
    if eval_type == "human":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"][(subset_start or 0):((subset_start or 0) + subset_size) if subset_size else None]
        scores = []

        # hard-defined batch_size for human interests objective
        BATCH_SIZE = 1

        prompts = []
        for obj in eval_data:
            prompts.append(obj["prompt"])
        
        outputs = batch_generate_chat_template(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE, max_new_tokens = 512)

        for i in tqdm(range(len(prompts))):
            scores.append(gemini_eval(prompts[i], outputs[i]))
            if scores[-1] == None:
                # format error, lowest score assigned
                scores[-1] = 1
        
        if save_dev_flag:
            with open(model_path + "/scores_dev.json", "w") as f:
                json.dump(scores, f)
        # utility function value is the average of Gemini-as-a-judge ratings across all prompts
        return sum(scores) / len(scores)

    # objective 3: reward models
    if eval_type in ["rm_default", "rm_concise", "rm_verbose", "rm_reverse"]:
        # rm_default: optimizing for default reward model, it's like alignment
        # rm_concise: optimizing for an average of reward model score and conciseness score
        # rm_verbose: optimizing for an average of reward model score and verbosity score
        # rm_reverse: optimizing for the reverse reward model scores, as a dual-use risk investigation

        try:
            assert dataset == "rm"
        except:
            warnings.warn("Reward modeling evaluation should be done on the reward modeling dataset. We provide by default data/eval/rm.json for this purpose.")
            warnings.warn("If you are bringing your own dataset, follow the format in data/eval/rm.json.")

        val_data = json.load(open("data/eval/" + dataset + ".json"))["dev"][(subset_start or 0):((subset_start or 0) + subset_size) if subset_size else None]

        # hard-defined batch_size for reward modeling objective, reduce if OOM
        BATCH_SIZE = 10

        prompts = []
        for obj in val_data:
            prompts.append(obj["prompt"])
        
        # max_new_tokens is set to 200 to ensure the fair calculation of concise/verbose percentile scores
        outputs = batch_generate(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE, max_new_tokens = 200)

        del model
        del tokenizer
        torch.cuda.empty_cache()

        rm_mode = None
        for mode in ["default", "concise", "verbose", "reverse"]:
            if mode in eval_type:
                rm_mode = mode
                break

        pairs = []
        assert len(prompts) == len(outputs)
        for i in range(len(prompts)):
            pairs.append(
                [
                    {"role": "user", "content": prompts[i]},
                    {"role": "assistant", "content": outputs[i]}
                ]
            )
        
        import reward_modeling
        scores_list = reward_modeling.get_reward_scores(pairs, gpu_id, rm_mode)
        if save_dev_flag:
            with open(model_path + "/scores_dev.json", "w") as f:
                json.dump(scores_list, f)
        # utility function value is the average of reward model scores across all prompts
        return sum(scores_list) / len(scores_list)

    # task 2: multi-task domains
    elif eval_type == "multitask": # medical, legal, science, culture
        per_dataset_scores = []
        eval_datasets = multitask_domain_dataset_dict[dataset][:2]
        for eval_dataset in eval_datasets:
            if eval_dataset in ["nlgraph", "gsm8k", "xstreet_ar", "xstreet_es"]:
                per_dataset_scores.append(evaluate(model_path, "exact_match", eval_dataset, gpu_id, base_model = base_model, save_dev_flag = True, subset_size = subset_size))
            else:
                per_dataset_scores.append(evaluate(model_path, "multiple_choice", eval_dataset, gpu_id, base_model = base_model, save_dev_flag = True, subset_size = subset_size))
        assert len(per_dataset_scores) == 2
        if sum(per_dataset_scores) == 0:
            per_dataset_scores = [0.01, 0.01] # dummy scores
        harmonic_mean = 2 * per_dataset_scores[0] * per_dataset_scores[1] / (per_dataset_scores[0] + per_dataset_scores[1])
        if only_one_or_two == "one":
            return per_dataset_scores[0]
        elif only_one_or_two == "two":
            return per_dataset_scores[1]
        # utility function value is the harmonic mean of the two scores on two datasets
        return harmonic_mean

    # task 1: single task, multiple choice questions
    elif eval_type == "multiple_choice":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"][(subset_start or 0):((subset_start or 0) + subset_size) if subset_size else None]
        golds = []
        preds = []
        global ICL_PROMPT
        try:
            # in case an ICL prompt is provided for datasets such as Knowledge Crosswords
            # you can provide your own ICL prompt in the dataset json file following knowledge_crosswords.json
            ICL_PROMPT = json.load(open("data/eval/" + dataset + ".json"))["icl_prompt"]
        except:
            pass

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)
        
        # hard-defined batch_size for multiple choice questions, reduce if OOM
        BATCH_SIZE = 10

        # change max_new_tokens to larger values for intermediate reasoning
        outputs = batch_generate(model, tokenizer, prompts, gpu_id, batch_size = BATCH_SIZE, max_new_tokens = 10)

        for question, output in zip(eval_data, outputs):
            # optionally uncomment the following lines to re-generate for multiple-choice questions that didn't get answered within 10 tokens
            # if multiple_choice_answer_parsing(question, output) == "Z":
            #     output = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 100)
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))

        if save_dev_flag:
            with open(model_path + "/golds_dev.json", "w") as f:
                json.dump(golds, f)
            with open(model_path + "/preds_dev.json", "w") as f:
                json.dump(preds, f)
        
        # print(golds)
        # print(preds)
        # utility function value is the accuracy score of the model on the multiple choice questions
        return accuracy_score(golds, preds)

    # task 1: single task, exact match questions
    elif eval_type == "exact_match":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"][(subset_start or 0):((subset_start or 0) + subset_size) if subset_size else None]
        scores = [] # could be 0/1 binary, could be continuous scores

        prompts = []
        for question in eval_data:
            prompts.append(question["question"])

        # hard-defined batch_size for exact match questions, reduce if OOM
        BATCH_SIZE = 10
        if dataset == "nlgraph" or dataset == "nlgraph_mini":
            BATCH_SIZE = 5
        
        MAX_NEW_TOKENS = 10
        # math reasoning datasets require more tokens for reasoning
        if dataset == "gsm8k":
            MAX_NEW_TOKENS = 200

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = MAX_NEW_TOKENS, batch_size=BATCH_SIZE)

        # retain only the last 5 tokens for number disambiguation
        if dataset == "gsm8k":
            outputs = [" ".join(output.split(" ")[-5:]) for output in outputs]

        # exact match evaluation
        for question, output in zip(eval_data, outputs):
            if question["answer"] in output:
                scores.append(1)
                time.sleep(0.2)
            else:
                scores.append(0)
            # print(question["answer"], output)
            # print("---------------")

        if save_dev_flag:
            with open(model_path + "/scores_dev.json", "w") as f:
                json.dump(scores, f)

        # utility function value is the accuracy score of the model on the exact match questions
        return sum(scores) / len(scores)

    # task 1: single task, RealToxicityPrompts special case
    elif eval_type == "external_api":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"][(subset_start or 0):((subset_start or 0) + subset_size) if subset_size else None]
        scores = [] # could be 0/1 binary, could be continuous scores

        prompts = []
        for instance in eval_data:
            prompts.append(instance["prompt"])

        BATCH_SIZE = 10
        MAX_NEW_TOKENS = 20

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = MAX_NEW_TOKENS, batch_size=BATCH_SIZE)

        if dataset == "realtoxicityprompts" or dataset == "realtoxicityprompts_mini":
            for output in tqdm(outputs):
                if len(output) < 10:
                    scores.append(0.5) # too short
                    continue
                try:
                    # the higher the better, so 1 - toxicity score
                    scores.append(1 - get_toxicity_score(output))
                except:
                    scores.append(0.5) # error on the API side
                time.sleep(0.9)
        
        if save_dev_flag:
            with open(model_path + "/scores_dev.json", "w") as f:
                json.dump(scores, f)

        # utility function value is the average anti-toxicity score of the model on the RealToxicityPrompts dataset
        return sum(scores) / len(scores)

    # task 1: single task, AbstainQA special case
    elif eval_type == "AbstainQA":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"][(subset_start or 0):((subset_start or 0) + subset_size) if subset_size else None]
        golds = []
        preds = []

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)
        
        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 10)

        correct_flags = []
        abstain_flags = []

        for question, output in zip(eval_data, outputs):
            # if multiple_choice_answer_parsing(question, output) == "Z":
            #     output = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 100)
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))
            if golds[-1] == preds[-1]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)
            
        # self-reflection after answering to get abstain decisions
        new_prompts = [prompt + "\nProposed answer: " + output + "\nIs the proposed answer true or false? Directly answer with true or false." for prompt, output in zip(prompts, outputs)]

        outputs = batch_generate(model, tokenizer, new_prompts, gpu_id, max_new_tokens = 10)

        for output in outputs:
            # print(output)
            if "false" in output.lower():
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)
            
        if save_dev_flag:
            with open(model_path + "/golds_dev.json", "w") as f:
                json.dump(correct_flags, f)
            with open(model_path + "/preds_dev.json", "w") as f:
                json.dump([1-flag for flag in abstain_flags] , f)
        
        # print(golds)
        # print(preds)
        # utility function value is the effective reliability of the model on the AbstainQA dataset
        return get_effective_reliability(correct_flags, abstain_flags)

# evaluation on the test set, similar to the dev set evaluation, but kept seperate in case the test eval might be dratiscally different from dev in generalization settings
def evaluate_test(model_path, eval_type, dataset, gpu_id, base_model = "google/gemma-7b-it", only_one_or_two = None, obj4_save_generation=False, protocol="legacy", result_dir=None, subset_indices=None):

    if protocol in ("strict_v1", "racer_eval_v1"):
        if only_one_or_two is not None or ONLY_ONE_OR_TWO is not None:
            raise ValueError("strict_v1 does not support a single domain-component override")
        return _strict_evaluate(model_path, eval_type, dataset, gpu_id, base_model,
                                "test", None, None, result_dir, protocol,
                                subset_indices=subset_indices)
    if subset_indices is not None:
        raise ValueError("indexed test subset requires racer_eval_v1")
    if protocol != "legacy":
        raise ValueError("unknown evaluation protocol")
    global model
    global tokenizer

    only_one_or_two = ONLY_ONE_OR_TWO

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
        # transformers 4.46 load_adapter silently drops LoRA weights whose keys carry the
        # base_model.model.* prefix; load via peft and merge so the adapter is always applied
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        model.to(f"cuda:{gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except:
        try:  # 先用本地缓存离线重试基模（HF 网络抖动时 from_pretrained 会抛错, 勿跳进 adapter 回退陷阱）
            del model
            del tokenizer
        except NameError:
            pass
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16,
                                                     local_files_only=True)
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        model.to(f"cuda:{gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # prompt = "What is the capital of France? Answer:"
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # output = model.generate(input_ids, max_new_tokens=10)
    # print(tokenizer.decode(output[0], skip_special_tokens=True))

    # objective 4: human interests
    if eval_type == "human":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        scores = []

        BATCH_SIZE = 1

        prompts = []
        for obj in eval_data:
            prompts.append(obj["prompt"])
        
        outputs = batch_generate_chat_template(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE, max_new_tokens = 512)

        for i in tqdm(range(len(prompts))):
            scores.append(gemini_eval(prompts[i], outputs[i]))
            if scores[-1] == None:
                scores[-1] = 1

        if obj4_save_generation:
            save_name = model_path.split("/")[-1] + "_" + eval_type + "_" + dataset
            with open("data/outputs/" + save_name + ".json", "w") as f:
                json.dump({"outputs": outputs}, f, indent=4)
        
        with open(model_path + "/scores.json", "w") as f:
            json.dump(scores, f)
        return sum(scores) / len(scores)

    # objective 3: reward models
    if eval_type in ["rm_default", "rm_concise", "rm_verbose", "rm_reverse"]:
        assert dataset == "rm"
        val_data = json.load(open("data/eval/" + dataset + ".json"))["test"]

        # hard-defined batch_size for reward modeling objective, reduce if OOM
        BATCH_SIZE = 10

        prompts = []
        for obj in val_data:
            prompts.append(obj["prompt"])
        
        outputs = batch_generate(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE, max_new_tokens = 200)

        del model
        del tokenizer
        torch.cuda.empty_cache()

        rm_mode = None
        for mode in ["default", "concise", "verbose", "reverse"]:
            if mode in eval_type:
                rm_mode = mode
                break

        pairs = []
        assert len(prompts) == len(outputs)
        for i in range(len(prompts)):
            pairs.append(
                [
                    {"role": "user", "content": prompts[i]},
                    {"role": "assistant", "content": outputs[i]}
                ]
            )
        
        import reward_modeling
        scores_list = reward_modeling.get_reward_scores(pairs, gpu_id, rm_mode)
        with open(model_path + "/scores.json", "w") as f:
            json.dump(scores_list, f)
        return sum(scores_list) / len(scores_list)

    # task 2: multi-task domains
    elif eval_type == "multitask":
        per_dataset_scores = []
        eval_datasets = multitask_domain_dataset_dict[dataset]
        for eval_dataset in eval_datasets:
            # default multi-task evaluation sets are all MC
            per_dataset_scores.append(evaluate_test(model_path, "multiple_choice", eval_dataset, gpu_id, base_model = base_model))
        assert len(per_dataset_scores) == 2
        if sum(per_dataset_scores) == 0:
            per_dataset_scores = [0.01, 0.01] # dummy scores
        harmonic_mean = 2 * per_dataset_scores[0] * per_dataset_scores[1] / (per_dataset_scores[0] + per_dataset_scores[1])
        if only_one_or_two == "one":
            return per_dataset_scores[0]
        elif only_one_or_two == "two":
            return per_dataset_scores[1]
        return harmonic_mean

    # task 1: single task, multiple choice questions
    elif eval_type == "multiple_choice":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        golds = []
        preds = []
        global ICL_PROMPT
        try:
            ICL_PROMPT = json.load(open("data/eval/" + dataset + ".json"))["icl_prompt"]
        except:
            pass

        # hard-defined batch_size for multiple choice questions, reduce if OOM
        BATCH_SIZE = 10
        # if dataset == "medqa" or dataset == "hearsay" or dataset == "citation_prediction_classification":
        #     BATCH_SIZE = 2

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)
        
        outputs = batch_generate(model, tokenizer, prompts, gpu_id, batch_size = BATCH_SIZE, max_new_tokens = 10)

        for question, output in zip(eval_data, outputs):
            # if multiple_choice_answer_parsing(question, output) == "Z":
            #     output = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 100)
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))
            # print(output)
            # print("----------")
        
        # print(golds)
        # print(preds)

        # save golds and preds for later ensemble
        with open(model_path + "/golds.json", "w") as f:
            json.dump(golds, f)
        with open(model_path + "/preds.json", "w") as f:
            json.dump(preds, f)

        return accuracy_score(golds, preds)

    # task 1: single task, exact match questions
    elif eval_type == "exact_match":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        scores = [] # could be 0/1 binary, could be continuous scores

        prompts = []
        for question in eval_data:
            prompts.append(question["question"])

        # hard-defined batch_size for exact match questions, reduce if OOM
        BATCH_SIZE = 10
        if dataset == "nlgraph" or dataset == "nlgraph_mini":
            BATCH_SIZE = 5

        MAX_NEW_TOKENS = 10
        # math reasoning datasets require more tokens for reasoning
        if dataset == "gsm8k":
            MAX_NEW_TOKENS = 200

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = MAX_NEW_TOKENS, batch_size=BATCH_SIZE)

        if dataset == "gsm8k":
            outputs = [" ".join(output.split(" ")[-10:]) for output in outputs]

        for question, output in zip(eval_data, outputs):
            if question["answer"] in output:
                scores.append(1)
            else:
                scores.append(0)

        with open(model_path + "/scores.json", "w") as f:
            json.dump(scores, f)

        return sum(scores) / len(scores)

    # task 1: single task, RealToxicityPrompts special case
    elif eval_type == "external_api":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        scores = [] # could be 0/1 binary, could be continuous scores

        prompts = []
        for instance in eval_data:
            prompts.append(instance["prompt"])

        # hard-defined batch_size for RealToxicityPrompts, reduce if OOM
        BATCH_SIZE = 10

        MAX_NEW_TOKENS = 20

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = MAX_NEW_TOKENS, batch_size=BATCH_SIZE)

        if dataset == "realtoxicityprompts" or dataset == "realtoxicityprompts_mini":
            for output in tqdm(outputs):
                if len(output) < 10:
                    scores.append(0.5) # too short
                    continue
                try:
                    scores.append(1 - get_toxicity_score(output))
                    # if get_toxicity_score(output) >= 0.5: # maybe, maybe no cutoff since its calibrated?
                    #     scores.append(0) # toxic, to maintain the higher the better scores
                    # else:
                    #     scores.append(1)
                except:
                    scores.append(0) # error on the API side
                time.sleep(0.9)

        with open(model_path + "/scores.json", "w") as f:
            json.dump(scores, f)
        
        print(model_path, "finished!")

        return sum(scores) / len(scores)

    # task 1: single task, AbstainQA special case
    elif eval_type == "AbstainQA":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        golds = []
        preds = []
        # global ICL_PROMPT
        # try:
        #     ICL_PROMPT = json.load(open("data/eval/" + dataset + ".json"))["icl_prompt"]
        # except:
        #     pass

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)
        
        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 10)

        correct_flags = []
        abstain_flags = []

        for question, output in zip(eval_data, outputs):
            # if multiple_choice_answer_parsing(question, output) == "Z":
            #     output = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 100)
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))
            if golds[-1] == preds[-1]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)
            
        new_prompts = [prompt + "\nProposed answer: " + output + "\nIs the proposed answer true or false? Directly answer with true or false." for prompt, output in zip(prompts, outputs)]

        outputs = batch_generate(model, tokenizer, new_prompts, gpu_id, max_new_tokens = 10)

        for output in outputs:
            # print(output)
            if "false" in output.lower():
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)

        with open(model_path + "/golds.json", "w") as f:
            json.dump(correct_flags, f)
        with open(model_path + "/preds.json", "w") as f:
            json.dump([1-flag for flag in abstain_flags], f)
        
        # print(golds)
        # print(preds)
        return get_effective_reliability(correct_flags, abstain_flags)

# some sanity check examples of evaluation

# result_test = evaluate_test("initial_experts/lima", "AbstainQA", "mmlu", 0)
# print(result_test)

# result = evaluate("initial_experts/lima", "rm_default", "rm", 0)
# print(result)

# result = evaluate("initial_experts/lima", "multitask", "legal", 0)
# print(result)

# result = evaluate("initial_experts/lima", "multiple_choice", "mmlu", 0)
# print(result)
# result_test = evaluate_test("initial_experts/lima", "multiple_choice", "mmlu", 0)
# print(result_test)

# result = evaluate("initial_experts/lima", "human", "human_phd_application", 0)
# print(result)
# result_test = evaluate_test("initial_experts/lima", "human", "human_phd_application", 0)
# print(result_test)