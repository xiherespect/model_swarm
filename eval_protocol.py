"""Dependency-free evaluation protocols; importing this module has no side effects.

``legacy`` intentionally preserves historical mistakes, including MC substring
matching, numeric substring scoring, gsm8k's dev/test window mismatch, and the
0.01 dummy harmonic mean. It is not a corrected metric.

``strict_v1`` freezes conservative MC and numeric/boolean answer grammars. Only
nlgraph, nlgraph_mini and gsm8k have strict exact-answer support. Other exact
match tasks retain the historical task-specific substring rule, NOT a claim of
strict scoring; callers requiring full support must gate on STRICT_EXACT_DATASETS.
``racer_eval_v1`` keeps the same grammars/slices/aggregation and additionally
appends a frozen final-answer instruction to supported exact-match prompts
(dev and test alike); it must not be mixed into strict_v1 archives.
No model, network, file IO, reward model or cloud dependency belongs here.
"""

import math
import re
import unicodedata
from decimal import Decimal


PROTOCOLS = frozenset({"legacy", "strict_v1", "racer_eval_v1"})
STRICT_EXACT_DATASETS = frozenset({"nlgraph", "nlgraph_mini", "gsm8k"})
# racer_eval_v1 shares the strict grammar/slicing/aggregation and only adds a
# frozen final-answer instruction for supported exact-match prompts. It is a
# new protocol identity, never silently merged into strict_v1 archives.
_STRICT_LIKE = frozenset({"strict_v1", "racer_eval_v1"})
INSTRUCTION_SUFFIX_VERSION = "racer_instructions_v1"
# Frozen final-answer instructions appended verbatim to supported exact-match
# prompts under racer_eval_v1, on both dev and test. Multiple-choice prompts are
# unchanged. Values are frozen before any formal run; the dev gate may only
# replace the whole map (new version string), never edit scores afterwards.
RACER_EVAL_INSTRUCTIONS = {
    "nlgraph": " Answer with only the final answer: a number, or Yes or No. Do not explain.",
    "nlgraph_mini": " Answer with only the final answer: a number, or Yes or No. Do not explain.",
    "gsm8k": " Work through the problem, then end your reply with the final numeric answer after #### .",
}
PROTOCOL_METADATA = {
    "legacy": {
        "mc": "legacy_substring_v0",
        "exact": "legacy_substring_split_windows_v0",
        "slice": "legacy_python_slice_zero_is_none_v0",
        "aggregation": "legacy_pair_harmonic_zero_dummy_v0",
    },
    "strict_v1": {
        "mc": "explicit_label_or_unique_complete_option_v1",
        "exact": "partial_numeric_and_nlgraph_boolean_v1",
        "exact_supported_datasets": tuple(sorted(STRICT_EXACT_DATASETS)),
        "exact_other_datasets": "legacy_substring_unchanged",
        "slice": "nonempty_in_bounds_v1",
        "aggregation": "nonnegative_pair_harmonic_zero_is_zero_v1",
    },
    "racer_eval_v1": {
        "mc": "explicit_label_or_unique_complete_option_v1",
        "exact": "partial_numeric_and_nlgraph_boolean_v1",
        "exact_supported_datasets": tuple(sorted(STRICT_EXACT_DATASETS)),
        "exact_other_datasets": "legacy_substring_unchanged",
        "slice": "nonempty_in_bounds_v1",
        "aggregation": "nonnegative_pair_harmonic_zero_is_zero_v1",
        "instructions": INSTRUCTION_SUFFIX_VERSION,
    },
}


def _check_protocol(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError("unknown evaluation protocol: {!r}".format(protocol))


def instruction_suffix(dataset, protocol):
    """Frozen final-answer instruction for a dataset, or ``''`` when none.

    Only racer_eval_v1 appends instructions, and only for exact-match datasets
    with a frozen entry; strict_v1 and legacy prompts stay byte-identical.
    """
    _check_protocol(protocol)
    if protocol != "racer_eval_v1":
        return ""
    return RACER_EVAL_INSTRUCTIONS.get(dataset, "")


def _text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def parse_mc(instance, output, protocol="legacy"):
    """Return a choice label or ``'Z'`` for invalid/ambiguous answers.

    Legacy follows evaluate.py's three ordered loops exactly. Strict accepts a
    bare label, a punctuated leading label, labels in explicit answer sentences,
    or one complete choice text with token boundaries (case/space normalized).
    All evidence must agree, regardless of choice insertion order. A word's
    first letter is never evidence. Mentioning conflicting labels/texts rejects
    the output rather than choosing a label based on dictionary order.
    """
    _check_protocol(protocol)
    choices = instance["choices"]
    if protocol == "legacy":
        for key in choices.keys():
            if key in output[:5]:
                return key
        for key in choices.keys():
            if key in output[-5:]:
                return key
        for key in choices.keys():
            if choices[key].lower() in output.lower():
                return key
        return "Z"

    text = _text(output)
    if not text or not choices:
        return "Z"
    labels = "|".join(re.escape(key) for key in sorted(choices, key=lambda k: (-len(k), k)))
    evidence = set()
    bare = re.fullmatch(r"[\(\[]?\s*(" + labels + r")\s*[\)\]]?[.!:]?", text,
                        flags=re.IGNORECASE)
    if bare:
        evidence.update(key for key in choices if key.casefold() == bare[1].casefold())

    # A. / A: / (A) are explicit labels; 'Apple', 'A bird' are not.
    # Collect every punctuated label so a second label cannot be silently ignored.
    evidence.update(re.findall(r"(?<!\w)(" + labels + r")[.:](?=\s|$)", text))
    evidence.update(re.findall(r"[\(\[](" + labels + r")[\)\]]", text))

    cue = re.compile(r"\b(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is\b|[:=])"
                     r"|(?:i\s+)?(?:choose|select|pick)\b)\s*(?:option\s+)?",
                     re.IGNORECASE)
    # Keep the whole answer clause: 'answer is A or B' is not a unique answer.
    for match in cue.finditer(output):
        clause = output[match.end():].split("\n", 1)[0]
        found = re.findall(r"(?<!\w)(" + labels + r")(?!\w)", clause, re.IGNORECASE)
        evidence.update(key for key in choices
                        if any(key.casefold() == value.casefold() for value in found))

    normalized = text.casefold()
    for key, value in choices.items():
        option = _text(value).casefold()
        if option and re.search(r"(?<!\w)" + re.escape(option) + r"(?!\w)", normalized):
            evidence.add(key)
    # Unadorned labels are not positive evidence (e.g. the article 'A'), but
    # cannot contradict an otherwise recognized answer: '(A) or B' is ambiguous.
    mentioned = set(re.findall(r"(?<!\w)(" + labels + r")(?!\w)", text))
    if evidence and mentioned - evidence:
        return "Z"
    # A negated/tentative answer is not positive answer evidence. Do not
    # penalize a complete option whose literal wording itself begins with 'not'.
    exact_option = any(normalized.rstrip(".! ") == _text(value).casefold().rstrip(".! ")
                       for value in choices.values() if value)
    if not exact_option and (re.match(r"(?:not|neither|maybe|perhaps)\b", normalized)
                             or re.search(r"\banswer\s*(?:is\s*|[:=]\s*)(?:not|maybe|perhaps)\b",
                                          normalized)):
        return "Z"
    return next(iter(evidence)) if len(evidence) == 1 else "Z"


_NUMBER = re.compile(r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)")
_ANSWER_CUE = re.compile(
    r"(?:\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\b|[:=])\s*|####\s*)",
    re.IGNORECASE,
)
_BOX = re.compile(r"\\boxed\{([^{}]*)\}")


def _scalar(text, allow_boolean):
    """Parse a whole scalar, never search for a number inside prose."""
    text = _text(text).replace("\N{MINUS SIGN}", "-").strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    box = _BOX.fullmatch(text)
    if box:
        text = box[1].strip()
    if text.startswith("$") and text.endswith("$") and len(text) > 2:
        text = text[1:-1].strip()
    if allow_boolean and text.casefold() in {"yes", "no"}:
        return text.casefold()
    if not _NUMBER.fullmatch(text):
        return None
    number = Decimal(text.replace(",", ""))
    if number == 0:
        return "0"
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def parse_exact_answer(dataset, output, protocol="strict_v1"):
    """Return a canonical scalar or None for invalid strict exact answers.

    Strict supported tasks accept a whole numeric reply, or a complete scalar
    after ``answer is``, ``answer:``, ``answer=``, ``####`` or ``\\boxed{...}``.
    nlgraph variants additionally accept whole Yes/No replies. Earlier reasoning
    numbers are ignored ONLY with an explicit answer marker; conflicting markers,
    multiple numbers, trailing prose/units, unfinished answers and arbitrary gold
    substrings are rejected. Decimal handles signs, fractions written as decimals,
    and correctly grouped thousands without float rounding. No fractions,
    exponent notation, units or implicit 'last number' heuristic are supported.

    Legacy and unsupported strict datasets return output unchanged, indicating
    the historical text rule rather than fabricating numeric support. Legacy's
    gsm8k split-dependent truncation is done by score_exact, not by this helper.
    """
    _check_protocol(protocol)
    if protocol == "legacy" or dataset not in STRICT_EXACT_DATASETS:
        return output
    allow_boolean = dataset in {"nlgraph", "nlgraph_mini"}
    matches = list(_ANSWER_CUE.finditer(output))
    if matches:
        candidates = [output[match.end():matches[i + 1].start() if i + 1 < len(matches) else None]
                      for i, match in enumerate(matches)]
        prefix = output[:matches[0].start()]
        if _scalar(prefix, allow_boolean) is not None:
            candidates.insert(0, prefix)
        # Explicit boxed answers in earlier reasoning are answer evidence too.
        candidates.extend(match[1] for match in _BOX.finditer(prefix))
    else:
        boxes = list(_BOX.finditer(output))
        if boxes:
            # A boxed result may follow reasoning, but may not have a trailing
            # second answer or unfinished continuation.
            if output[boxes[-1].end():].strip(" \t\r\n."):
                return None
            candidates = [match[1] for match in boxes]
        else:
            candidates = [output]
    parsed = [_scalar(candidate, allow_boolean) for candidate in candidates]
    if not parsed or any(value is None for value in parsed) or len(set(parsed)) != 1:
        return None
    return parsed[0]


def score_exact(dataset, answer, output, protocol="legacy", split="dev"):
    """Score one raw generation. Returns float 0/1, preserving legacy bugs.

    The caller must pass raw output, not an already-truncated gsm8k generation.
    Strict supported tasks share exactly the same extraction on dev and test.
    Other datasets explicitly retain the historical substring task logic.
    """
    _check_protocol(protocol)
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")
    if protocol in _STRICT_LIKE and dataset in STRICT_EXACT_DATASETS:
        gold = _scalar(answer, dataset in {"nlgraph", "nlgraph_mini"})
        parsed = parse_exact_answer(dataset, output, protocol=protocol)
        return float(gold is not None and parsed is not None and gold == parsed)
    if dataset == "gsm8k":
        window = 5 if split == "dev" else 10
        output = " ".join(output.split(" ")[-window:])
    return float(answer in output)


def slice_bounds(length, subset_size=None, subset_start=None, protocol="legacy"):
    """Return actual ``(start, stop)`` indices for a unit-step slice.

    Legacy is exactly ``[(start or 0):((start or 0)+size) if size else None]``,
    normalized by Python's slice.indices; size=0 means unlimited, negative and
    empty slices are allowed. Strict requires integer (not bool), nonnegative
    start, positive size, nonempty data and no clipping of an explicit range.
    """
    _check_protocol(protocol)
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if protocol == "legacy":
        start = subset_start or 0
        stop = start + subset_size if subset_size else None
        start, stop, _ = slice(start, stop).indices(length)
        return start, stop
    for name, value in (("subset_start", subset_start), ("subset_size", subset_size)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(name + " must be an integer or None")
    start = 0 if subset_start is None else subset_start
    if length == 0 or start < 0 or start >= length:
        raise ValueError("slice start must select a nonempty in-bounds range")
    if subset_size is not None and subset_size <= 0:
        raise ValueError("subset_size must be positive")
    stop = length if subset_size is None else start + subset_size
    if stop > length:
        raise ValueError("requested slice exceeds dataset length")
    return start, stop


def harmonic_mean(scores, protocol="legacy"):
    """Pairwise domain harmonic mean, including the legacy zero dummy rule."""
    _check_protocol(protocol)
    scores = list(scores)
    if len(scores) != 2:
        raise ValueError("domain harmonic mean requires exactly two scores")
    a, b = scores
    if protocol == "legacy":
        if sum(scores) == 0:
            return 0.01
        return 2 * a * b / (a + b)
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("strict harmonic mean requires finite nonnegative scores")
    if a == 0 or b == 0:
        return 0.0
    low, high = sorted(scores)
    return low * (2 / (1 + low / high))


def stable_argmax(values):
    """Return the earliest index attaining the maximum; reject empty input."""
    values = list(values)
    if not values:
        raise ValueError("argmax requires at least one value")
    return max(range(len(values)), key=values.__getitem__)
