"""CPU-only golden tests, runnable using python -I -m unittest discover."""

import itertools
import math
from pathlib import Path
import subprocess
import sys
import unittest

# Isolated mode deliberately excludes cwd; never import evaluate (GPU/cloud).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import eval_protocol as protocol


class LegacyGoldenTests(unittest.TestCase):
    def setUp(self):
        self.instance = {"choices": {"A": "red", "B": "blue", "C": "green"}}

    def test_mc_historical_false_positive_word_initial(self):
        self.assertEqual(protocol.parse_mc(self.instance, "Apple pie"), "A")
        self.assertEqual(protocol.parse_mc(self.instance, "Blueberries"), "B")

    def test_mc_historical_last_five_characters(self):
        self.assertEqual(protocol.parse_mc(self.instance, "something long ends BAD"), "A")

    def test_mc_historical_option_substring(self):
        self.assertEqual(protocol.parse_mc(self.instance, "the infrared result"), "A")

    def test_mc_historical_dictionary_order_and_empty_text(self):
        first = {"choices": {"A": "red", "B": "blue"}}
        second = {"choices": {"B": "blue", "A": "red"}}
        self.assertEqual(protocol.parse_mc(first, "B A"), "A")
        self.assertEqual(protocol.parse_mc(second, "B A"), "B")
        self.assertEqual(protocol.parse_mc({"choices": {"A": ""}}, "nothing"), "A")
        self.assertEqual(protocol.parse_mc(self.instance, "nothing"), "Z")

    def test_mc_matches_frozen_original_function(self):
        # Frozen from pre-protocol evaluate.py; independent of later wiring edits.
        def original(instance_dict, output_text):
            for key in instance_dict["choices"].keys():
                if key in output_text[:5]:
                    return key
            for key in instance_dict["choices"].keys():
                if key in output_text[-5:]:
                    return key
            for key in instance_dict["choices"].keys():
                if instance_dict["choices"][key].lower() in output_text.lower():
                    return key
            return "Z"
        for permutation in itertools.permutations(self.instance["choices"].items()):
            instance = {"choices": dict(permutation)}
            for output in ["A", "(B)", "Apple pie", "B A", "infrared", "", "answer is C",
                           "xxxxxxxxxxxxx B A", "the color is BLUE", "something"]:
                with self.subTest(permutation=permutation, output=output):
                    self.assertEqual(protocol.parse_mc(instance, output), original(instance, output))

    def test_exact_numeric_substring_bug(self):
        for dataset in ("nlgraph", "nlgraph_mini", "gsm8k", "custom"):
            for split in ("dev", "test"):
                for output in ("12", "20", "-2", "2.5"):
                    self.assertEqual(protocol.score_exact(dataset, "2", output, split=split), 1.0)
        self.assertEqual(protocol.score_exact("nlgraph", "No", "Not a path"), 1.0)

    def test_gsm8k_historical_window_split_difference(self):
        output = "2 a b c d e f g"
        self.assertEqual(protocol.score_exact("gsm8k", "2", output, split="dev"), 0.0)
        self.assertEqual(protocol.score_exact("gsm8k", "2", output, split="test"), 1.0)
        self.assertEqual(protocol.score_exact("gsm8k", "2", "2 a b c d"), 1.0)
        self.assertEqual(protocol.score_exact("gsm8k", "2", "2 a b c d e"), 0.0)
        # Historical code splits literal spaces, not arbitrary whitespace.
        self.assertEqual(protocol.score_exact("gsm8k", "2", "2\na b c d e"), 1.0)
        self.assertEqual(protocol.score_exact("gsm8k", "2", "2     a"), 0.0)

    def test_non_numeric_task_rule_remains_case_sensitive(self):
        self.assertEqual(protocol.score_exact("xstreet_ar", "Hello", "Hello world"), 1.0)
        self.assertEqual(protocol.score_exact("xstreet_es", "Hello", "hello world"), 0.0)
        self.assertEqual(protocol.score_exact("custom", "", "anything"), 1.0)
        self.assertEqual(protocol.parse_exact_answer("gsm8k", "raw text", "legacy"), "raw text")

    def test_legacy_slice_matches_literal_original_exhaustively(self):
        for length, size, start in itertools.product(range(7), [None, -9, -2, 0, 1, 4, 10],
                                                     [None, -8, -1, 0, 1, 4, 10]):
            data = list(range(length))
            expected = data[(start or 0):((start or 0) + size) if size else None]
            actual_start, actual_stop = protocol.slice_bounds(length, size, start)
            self.assertEqual(data[actual_start:actual_stop], expected, (length, size, start))
        self.assertEqual(protocol.slice_bounds(10, 0, 3), (3, 10))
        self.assertEqual(protocol.slice_bounds(10, 3, 20), (10, 10))

    def test_legacy_harmonic_dummy_and_pair_formula(self):
        self.assertEqual(protocol.harmonic_mean([0, 0]), 0.01)
        self.assertEqual(protocol.harmonic_mean([0, 0.8]), 0.0)
        self.assertEqual(protocol.harmonic_mean([0.25, 0.75]), 0.375)
        # This is also what old sum==0 would do for negative inputs.
        self.assertEqual(protocol.harmonic_mean([-1, 1]), 0.01)


class StrictMCTests(unittest.TestCase):
    def setUp(self):
        self.instance = {"choices": {"A": "red fox", "B": "blue whale", "C": "green frog"}}

    def parse(self, output, instance=None):
        return protocol.parse_mc(instance or self.instance, output, protocol="strict_v1")

    def test_explicit_labels(self):
        for output, expected in [("A", "A"), ("(B)", "B"), ("[C]", "C"), (" B. ", "B"),
                                 ("A: explanation", "A"), ("(B) because it fits", "B")]:
            with self.subTest(output=output):
                self.assertEqual(self.parse(output), expected)

    def test_explicit_answer_sentences(self):
        for output, expected in [("The answer is B.", "B"), ("Answer: (C)", "C"),
                                 ("Final answer = A", "A"), ("I choose option B", "B"),
                                 ("Some reasoning. The final answer is C.", "C"),
                                 ("the answer is b", "B")]:
            with self.subTest(output=output):
                self.assertEqual(self.parse(output), expected)

    def test_unique_complete_choice_text(self):
        for output in ("blue whale", "The best animal is BLUE WHALE.", "blue\nwhale"):
            self.assertEqual(self.parse(output), "B")
        self.assertEqual(self.parse("blue whales"), "Z")
        self.assertEqual(self.parse("blue"), "Z")
        self.assertEqual(self.parse("a red foxhound"), "Z")

    def test_word_initial_is_not_a_label(self):
        for output in ("Apple", "Because", "Correct", "A bird", "A", ""):
            if output != "A":
                self.assertEqual(self.parse(output), "Z", output)

    def test_conflicting_labels_and_text_reject(self):
        for output in ("The answer is A or B", "(A) or (B)", "A. blue whale", "red fox or blue whale",
                       "The answer is A. The answer is B.", "A B", "Answer: A/B",
                       "A. or B.", "(A) or B", "The answer is a or b"):
            self.assertEqual(self.parse(output), "Z", output)

    def test_negative_or_tentative_label_not_an_answer(self):
        for output in ("The answer is not B", "Not blue whale", "Maybe (B)", "Perhaps B."):
            self.assertEqual(self.parse(output), "Z", output)
        self.assertEqual(self.parse("not enough data", {"choices": {"A": "not enough data", "B": "yes"}}), "A")

    def test_choice_permutations_do_not_select_by_order(self):
        for permutation in itertools.permutations(self.instance["choices"].items()):
            instance = {"choices": dict(permutation)}
            self.assertEqual(self.parse("The answer is B.", instance), "B")
            self.assertEqual(self.parse("red fox or blue whale", instance), "Z")
            self.assertEqual(self.parse("BLUE WHALE", instance), "B")

    def test_duplicate_and_overlapping_choice_text_reject(self):
        for choices in ({"A": "blue whale", "B": "blue whale"},
                        {"A": "whale", "B": "blue whale"}):
            self.assertEqual(self.parse("blue whale", {"choices": choices}), "Z")
        self.assertEqual(self.parse("nothing", {"choices": {"A": "", "B": "other"}}), "Z")


class StrictExactTests(unittest.TestCase):
    def score(self, gold, output, dataset="gsm8k", split="dev"):
        return protocol.score_exact(dataset, gold, output, protocol="strict_v1", split=split)

    def test_whole_number_not_substring(self):
        for output in ("12", "20", "-2", "2.5", "There are 2 and 12", "The answer is 12"):
            self.assertEqual(self.score("2", output), 0.0, output)
        self.assertEqual(self.score("2", "2"), 1.0)

    def test_signed_decimal_thousands_canonicalization(self):
        for gold, output, canonical in [("-2", "-2.00", "-2"), ("2", "+2.000", "2"),
                                        ("0.5", ".50", "0.5"), ("-.5", "−0.500", "-0.5"),
                                        ("1200", "1,200.00", "1200"),
                                        ("1234567.25", "1,234,567.250", "1234567.25"),
                                        ("0", "-0.00", "0"), ("002", "2", "2")]:
            with self.subTest(gold=gold, output=output):
                self.assertEqual(self.score(gold, output), 1.0)
                self.assertEqual(protocol.parse_exact_answer("gsm8k", output), canonical)

    def test_decimal_does_not_round_through_float_or_decimal_context(self):
        value = "123456789012345678901234567890123456789.1234567890123456789"
        self.assertEqual(self.score(value, value), 1.0)
        self.assertEqual(self.score(value, value[:-1] + "8"), 0.0)
        self.assertEqual(protocol.parse_exact_answer("gsm8k", value), value)

    def test_explicit_final_answer_over_reasoning_numbers(self):
        for output in ("We tried 2 and 20. The answer is 12.", "2 + 10 = 12\n#### 12",
                       "Initially 2; final answer: 12", "Reasoning: 2. \\boxed{12}",
                       "Answer = $12$", "The answer is **12**", "The answer is \\boxed{12}"):
            self.assertEqual(self.score("12", output), 1.0, output)
            self.assertEqual(self.score("2", output), 0.0, output)

    def test_ambiguous_malformed_or_truncated_answers_reject(self):
        for output in ("", "The answer is", "2 or 12", "2 and 2", "Final answer: 2 or 3",
                       "The answer is 2. The answer is 3.", "1,20", "1,000,00", "1 000",
                       "2/1", "2e0", "2 apples", "We calculate 2", "NaN", "Infinity",
                       "The answer is 2 because that works", "\\boxed{2} or 3", "\\boxed{2",
                       "1.2.3", "2..", "--2", "The answer is not 2",
                       "2. The answer is 3", "\\boxed{2}. The answer is 3"):
            with self.subTest(output=output):
                self.assertIsNone(protocol.parse_exact_answer("gsm8k", output))
                self.assertEqual(self.score("2", output), 0.0)

    def test_conflicting_boxes_reject(self):
        self.assertIsNone(protocol.parse_exact_answer("gsm8k", "\\boxed{2} and \\boxed{3}"))

    def test_nonfinite_or_nonnumeric_gold_not_guessed(self):
        for gold in ("", "two", "NaN", "2 apples", "1,20"):
            self.assertEqual(self.score(gold, gold), 0.0)

    def test_nlgraph_yes_no_exact_not_prefix(self):
        for dataset in ("nlgraph", "nlgraph_mini"):
            for gold, output, expected in [("Yes", "yes", 1), ("No", "No.", 1),
                                           ("No", "The answer is NO", 1), ("Yes", "Final answer: Yes", 1),
                                           ("No", "Not a path", 0), ("No", "Nobody", 0),
                                           ("Yes", "Yes or No", 0), ("No", "Answer: No. Answer: Yes.", 0),
                                           ("No", "Yes. Answer: No", 0),
                                           ("Yes", "Yes, there is a path", 0), ("No", "yes", 0)]:
                self.assertEqual(self.score(gold, output, dataset), float(expected), (dataset, output))
        self.assertIsNone(protocol.parse_exact_answer("gsm8k", "yes"))

    def test_strict_dev_test_rules_identical_for_same_raw_output(self):
        for dataset in protocol.STRICT_EXACT_DATASETS:
            for gold, output in itertools.product(["2", "12", "No", "Yes"],
                                                   ["2", "12", "2 a b c d e f g", "No", "Not", "Answer: 2"]):
                self.assertEqual(self.score(gold, output, dataset, "dev"),
                                 self.score(gold, output, dataset, "test"))

    def test_parse_score_and_gold_normalization_agree(self):
        output = "Work: 9 + 3. Final answer: 1,200.00"
        parsed = protocol.parse_exact_answer("gsm8k", output)
        self.assertEqual(parsed, "1200")
        self.assertEqual(self.score(parsed, output), 1.0)
        self.assertEqual(self.score("1200.000", output), 1.0)

    def test_unsupported_tasks_explicitly_keep_legacy_logic(self):
        for dataset in ("xstreet_ar", "xstreet_es", "custom"):
            self.assertEqual(self.score("red", "infrared", dataset), 1.0)
            self.assertEqual(protocol.parse_exact_answer(dataset, "infrared"), "infrared")
        metadata = protocol.PROTOCOL_METADATA["strict_v1"]
        self.assertIn("partial", metadata["exact"])
        self.assertEqual(metadata["exact_other_datasets"], "legacy_substring_unchanged")


class SliceAndAggregateTests(unittest.TestCase):
    def test_strict_offset_changes_real_ids(self):
        ids = list(range(10))
        self.assertEqual(ids[slice(*protocol.slice_bounds(10, 3, 4, "strict_v1"))], [4, 5, 6])
        self.assertEqual(protocol.slice_bounds(10, None, 4, "strict_v1"), (4, 10))
        self.assertEqual(protocol.slice_bounds(10, 10, 0, "strict_v1"), (0, 10))
        self.assertEqual(protocol.slice_bounds(1, None, None, "strict_v1"), (0, 1))

    def test_strict_rejects_empty_negative_and_out_of_bounds(self):
        for length, size, start in [(0, None, None), (10, 0, 0), (10, -1, 0), (10, None, -1),
                                    (10, None, 10), (10, 1, 10), (10, 6, 5), (10, 11, None),
                                    (10, 1.0, 0), (10, 1, False), (10, True, 0), (-1, None, None)]:
            with self.subTest(length=length, size=size, start=start):
                with self.assertRaises(ValueError):
                    protocol.slice_bounds(length, size, start, "strict_v1")

    def test_strict_harmonic_zero_and_nonzero(self):
        for scores in ([0, 0], [0, 0.8], [0.8, 0]):
            self.assertEqual(protocol.harmonic_mean(scores, "strict_v1"), 0.0)
        self.assertEqual(protocol.harmonic_mean([0.25, 0.75], "strict_v1"), 0.375)
        self.assertAlmostEqual(protocol.harmonic_mean([0.4, 0.6], "strict_v1"), 0.48)
        self.assertEqual(protocol.harmonic_mean([1e308, 1e308], "strict_v1"), 1e308)
        self.assertEqual(protocol.harmonic_mean([1e-308, 1e-308], "strict_v1"), 1e-308)

    def test_strict_harmonic_rejects_invalid_values(self):
        for scores in ([-1, 1], [math.nan, 1], [1, math.inf], [], [1], [1, 2, 3]):
            with self.assertRaises(ValueError):
                protocol.harmonic_mean(scores, "strict_v1")

    def test_argmax_is_stable_and_accepts_iterables(self):
        self.assertEqual(protocol.stable_argmax([0.5, 0.8, 0.8, 0.3]), 1)
        self.assertEqual(protocol.stable_argmax([0, 0, 0]), 0)
        self.assertEqual(protocol.stable_argmax(iter([-3, -2, -2])), 1)
        with self.assertRaises(ValueError):
            protocol.stable_argmax([])

    def test_unknown_protocol_and_split_reject(self):
        for call in (lambda: protocol.parse_mc({"choices": {}}, "", "typo"),
                     lambda: protocol.score_exact("gsm8k", "2", "2", "typo"),
                     lambda: protocol.parse_exact_answer("gsm8k", "2", "typo"),
                     lambda: protocol.slice_bounds(10, protocol="typo"),
                     lambda: protocol.harmonic_mean([1, 1], "typo"),
                     lambda: protocol.score_exact("gsm8k", "2", "2", split="validation")):
            with self.assertRaises(ValueError):
                call()

    def test_module_import_never_loads_gpu_or_cloud(self):
        code = ("import sys; sys.path.insert(0, " + repr(str(REPO)) + "); import eval_protocol; "
                "blocked={'torch','transformers','datasets','vertexai','googleapiclient','reward_modeling','requests'}; "
                "assert not blocked.intersection(sys.modules)")
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


# Tiny dense matrices reproduce factor-wise materialization without importing
# alpha_space (which imports torch). These are mathematical identity tests, not
# a claim that real adapters or the GPU materializer have been integration-tested.
def matrix_add(left, right):
    return [[x + y for x, y in zip(lrow, rrow)] for lrow, rrow in zip(left, right)]


def matrix_scale(value, matrix):
    return [[value * x for x in row] for row in matrix]


def matrix_multiply(left, right):
    return [[sum(x * y for x, y in zip(row, column)) for column in zip(*right)] for row in left]


def factor_merge(alphas, experts):
    merged = []
    for index in (0, 1):
        value = matrix_scale(0, experts[0][index])
        for alpha, expert in zip(alphas, experts):
            value = matrix_add(value, matrix_scale(alpha, expert[index]))
        merged.append(value)
    return matrix_multiply(merged[1], merged[0])


class FactorMergeSemanticsTests(unittest.TestCase):
    def setUp(self):
        # Expert tuples are (A[r,in], B[out,r]).
        self.experts = [([[1, 2]], [[3], [4]]), ([[5, -1]], [[2], [-2]])]

    def test_factor_merge_includes_cross_terms_not_linear_delta_merge(self):
        alpha, beta = 0.5, 0.25
        (a1, b1), (a2, b2) = self.experts
        diagonal = matrix_add(matrix_scale(alpha ** 2, matrix_multiply(b1, a1)),
                              matrix_scale(beta ** 2, matrix_multiply(b2, a2)))
        cross = matrix_scale(alpha * beta, matrix_add(matrix_multiply(b1, a2),
                                                     matrix_multiply(b2, a1)))
        actual = factor_merge([alpha, beta], self.experts)
        self.assertEqual(actual, matrix_add(diagonal, cross))
        self.assertNotEqual(actual, diagonal)
        linear_delta = matrix_add(matrix_scale(alpha, matrix_multiply(b1, a1)),
                                  matrix_scale(beta, matrix_multiply(b2, a2)))
        self.assertNotEqual(actual, linear_delta)

    def test_coefficient_scale_squared_even_with_negative_scale(self):
        alphas = [0.5, -0.25]
        baseline = factor_merge(alphas, self.experts)
        for scale in (0, 0.5, 1, 1.5, -2):
            self.assertEqual(factor_merge([scale * alpha for alpha in alphas], self.experts),
                             matrix_scale(scale ** 2, baseline))

    def test_onehot_recovers_each_expert_delta(self):
        for index, (a, b) in enumerate(self.experts):
            alphas = [int(i == index) for i in range(len(self.experts))]
            self.assertEqual(factor_merge(alphas, self.experts), matrix_multiply(b, a))


if __name__ == "__main__":
    unittest.main()
