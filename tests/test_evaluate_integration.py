"""CPU integration tests using a fake backend, not real inference."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def strict_namespace():
    # Execute the real strict entrypoint without importing CUDA/cloud dependencies.
    source = (ROOT / 'evaluate.py').read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_strict_evaluate')
    import eval_protocol
    namespace = dict(json=json, Path=Path, hashlib=__import__('hashlib'),
                     __file__=str(ROOT / 'evaluate.py'),
                     slice_bounds=eval_protocol.slice_bounds,
                     harmonic_mean=eval_protocol.harmonic_mean,
                     parse_mc=eval_protocol.parse_mc,
                     parse_exact_answer=eval_protocol.parse_exact_answer,
                     score_exact=eval_protocol.score_exact,
                     multitask_domain_dataset_dict={'medical': ['medqa', 'medmcqa']})
    namespace['torch'] = Mock()
    namespace['AutoModelForCausalLM'] = Mock()
    namespace['AutoTokenizer'] = Mock()
    namespace['PeftModel'] = Mock()
    namespace['multiple_choice_prompt'] = lambda row, dataset: row['question']
    namespace['batch_generate'] = Mock()
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / 'evaluate.py'), 'exec'), namespace)
    return namespace


class StrictIntegrationTest(unittest.TestCase):
    def test_exact_raw_archive_and_local_loading(self):
        ns = strict_namespace()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'base'
            adapter = Path(directory) / 'adapter'
            base.mkdir()
            adapter.mkdir()
            data = json.loads((ROOT / 'data/eval/gsm8k.json').read_text())
            ns['batch_generate'].return_value = [str(data['dev'][0]['answer'])]
            output = Path(directory) / 'result'
            result = ns['_strict_evaluate'](str(adapter), 'exact_match', 'gsm8k', 0,
                                           str(base), 'dev', 1, 0, output)
            self.assertEqual(result, 1.0)
            stored = json.loads((output / 'evaluation.json').read_text())
            self.assertEqual(len(stored['records']), 1)
            self.assertIn('raw_generation', stored['records'][0])
            for backend in ('AutoModelForCausalLM', 'AutoTokenizer', 'PeftModel'):
                self.assertTrue(ns[backend].from_pretrained.call_args.kwargs['local_files_only'])
            with self.assertRaises(FileExistsError):
                ns['_strict_evaluate'](str(adapter), 'exact_match', 'gsm8k', 0,
                                       str(base), 'dev', 1, 0, output)

    def test_invalid_slice_rejected_before_loading(self):
        ns = strict_namespace()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ns['_strict_evaluate'](directory, 'exact_match', 'gsm8k', 0,
                                       directory, 'dev', 0, 0, Path(directory) / 'result')
            ns['AutoModelForCausalLM'].from_pretrained.assert_not_called()

    def test_multitask_transmits_ranges_and_keeps_children(self):
        ns = strict_namespace()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result'
            def generate(model, tokenizer, prompts, gpu_id, **kwargs):
                self.assertEqual(len(prompts), 2)
                return ['A', 'A']
            ns['batch_generate'].side_effect = generate
            value = ns['_strict_evaluate'](directory, 'multitask', 'medical', 0,
                                          directory, 'dev', 2, 3, output)
            scores = []
            for child in ('medqa', 'medmcqa'):
                result = json.loads((output / child / 'evaluation.json').read_text())
                self.assertEqual([r['index'] for r in result['records']], [3, 4])
                scores.append(result['score'])
            self.assertEqual(value, ns['harmonic_mean'](scores, 'strict_v1'))
            self.assertTrue((output / 'evaluation.json').exists())

    def test_public_entrypoints_route_and_reject_component_override(self):
        tree = ast.parse((ROOT / 'evaluate.py').read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('evaluate', 'evaluate_test')]
        ns = {'_strict_evaluate': Mock(return_value=.5), 'ONLY_ONE_OR_TWO': None}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<entrypoints>', 'exec'), ns)
        for name, split in (('evaluate', 'dev'), ('evaluate_test', 'test')):
            self.assertEqual(ns[name]('adapter', 'exact_match', 'gsm8k', 0,
                                     protocol='strict_v1', result_dir='output'), .5)
            self.assertEqual(ns['_strict_evaluate'].call_args.args[5], split)
            with self.assertRaises(ValueError):
                ns[name]('adapter', 'multitask', 'medical', 0,
                         protocol='strict_v1', only_one_or_two='one')
            ns['ONLY_ONE_OR_TWO'] = 'two'
            with self.assertRaises(ValueError):
                ns[name]('adapter', 'multitask', 'medical', 0, protocol='strict_v1')
            ns['ONLY_ONE_OR_TWO'] = None

    def test_no_cloud_initialization_at_import(self):
        tree = ast.parse((ROOT / 'evaluate.py').read_text())
        top_level = ast.Module(body=[node for node in tree.body
                                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))],
                               type_ignores=[])
        calls = [ast.unparse(node.func) for node in ast.walk(top_level)
                 if isinstance(node, ast.Call)]
        self.assertFalse(any('discovery.build' in call or 'vertexai.init' in call
                             or 'GenerativeModel' in call for call in calls))


if __name__ == '__main__':
    unittest.main()
