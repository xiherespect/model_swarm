"""Standard-library CPU tests; invoke with python -I -m unittest discover.

Load only our explicitly addressed reader; legacy archives are treated as data.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("thesis_results", ROOT / "tools/thesis_results.py")
tr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tr)


def load_manifest():
    return json.loads((ROOT / "experiments/legacy_manifest.json").read_text(encoding="utf-8"))


class ParserTests(unittest.TestCase):
    def test_literal_dict_and_namespace(self):
        self.assertEqual(tr.literal_mapping("{'seed': 42, 'x': None}"), {"seed": 42, "x": None})
        self.assertEqual(tr.literal_mapping("Namespace(seed=42, x=False)"), {"seed": 42, "x": False})

    def test_no_code_execution(self):
        for text in ["__import__('os').system('false')", "{'a': open('/tmp/no')}",
                     "Namespace(**{'seed': 42})", "Namespace(42)", "[1, 2]",
                     "Namespace(seed=42, seed=43)"]:
            with self.subTest(text=text), self.assertRaises(tr.EvidenceError):
                tr.literal_mapping(text)

    def test_harmonic_zero_and_components(self):
        self.assertEqual(tr.harmonic([0, 0]), 0)
        self.assertEqual(tr.harmonic([0, .5]), 0)
        self.assertAlmostEqual(tr.harmonic([.7406015037593985, .15644820295983086]), .25832631905373676)
        for values in [[1], [1, 2], [True, .4], [float('nan'), .5], [.5, float('inf')]]:
            with self.subTest(values=values), self.assertRaises(tr.EvidenceError):
                tr.harmonic(values)

    def test_numeric_stable_tie_not_lexical_or_test(self):
        scratch = {f"particle_{i}_best": .2 for i in range(12)}
        scratch['particle_10_best'] = .7
        scratch['particle_3_best'] = .7
        scratch['test_best_particle'] = 10
        self.assertEqual(tr.stable_particle(scratch), (3, .7))
        with self.assertRaises(tr.EvidenceError):
            tr.stable_particle({'particle_1_best': .5})

    def test_path_escape_glob_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            root = Path(temp)
            (root / 'escape').symlink_to(outside, target_is_directory=True)
            for name in ['../x', '/tmp/x', 'search/*/log.txt', 'a?b', 'a[0]', 'a\\b', 'escape/file']:
                with self.subTest(name=name), self.assertRaises(tr.EvidenceError):
                    tr.safe_path(root, name)
            self.assertEqual(tr.safe_path(root, 'search/a/log.txt'), root / 'search/a/log.txt')


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest()

    def reject(self, manifest):
        with self.assertRaises(tr.EvidenceError):
            tr.validate_manifest(manifest)

    def test_valid_inventory(self):
        tr.validate_manifest(self.manifest)
        self.assertEqual(len(self.manifest['runs']), 113)
        self.assertEqual(self.manifest['expert_pool']['original_expert_count'], 10)
        self.assertEqual(self.manifest['expert_pool']['candidate_or_particle_count'], 20)

    def test_duplicate_task_seed(self):
        duplicate = copy.deepcopy(self.manifest['runs'][0])
        duplicate['id'] += '_another'
        self.manifest['runs'].append(duplicate)
        self.reject(self.manifest)

    def test_missing_matrix_row(self):
        self.manifest['runs'].pop(0)
        self.reject(self.manifest)

    def test_test_selected_policy_and_candidate(self):
        bad = copy.deepcopy(self.manifest)
        bad['selection_policy'] = 'best_test'
        self.reject(bad)
        self.manifest['runs'][0]['selection'] = 'max_test'
        self.reject(self.manifest)

    def test_mixed_protocol_even_if_cohort_relabelled(self):
        self.manifest['runs'][0]['protocol'] = 'strict_v1'
        self.manifest['cohorts'][0]['methods']['RACER']['protocol'] = 'strict_v1'
        self.reject(self.manifest)

    def test_unlabelled_sumo_config_mix(self):
        row = next(r for r in self.manifest['runs'] if r['cohort'] == 'sumo2_pair_enabled')
        row['config_id'] = 'no_pair_seed=True'
        self.reject(self.manifest)

    def test_logged_sumo_flag_not_just_label(self):
        row = next(r for r in self.manifest['runs'] if r['cohort'] == 'sumo2_pair_enabled')
        row['args']['no_pair_seed'] = True
        self.reject(self.manifest)

    def test_baseline_method_and_task_not_guessed_from_name(self):
        for key, value in [('method', 'random'), ('dataset', 'other')]:
            bad = copy.deepcopy(self.manifest)
            row = next(r for r in bad['runs'] if r['method'] == 'CMA')
            row['args'][key] = value
            self.reject(bad)

    def test_legal_independence_label(self):
        row = next(r for r in self.manifest['runs'] if r['task'] == 'legal')
        row['data_independence'] = 'independent'
        self.manifest['task_independence']['legal'] = 'independent'
        self.reject(self.manifest)

    def test_unknown_version_not_inferred(self):
        self.manifest['runs'][0]['code_version'] = 'HEAD'
        self.reject(self.manifest)

    def test_missing_or_wrong_components(self):
        row = next(r for r in self.manifest['runs'] if r['task'] == 'science' and r['method'] == 'PSO')
        row['components'] = ['stem']
        self.reject(self.manifest)

    def test_schema_json_and_row_contract(self):
        # No jsonschema dependency: schema readability plus required-field contract.
        schema = json.loads((ROOT / 'experiments/run_manifest.schema.json').read_text())
        self.assertEqual(schema['properties']['evidence_track']['const'], self.manifest['evidence_track'])
        self.assertTrue(set(schema['required']) <= set(self.manifest))
        row_schema = schema['$defs']['run']
        for row in self.manifest['runs']:
            self.assertTrue(set(row_schema['required']) <= set(row))
            self.assertTrue(set(row) <= set(row_schema['properties']))


class RealEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.report = tr.build_report(cls.manifest, ROOT)

    def test_reconstruct_main_means_and_soup_arithmetic(self):
        expected = {'RACER': .44391666666666657, 'PSO': .4327983440592506,
                    'CMA': .4363333333333333, 'Soup': .40033333333333326,
                    'Random': .40825, 'Anchors': .40275}
        table = self.report['tables']['main']['methods']
        for method, value in expected.items():
            with self.subTest(method=method):
                self.assertAlmostEqual(table[method]['macro_mean'], value, places=13)
        self.assertEqual(len(table['RACER']['run_ids']), 24)
        self.assertEqual(len(table['PSO']['run_ids']), 12)
        self.assertFalse(self.report['strict_rescoring'])

    def test_pso_components_corrected_not_rescored(self):
        expected = {'science': .25832631905373676, 'legal': .5886990801576872,
                    'medical': .4284014002333722, 'culture': .553240849525531}
        for task, value in expected.items():
            row = next(r for r in self.report['rows'] if r['cohort'] == 'main' and r['method'] == 'PSO' and r['task'] == task)
            self.assertAlmostEqual(row['test'], value)
            self.assertEqual(len(row['test_components']), 2)
            self.assertNotEqual(row['test'], row['legacy_reported_test'])
        science = next(r for r in self.report['rows'] if r['id'] == 'pso_science_r01')
        self.assertEqual(science['selected_particle'], 3)

    def test_missing_final_log_uses_frozen_dev_selected_scores(self):
        for name, index, value in [('pso_nl_r01cfg', 16, .472), ('pso_gsm8k_r01', 0, .384)]:
            row = next(r for r in self.report['rows'] if r['id'] == name)
            self.assertEqual(row['selected_particle'], index)
            self.assertEqual(row['test_count'], 1000)
            self.assertAlmostEqual(row['test'], value)

    def test_every_main_cell_traceable(self):
        for row in self.report['rows']:
            self.assertTrue(row['sources'])
            for source in row['sources']:
                self.assertEqual(len(source['sha256']), 64)
            if row['cohort'] == 'main' and row['method'] != 'PSO':
                self.assertTrue(any(s['role'] == 'metrics' and s.get('line') for s in row['sources']))

    def test_data_independence_question_not_whole_record(self):
        audit = {a['dataset']: a for a in self.report['data_audit']}
        self.assertEqual(audit['hearsay']['shared_unique_questions'], 94)
        self.assertEqual(audit['citation_prediction_classification']['shared_unique_questions'], 108)
        self.assertEqual(audit['citation_prediction_classification']['identical_full_records'], 48)
        self.assertEqual(audit['knowledge_crosswords']['shared_unique_questions'], 120)
        self.assertEqual(audit['knowledge_crosswords']['identical_full_records'], 0)

    def test_router_is_real_event_not_accuracy(self):
        rows = [r for r in self.report['rows'] if r['cohort'] == 'amphi_route_events']
        self.assertEqual(sum(r['decision'] == 'SWITCH' for r in rows), 3)
        self.assertEqual(sum(r['decision'] == 'KEEP' for r in rows), 9)
        self.assertTrue(all(r['counterfactual_accuracy'] is None and r['test'] is None for r in rows))
        self.assertIsNone(self.report['tables']['amphi_route_events']['methods']['AMPHI']['macro_mean'])

    def test_sumo_groups_no_cross_version_seed_mean(self):
        groups = self.report['tables']
        self.assertEqual(set(groups['sumo2_pair_enabled']['tasks']), {'nlgraph', 'gsm8k', 'legal'})
        self.assertNotIn('hellaswag', groups['sumo2_no_pair']['tasks'])
        self.assertNotIn('SUMO', groups['main']['methods'])
        self.assertAlmostEqual(groups['sumo_original']['methods']['SUMO']['per_task']['gsm8k'], .469)
        self.assertAlmostEqual(groups['sumo2_pair_enabled']['methods']['SUMO']['per_task']['gsm8k'], .419)

    def test_changed_source_and_selection_fail_closed(self):
        row = copy.deepcopy(self.manifest['runs'][0])
        row['sources'][0]['sha256'] = '0' * 64
        with self.assertRaises(tr.EvidenceError):
            tr.extract_row(ROOT, row)
        row = copy.deepcopy(next(r for r in self.manifest['runs'] if r['id'] == 'pso_science_r01'))
        row['selected_particle'] = 10  # tied dev score, but not first index
        with self.assertRaises(tr.EvidenceError):
            tr.extract_row(ROOT, row)

    def test_arguments_and_expected_values_checked(self):
        for key in ['args', 'expected']:
            row = copy.deepcopy(self.manifest['runs'][0])
            row[key]['seed' if key == 'args' else 'test'] = -1
            with self.assertRaises(tr.EvidenceError):
                tr.extract_row(ROOT, row)

    def test_generated_table_in_chapter(self):
        chapter = (ROOT / 'thesis/ch4_racer.md').read_text()
        self.assertIn(tr.markdown_table(self.report), chapter)

    def test_cli_new_output_only_and_no_legacy_writes(self):
        sources = {s['path']: s['sha256'] for r in self.manifest['runs'] for s in r['sources']}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'report.json'
            command = [sys.executable, '-I', str(ROOT/'tools/thesis_results.py'), '--output', str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertAlmostEqual(json.loads(output.read_text())['tables']['main']['methods']['PSO']['macro_mean'], .4327983440592506)
            before = output.read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), before)
            forbidden = subprocess.run([sys.executable, '-I', str(ROOT/'tools/thesis_results.py'), '--output', str(ROOT/'search/__forbidden_report.json')], capture_output=True, text=True)
            self.assertNotEqual(forbidden.returncode, 0)
            self.assertFalse((ROOT/'search/__forbidden_report.json').exists())
        for path, digest in sources.items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest)


class FactorMergeMathTests(unittest.TestCase):
    def test_factor_cross_terms_scale_square_and_onehot(self):
        # Scalar 1x1 A/B are sufficient to falsify delta-linear interpretation.
        a, b = [2., 5.], [3., 7.]
        def factor(alpha):
            return sum(x*y for x,y in zip(alpha,a)) * sum(x*y for x,y in zip(alpha,b))
        alpha = [.4, .6]
        diagonal = sum(x*x*aa*bb for x,aa,bb in zip(alpha,a,b))
        cross = alpha[0]*alpha[1]*(b[0]*a[1]+b[1]*a[0])
        self.assertAlmostEqual(factor(alpha), diagonal+cross)
        linear_delta = sum(x*aa*bb for x,aa,bb in zip(alpha,a,b))
        self.assertNotAlmostEqual(factor(alpha), linear_delta)
        for scale in [-1., .5, 1.5]:
            self.assertAlmostEqual(factor([scale*x for x in alpha]), scale**2*factor(alpha))
        self.assertEqual(factor([1.,0.]), a[0]*b[0])
        self.assertEqual(factor([0.,1.]), a[1]*b[1])
        self.assertEqual(factor([-1.,0.]), factor([1.,0.]))


if __name__ == '__main__':
    unittest.main()
