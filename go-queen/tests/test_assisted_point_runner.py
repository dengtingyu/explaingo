"""Source-level checks for the bounded APS fit runner; no pretrained model and no fit.

Written without being executed by its author; results count only once run.
Intended location: tests/test_assisted_point_runner.py beside
scripts/train_grounding_assisted_point.py. Pure helpers need only the standard
library; small checks use torch, safetensors and a tiny fake Qwen. Nothing here
loads Qwen3-0.6B, cached features or the fixed questions, and no optimizer
touches an APS model.
"""
from __future__ import annotations

import builtins
import copy
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/train_grounding_assisted_point.py"
HEAVY = {"numpy", "torch", "transformers", "safetensors"}


def load_runner(name="aps_runner_under_test"):
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from goqueen.assisted_point import AssistedPointGoQueen  # noqa: E402
from goqueen.assisted_point_eval import point_indices_for  # noqa: E402
from goqueen.checkpoints import load_adapter_checkpoint  # noqa: E402
from goqueen.fusion import GoQueen  # noqa: E402
from goqueen.grounding_data import occupancy_prompt, point_to_gtp  # noqa: E402


def canonical_question(index, label=0, player="B"):
    point = point_to_gtp(index)
    return {"question_id": f"q{index}", "position_id": "board", "to_play": player,
            "point_index": index, "point": point, "prompt": occupancy_prompt(player, point),
            "label": label, "answer": {"occupancy": ("empty", "black", "white")[label]}}


class TinyDecoder(nn.Module):
    attention_type = "full_attention"

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=4, num_hidden_layers=11)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(7, 4)
        self.model.layers = nn.ModuleList(TinyDecoder() for _ in range(11))
        self.lm_head = nn.Linear(4, 7, bias=False)

    def forward(self, input_ids, use_cache=None, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def tiny(kind=AssistedPointGoQueen, seed=7):
    torch.manual_seed(seed)
    return kind(TinyQwen(), expert_width=2, heads=1)


def complete_report():
    questions = [{"question_id": f"q{i}"} for i in range(21)]
    return {
        "run_completed": True, "updates_completed": 180,
        "update_attempts": [{"update": i + 1, "optimizer_step_returned": True} for i in range(180)],
        "fixed_questions": questions,
        "evaluations": [{"name": "initial", "updates_completed": 0},
                        {"name": "final", "updates_completed": 180, "coverage_complete": True,
                         "completed_questions": 21, "rows": copy.deepcopy(questions)}],
        "validity": {"pre_update": True, "checks": [{"name": "source", "matched": True}]},
        "final_parameters_finite": True,
        "assisted_point_contract": {"optimizer_contract_verified": True, "routing_verified": True,
                                    "inactive_zero_gradient_updates": 180,
                                    "inactive_44_bytes_unchanged": True, "binding_clear_at_end": True},
        "source_validity_after": True,
        "frozen_base_final_hash_status": "measured",
        "frozen_base_sha256_before": "a" * 64, "frozen_base_sha256_after": "a" * 64,
        "frozen_base_gradients_absent": True,
        "checkpoint_status": "saved_and_loader_verified",
    }


class ImportBoundaryTests(unittest.TestCase):
    def test_module_load_imports_no_heavy_dependency(self):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in HEAVY:
                raise AssertionError(f"Heavy import at module load: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            module = load_runner("aps_runner_import_guard")
        self.assertEqual(module.STAGE, "a0_assisted_point")
        self.assertEqual(module.DEFAULT_OUTPUT, ROOT / "outputs/grounding-assisted-point/report.json")

    def test_required_implementation_files_exist(self):
        for name in runner.REQUIRED_IMPLEMENTATION:
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file())


class ArgumentAndPathTests(unittest.TestCase):
    unused = ROOT / "outputs/aps-runner-unused-test"

    def test_defaults_and_budget_bounds(self):
        args = runner.parse_arguments([])
        self.assertEqual((args.seconds, args.output), (300, ROOT / "outputs/grounding-assisted-point/report.json"))
        self.assertEqual(runner.parse_arguments(["--seconds", "1"]).seconds, 1)
        for seconds in ("0", "-1", "301"):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                runner.parse_arguments(["--seconds", seconds])

    def test_dedicated_new_directory_paths(self):
        with patch.object(Path, "exists", return_value=False):
            paths = runner.artifact_paths(self.unused / "report.json")
        self.assertEqual({name: path.name for name, path in paths.items()},
                         {"report": "report.json", "temporary": "report.tmp", "start": "run-start.json",
                          "runner_copy": "runner-at-run.py", "checkpoint": "adapter.safetensors",
                          "checkpoint_temp": "adapter.partial.safetensors"})
        self.assertTrue(all(path.parent == self.unused.resolve() for path in paths.values()))

    def test_protected_outside_and_reserved_outputs_rejected(self):
        bad = [ROOT / "outputs/report.json", ROOT / "report.json", ROOT / "src/report.json",
               Path("/tmp/aps-report.json"), self.unused / "report.txt",
               ROOT / "outputs/grounding-m-key/nested/report.json"]
        bad += [ROOT / "outputs" / directory / "new.json" for directory in runner.PROTECTED_DIRECTORIES]
        bad += [self.unused / name for name in ("run-start.json", "RUN-START.json", "process-start.json",
                                                "process-completion.json", "adapter.safetensors")]
        with patch.object(Path, "exists", return_value=False):
            for output in bad:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    runner.artifact_paths(output)

    def test_any_existing_child_artifact_blocks_resume(self):
        directory = self.unused.resolve()
        for name in ("report.json", "report.tmp", "run-start.json", "runner-at-run.py",
                     "adapter.safetensors", "adapter.partial.safetensors"):
            existing = directory / name
            with self.subTest(name=name), \
                    patch.object(Path, "exists", lambda path, e=existing: path.resolve() == e):
                with self.assertRaises(ValueError):
                    runner.artifact_paths(directory / "report.json")

    def test_launcher_records_do_not_block_the_child(self):
        directory = self.unused.resolve()
        launcher = {directory / name for name in ("process-start.json", "process-stdout.log", "process-stderr.log")}
        with patch.object(Path, "exists", lambda path: path.resolve() in launcher):
            runner.artifact_paths(directory / "report.json")


class PreImportRejectionTests(unittest.TestCase):
    def assert_rejected(self, arguments, existing=False):
        real_import = builtins.__import__

        def forbid_heavy_import(name, *args, **kwargs):
            if name.split(".")[0] in HEAVY:
                raise AssertionError(f"Unexpected heavy import: {name}")
            return real_import(name, *args, **kwargs)

        with patch.object(sys, "argv", ["train_grounding_assisted_point.py", *arguments]), \
                patch("builtins.__import__", side_effect=forbid_heavy_import), \
                patch.object(Path, "exists", return_value=existing), \
                patch.object(Path, "mkdir", side_effect=AssertionError("Unexpected mkdir")) as mkdir, \
                patch.object(Path, "open", side_effect=AssertionError("Unexpected open")) as opened, \
                patch.object(Path, "write_text", side_effect=AssertionError("Unexpected write")) as written, \
                patch.object(runner, "file_sha256", side_effect=AssertionError("Unexpected hash")) as hashed:
            with self.assertRaises(ValueError):
                runner.main()
            for mocked in (mkdir, opened, written, hashed):
                mocked.assert_not_called()

    def test_invalid_budget_rejected(self):
        for seconds in ("0", "301"):
            with self.subTest(seconds=seconds):
                self.assert_rejected(["--seconds", seconds])

    def test_old_run_directories_and_reserved_names_rejected(self):
        for output in (ROOT / "outputs/grounding-microfit/new.json", ROOT / "outputs/grounding-m-key/new.json",
                       ROOT / "outputs/assisted-point-engineering/new.json",
                       ROOT / "outputs/aps-runner-unused-test/run-start.json"):
            with self.subTest(output=output):
                self.assert_rejected(["--output", str(output)])

    def test_existing_artifact_prevents_a_second_attempt(self):
        self.assert_rejected(["--output", str(ROOT / "outputs/aps-runner-unused-test/report.json")], existing=True)


class HelperTests(unittest.TestCase):
    def test_routed_metadata_is_label_free_and_still_routable(self):
        question = canonical_question(300, label=2)
        original = copy.deepcopy(question)
        routed = runner.routed_metadata(question)
        self.assertEqual(question, original)
        self.assertNotIn("label", routed)
        self.assertNotIn("answer", routed)
        self.assertEqual(routed, {k: v for k, v in question.items() if k not in ("label", "answer")})
        self.assertEqual(point_indices_for([routed], "cpu").tolist(), [300])

    def test_source_feature_digest_matches_engineering_definition_and_ignores_order(self):
        entries = {"b": {"file": "outputs/x/b.npz", "sha256": "1" * 64, "board_id": "ignored"},
                   "a": {"file": "outputs/x/a.npz", "sha256": "2" * 64, "absolute_labels_checked": 361}}
        digest, payload = runner.source_features_digest(entries)
        self.assertEqual(payload, [{"file": "outputs/x/a.npz", "sha256": "2" * 64},
                                   {"file": "outputs/x/b.npz", "sha256": "1" * 64}])
        self.assertEqual(digest, runner.content_sha256(payload))
        self.assertEqual(runner.source_features_digest(dict(reversed(list(entries.items()))))[0], digest)

    def test_checkpoint_metadata_contract_and_rejections(self):
        digests = {"manifest_sha256": "a" * 64, "questions_sha256": "b" * 64, "source_features_sha256": "c" * 64}
        metadata = runner.checkpoint_metadata(updates=180, **digests)
        self.assertEqual({key: metadata[key] for key in ("variant", "routing_contract", "engineering_only",
                                                         "stage", "updates", "source_manifest_sha256",
                                                         "questions_sha256", "source_features_sha256")},
                         {"variant": "m_assisted_point", "routing_contract": "single_point_bool_mask_v1",
                          "engineering_only": "false", "stage": "a0_assisted_point", "updates": "180",
                          "source_manifest_sha256": "a" * 64, "questions_sha256": "b" * 64,
                          "source_features_sha256": "c" * 64})
        self.assertTrue(all(isinstance(value, str) for value in metadata.values()))
        for change in ({"updates": -1}, {"updates": 181}, {"updates": True}, {"updates": "180"},
                       {"manifest_sha256": "A" * 64}, {"questions_sha256": "b" * 63},
                       {"source_features_sha256": None}):
            arguments = {"updates": 180, **digests, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.checkpoint_metadata(**arguments)

    def test_fit_checkpoint_metadata_passes_the_actual_APS_loader(self):
        model = tiny()
        metadata = runner.checkpoint_metadata(updates=180, manifest_sha256="a" * 64,
                                              questions_sha256="b" * 64, source_features_sha256="c" * 64)
        state = {name: tensor.clone() + 0.5 for name, tensor in model.adapter_state_dict().items()}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_file(state, str(path), metadata=metadata)
            self.assertEqual(load_adapter_checkpoint(model, path), metadata)
            for name, tensor in model.adapter_state_dict().items():
                with self.subTest(name=name):
                    self.assertTrue(torch.equal(tensor, state[name]))
            with self.assertRaises(ValueError):
                load_adapter_checkpoint(tiny(GoQueen), path)

    def test_imported_first_party_hashes_only_project_sources(self):
        modules = {"fusion": SimpleNamespace(__file__=str(ROOT / "src/goqueen/fusion.py")),
                   "helpers": SimpleNamespace(__file__=str(ROOT / "scripts/train_grounding_microfit.py")),
                   "foreign": SimpleNamespace(__file__=str(Path(torch.__file__))),
                   "document": SimpleNamespace(__file__=str(ROOT / "README.md")),
                   "builtin": SimpleNamespace()}
        pins = runner.imported_first_party(modules)
        self.assertEqual(set(pins), {"src/goqueen/fusion.py", "scripts/train_grounding_microfit.py",
                                     "scripts/train_grounding_assisted_point.py"})
        self.assertEqual(pins["src/goqueen/fusion.py"], runner.file_sha256(ROOT / "src/goqueen/fusion.py"))


class ContractHelperTests(unittest.TestCase):
    def test_inactive_selection_is_the_44_query_key_and_query_norm_tensors(self):
        named = [(name, p) for name, p in tiny().named_parameters() if p.requires_grad]
        inactive = runner.inactive_parameters(named)
        self.assertEqual(len(named), 101)
        self.assertEqual(len(inactive), 44)
        self.assertEqual(sorted({name.split(".adapter.")[1] for name, _ in inactive}),
                         ["key.weight", "query.weight", "query_norm.bias", "query_norm.weight"])

    def test_exact_zero_gradient_contract(self):
        first, second = nn.Parameter(torch.ones(3)), nn.Parameter(torch.ones(3))
        named = [("adapter.query.weight", first), ("adapter.key.weight", second)]
        first.grad = torch.tensor([0.0, -0.0, 0.0])
        second.grad = torch.zeros(3)
        runner.require_exact_zero_gradients(named)
        for grad in (torch.tensor([0.0, 1e-30, 0.0]), torch.tensor([0.0, float("nan"), 0.0]), None):
            second.grad = grad
            with self.subTest(grad=grad), self.assertRaises(runner.ContractViolation):
                runner.require_exact_zero_gradients(named)

    def test_optimizer_contract(self):
        g = torch.Generator().manual_seed(3)
        params = [nn.Parameter(torch.randn(2, generator=g)) for _ in range(3)]
        named = [(f"p{i}", p) for i, p in enumerate(params)]
        fixed = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
        settings = runner.optimizer_contract(torch.optim.AdamW(params, **fixed), named)
        self.assertEqual({key: settings[key] for key in runner.EXPECTED_OPTIMIZER}, runner.EXPECTED_OPTIMIZER)
        stepped = torch.optim.AdamW(params, **fixed)
        for p in params:
            p.grad = torch.ones_like(p)
        stepped.step()
        cases = {"default_weight_decay": torch.optim.AdamW(params, lr=0.001),
                 "reordered": torch.optim.AdamW(list(reversed(params)), **fixed),
                 "missing_member": torch.optim.AdamW(params[:2], **fixed),
                 "two_groups": torch.optim.AdamW([{"params": params[:1]}, {"params": params[1:]}], **fixed),
                 "not_fresh": stepped}
        for name, optimizer in cases.items():
            with self.subTest(case=name), self.assertRaises(runner.ContractViolation):
                runner.optimizer_contract(optimizer, named)

    def test_zero_gradient_tensors_keep_their_bytes_under_fresh_AdamW(self):
        # The mechanism behind the 44-tensor contract, on tiny tensors including signed zeros.
        inactive = nn.Parameter(torch.tensor([0.0, -0.0, 1.5, -2.25, 3e-8]))
        active = nn.Parameter(torch.tensor([0.5, -1.0]))
        before = inactive.detach().clone()
        optimizer = torch.optim.AdamW([inactive, active], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            inactive.grad = torch.tensor([0.0, -0.0, 0.0, -0.0, 0.0])
            active.grad = torch.tensor([1.0, -2.0]) * (step + 1)
            optimizer.step()
        self.assertTrue(torch.equal(inactive.detach().view(torch.int32), before.view(torch.int32)))
        self.assertFalse(torch.equal(active.detach(), torch.tensor([0.5, -1.0])))
        state = optimizer.state[inactive]
        self.assertTrue(bool(state["exp_avg"].eq(0).all()) and bool(state["exp_avg_sq"].eq(0).all()))


class RoutingMonitorTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny()
        self.monitor = runner.RoutingMonitor(self.model)
        self.addCleanup(self.monitor.close)
        self.features = torch.randn(3, 11, 361, 2, generator=torch.Generator().manual_seed(1))
        self.ids = torch.zeros(3, 5, dtype=torch.long)

    def forward(self, points):
        batch = len(points)
        return self.model(input_ids=self.ids[:batch], expert_features=self.features[:batch],
                          point_indices=torch.tensor(points, dtype=torch.long))

    def test_hooks_register_no_tensor(self):
        model = tiny()
        before = list(model.state_dict())
        monitor = runner.RoutingMonitor(model)
        self.addCleanup(monitor.close)
        self.assertEqual(list(model.state_dict()), before)

    def test_correct_actual_batches_are_counted_per_adapter_and_cleared(self):
        with torch.no_grad(), self.monitor.expect([0, 300, 360], "training"):
            self.forward([0, 300, 360])
        with torch.no_grad(), self.monitor.expect([17], "evaluation"):
            self.forward([17])
        self.assertEqual(self.monitor.counts, {"training": 1, "evaluation": 1})
        self.assertEqual(self.monitor.adapter_calls, 22)
        self.assertEqual(self.monitor.routed_samples, 4)
        self.assertTrue(runner.binding_clear(self.model))

    def test_wrong_order_point_or_batch_is_a_violation_and_still_clears(self):
        for expected, actual in (([300, 0], [0, 300]), ([5], [6]), ([1, 2], [1])):
            with self.subTest(expected=expected, actual=actual):
                with self.assertRaises(runner.ContractViolation):
                    with torch.no_grad(), self.monitor.expect(expected, "training"):
                        self.forward(actual)
                self.assertTrue(runner.binding_clear(self.model))
                self.assertIsNone(self.monitor.expected)
        self.assertEqual(self.monitor.counts, {"training": 0, "evaluation": 0})

    def test_unchecked_missing_nested_or_unknown_expectations_rejected(self):
        with self.assertRaises(runner.ContractViolation), torch.no_grad():
            self.forward([3])
        with self.assertRaises(runner.ContractViolation):
            with self.monitor.expect([3], "evaluation"):
                pass
        with self.assertRaises(runner.ContractViolation):
            with self.monitor.expect([3], "training"):
                with self.monitor.expect([3], "training"):
                    pass
        with self.assertRaises(runner.ContractViolation):
            with self.monitor.expect([3], "generation"):
                pass
        self.assertIsNone(self.monitor.expected)
        self.assertTrue(runner.binding_clear(self.model))

    def test_close_removes_every_hook(self):
        self.monitor.close()
        with torch.no_grad():
            self.forward([3])
        self.assertEqual(self.monitor.adapter_calls, 0)


class EndpointConditionTests(unittest.TestCase):
    def test_complete_report_meets_every_condition(self):
        conditions = runner.endpoint_conditions(complete_report())
        self.assertTrue(conditions)
        self.assertTrue(all(conditions.values()), conditions)

    def test_each_condition_fails_alone(self):
        cases = {
            "pre_update_reproduction": lambda r: r["validity"].update(pre_update=False),
            "all_recorded_checks_matched": lambda r: r["validity"]["checks"].append({"name": "x", "matched": False}),
            "run_completed": lambda r: r.update(run_completed=False),
            "updates_180_returned": lambda r: r["update_attempts"][-1].update(optimizer_step_returned=False),
            "final_21_exact": lambda r: r["evaluations"][1]["rows"].pop(),
            "finite_final_parameters": lambda r: r.pop("final_parameters_finite"),
            "optimizer_contract": lambda r: r["assisted_point_contract"].update(optimizer_contract_verified=False),
            "routing_verified": lambda r: r["assisted_point_contract"].update(routing_verified=False),
            "inactive_zero_gradients_every_update": lambda r: r["assisted_point_contract"].update(inactive_zero_gradient_updates=179),
            "inactive_44_bytes_unchanged": lambda r: r["assisted_point_contract"].update(inactive_44_bytes_unchanged=False),
            "binding_clear_at_end": lambda r: r["assisted_point_contract"].update(binding_clear_at_end=None),
            "source_validity_after": lambda r: r.pop("source_validity_after"),
            "frozen_base_hash_measured_equal": lambda r: r.update(frozen_base_sha256_after="b" * 64),
            "frozen_base_gradients_absent": lambda r: r.update(frozen_base_gradients_absent=False),
            "checkpoint_saved_and_loader_verified": lambda r: r.update(checkpoint_status="not_saved_to_preserve_report_budget"),
        }
        self.assertEqual(set(cases), set(runner.endpoint_conditions(complete_report())))
        for name, change in cases.items():
            report = complete_report()
            change(report)
            failed = [key for key, passed in runner.endpoint_conditions(report).items() if not passed]
            with self.subTest(condition=name):
                self.assertEqual(failed, [name])

    def test_final_rows_must_be_exactly_the_fixed_21_at_the_fixed_endpoint(self):
        def duplicate(r):
            r["evaluations"][1]["rows"][20]["question_id"] = "q0"

        def foreign(r):
            r["evaluations"][1]["rows"][20]["question_id"] = "other"

        def intermediate(r):
            r["evaluations"][1]["updates_completed"] = 120

        def two_finals(r):
            r["evaluations"].append(copy.deepcopy(r["evaluations"][1]))

        def unmeasured_hash(r):
            r["frozen_base_final_hash_status"] = "not_measured_to_preserve_report_budget"

        for change, condition in ((duplicate, "final_21_exact"), (foreign, "final_21_exact"),
                                  (intermediate, "final_21_exact"), (two_finals, "final_21_exact"),
                                  (unmeasured_hash, "frozen_base_hash_measured_equal")):
            report = complete_report()
            change(report)
            with self.subTest(change=change.__name__):
                self.assertFalse(runner.endpoint_conditions(report)[condition])

    def test_empty_report_meets_nothing(self):
        self.assertFalse(any(runner.endpoint_conditions({}).values()))


if __name__ == "__main__":
    unittest.main()
