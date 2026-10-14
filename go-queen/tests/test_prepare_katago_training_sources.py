"""Pure selection/rule arithmetic; no models, archive downloads or source replay."""
from collections import Counter
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/prepare_katago_training_sources.py"
spec = importlib.util.spec_from_file_location("katago_selection_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SelectionTests(unittest.TestCase):
    def test_no_heavy_or_network_import_action(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("katago.game.board", sys.modules)

    def test_exact_rules_preserve_komi(self):
        rules = module.parse_rules_exact("koSIMPLEscoreAREAtaxNONEsui0", -1.5)
        self.assertEqual(rules["whiteKomi"], -1.5)
        self.assertEqual(rules["taxRule"], "TAX_NONE")
        self.assertFalse(rules["multiStoneSuicideLegal"])

    def test_unsupported_rules_are_not_normalized(self):
        for raw in ["koPOSITIONALscoreAREAtaxNONEsui0", "koSIMPLEscoreAREAtaxSEKIsui0",
                    "koSIMPLEscoreTERRITORYtaxNONEsui0", "koSIMPLEscoreAREAtaxNONEsui1",
                    "koSIMPLEscoreAREAtaxNONEsui0button1", "Japanese", None]:
            with self.assertRaises(ValueError):
                module.parse_rules_exact(raw, 7.5)

    def test_invalid_komi(self):
        for komi in [None, True, float("nan"), float("inf"), "7.5"]:
            with self.assertRaises(ValueError):
                module.parse_rules_exact("Chinese", komi)

    def test_d4_game_and_prefix_invariance(self):
        moves = [("B", 3), ("W", 144), ("B", None), ("W", 305)]
        for symmetry in range(8):
            transformed = [(p, module.transformed_point(i, symmetry)) for p, i in moves]
            self.assertEqual(module.canonical_moves(moves), module.canonical_moves(transformed))
            self.assertEqual(module.canonical_moves(moves, 3), module.canonical_moves(transformed, 3))
        self.assertNotEqual(module.canonical_moves(moves), module.canonical_moves(moves[:-1]))

    def test_balanced_points_without_replacement(self):
        labels = [0] * 300 + [1] * 40 + [2] * 21
        points = module.balanced_points(labels, 42, "boardA")
        self.assertEqual(len(points), 63)
        self.assertEqual(len(set(points)), 63)
        self.assertEqual(Counter(labels[i] for i in points), {0: 21, 1: 21, 2: 21})
        self.assertEqual(points, module.balanced_points(labels, 42, "boardA"))
        self.assertEqual(module.balanced_points([0] * 361, 42, "empty"), [])

    def test_game_split_is_stable_when_pool_grows(self):
        first = {str(i): module.game_split(str(i)) for i in range(100)}
        second = {str(i): module.game_split(str(i)) for i in range(200)}
        self.assertEqual(first, {k: second[k] for k in first})
        self.assertEqual(set(first.values()), {"train", "val"})

    def test_coordinate_convention(self):
        self.assertEqual(module.coordinate(0), "A19")
        self.assertEqual(module.coordinate(360), "T1")
        self.assertEqual(len({module.coordinate(i) for i in range(361)}), 361)
        self.assertTrue(all(not module.coordinate(i).startswith("I") for i in range(361)))

    def test_selected_plies_cover_both_sides_and_three_stages(self):
        self.assertEqual(module.DEFAULT_PLIES, (20, 39, 60, 79, 100, 119))
        self.assertEqual(Counter(p % 2 for p in module.DEFAULT_PLIES), {0: 3, 1: 3})

    def test_download_is_index_bound_https_verified_and_size_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "2020-12-09sgfs.tar.bz2"
            url = module.ARCHIVE_BASE + target.name
            with self.assertRaises(ValueError):
                module.download_archive(url, target, "missing", 100)
            def fake_download(command, check):
                self.assertTrue(check)
                self.assertIn("--max-filesize", command)
                self.assertEqual(command[command.index("--max-filesize") + 1], "100")
                self.assertNotIn("--insecure", command)
                self.assertEqual(command[-1], url)
                Path(command[command.index("--output") + 1]).write_bytes(b"BZh-payload")
            with patch.object(module.subprocess, "run", fake_download):
                module.download_archive(url, target, target.name, 100)
            self.assertEqual(target.read_bytes(), b"BZh-payload")

    def test_compressed_members_sorted_after_sequential_read(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "small.tar.bz2"
            with tarfile.open(path, "w:bz2") as handle:
                for name in ("z.sgf", "a.sgf", "m.sgf"):
                    payload = name.encode()
                    info = tarfile.TarInfo(name); info.size = len(payload)
                    handle.addfile(info, io.BytesIO(payload))
            self.assertEqual(module.archive_payloads(path), [(n, n.encode()) for n in ("a.sgf", "m.sgf", "z.sgf")])


if __name__ == "__main__":
    unittest.main()
