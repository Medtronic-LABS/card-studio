"""htr.data.importers.{public,banglawriting,bnhtrd,banglalekha}: index parsing, writer-id grouping, missing-download errors, and
the BanglaLekha folder-per-class glyph pool. All against synthetic fixtures built in a tempdir (no real dataset downloaded or
committed)."""
import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.data.importers.banglalekha as BL
import htr.data.importers.banglawriting as BW
import htr.data.importers.bnhtrd as BH
import htr.data.importers.public as PUB


class ReadIndex(unittest.TestCase):
    def test_missing_index_raises_with_a_helpful_message(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, "download"):
                PUB.read_index(Path(d) / "nope.tsv", d, "x", "src")

    def test_parses_tab_separated_lines_and_skips_blank_and_comments(self):
        with tempfile.TemporaryDirectory() as d:
            idx = Path(d) / "index.tsv"
            idx.write_text("# comment\n\nw1/a.png\ttext one\nw1/b.png\ttext two\n")
            rows = PUB.read_index(idx, d, "doc", "src")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "text one")
            self.assertEqual(rows[0]["image_path"], str(Path(d) / "w1/a.png"))
            self.assertEqual(rows[0]["document_type"], "doc")

    def test_bad_line_raises_with_line_number(self):
        with tempfile.TemporaryDirectory() as d:
            idx = Path(d) / "index.tsv"; idx.write_text("no-tab-here\n")
            with self.assertRaisesRegex(ValueError, ":1:"): PUB.read_index(idx, d, "doc", "src")

    def test_writer_of_callback_groups_rows(self):
        with tempfile.TemporaryDirectory() as d:
            idx = Path(d) / "index.tsv"; idx.write_text("w1/a.png\tx\nw2/b.png\ty\n")
            rows = PUB.read_index(idx, d, "doc", "src", writer_of=lambda rel: rel.split("/")[0])
            self.assertEqual({r["writer_id"] for r in rows}, {"w1", "w2"})


class BanglaWritingAndBnhtrd(unittest.TestCase):
    def test_banglawriting_groups_by_first_path_component(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "index.tsv").write_text("writer_03/p1.png\tকথা\nwriter_03/p2.png\tভাষা\nwriter_07/p1.png\tদেশ\n")
            rows = BW.import_dataset(d)
            self.assertEqual(len(rows), 3)
            self.assertEqual({r["writer_id"] for r in rows}, {"writer_03", "writer_07"})
            self.assertTrue(all(r["document_type"] == "banglawriting_line" for r in rows))

    def test_bnhtrd_same_shape_different_document_type(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "index.tsv").write_text("w1/p1.png\tপরীক্ষা\n")
            rows = BH.import_dataset(d)
            self.assertEqual(rows[0]["document_type"], "bnhtrd_line")
            self.assertEqual(rows[0]["source"], "bnhtrd")

    def test_missing_dataset_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): BW.import_dataset(d)


class BanglaLekha(unittest.TestCase):
    def test_load_pool_groups_images_by_folder(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for ch, files in {"ক": ["a.png", "b.png"], "খ": ["c.jpg"]}.items():
                sub = root / ch; sub.mkdir()
                for f in files: (sub / f).write_bytes(b"\x89PNG fake")
            pool = BL.load_pool(root)
            self.assertEqual(set(pool), {"ক", "খ"})
            self.assertEqual(len(pool["ক"]), 2)

    def test_class_map_renames_numeric_folders(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "12").mkdir(); (root / "12" / "a.png").write_bytes(b"x")
            pool = BL.load_pool(root, class_map={"12": "ক"})
            self.assertEqual(set(pool), {"ক"})

    def test_sample_glyph_returns_none_for_missing_character(self):
        import random
        self.assertIsNone(BL.sample_glyph({}, "ক", random.Random(0)))

    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError): BL.load_pool("/nonexistent/path/xyz")


if __name__ == "__main__":
    unittest.main()
