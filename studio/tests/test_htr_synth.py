"""htr.data.synth: random writable text, font rendering (shaping-support detection, real rendering when a font is available on the
machine - never committed to the repo), and the digit-compositor delegation to studio/synthgen.py."""
import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import htr.data.synth as S
import htr.text as T
import htr.vocab as V

try:
    import PIL  # noqa: F401
except ImportError:
    PIL = None

# A real Bengali font this dev machine happens to have (Android SDK assets, never part of this repo). Tests that need actual
# rendering skip cleanly when none is found - no font is ever read from or written into the repo tree. Exact candidate paths only
# (no recursive filesystem scan: that can take minutes on a real machine's SDK/Library trees).
_CANDIDATES = [
    Path.home() / "Library/Android/sdk/platforms/android-28/data/fonts/NotoSansBengali-Regular.ttf",
    Path("/System/Library/Fonts/Supplemental/NotoSansBengali-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf"),
]
_SYSTEM_FONT = next((str(p) for p in _CANDIDATES if p.exists()), None)


class RandomText(unittest.TestCase):
    def test_random_word_is_writable(self):
        vocab = V.Vocabulary.default(); rng = __import__("random").Random(0)
        for _ in range(100):
            w = S.random_word(vocab, rng)
            self.assertEqual(T.problems(w), [], w)
            self.assertGreaterEqual(len(w), 2, w)   # each of 2-8 iterations adds at least one consonant; a sign/virama can add more

    def test_random_line_is_space_separated_writable_words(self):
        vocab = V.Vocabulary.default(); rng = __import__("random").Random(1)
        line = S.random_line(vocab, rng, n_words=(2, 2))
        self.assertEqual(len(line.split(" ")), 2)
        for w in line.split(" "): self.assertEqual(T.problems(w), [])


class Fonts(unittest.TestCase):
    def test_available_fonts_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(S.available_fonts(d), [])

    def test_available_fonts_finds_ttf(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "Some-Font.ttf").write_bytes(b"not a real font, just a file to find")
            self.assertEqual(len(S.available_fonts(d)), 1)

    def test_shaping_support_returns_a_reason_either_way(self):
        ok, reason = S.shaping_support()
        self.assertIsInstance(ok, bool)
        if not ok: self.assertIn("raqm", reason.lower())

    def test_generate_font_lines_refuses_with_no_fonts(self):
        vocab = V.Vocabulary.default()
        with tempfile.TemporaryDirectory() as fonts, tempfile.TemporaryDirectory() as out:
            with self.assertRaises(FileNotFoundError): S.generate_font_lines(vocab, fonts, out, n=3)


@unittest.skipIf(PIL is None, "Pillow not installed")
@unittest.skipIf(_SYSTEM_FONT is None, "no Bengali font found on this machine to render with")
class RealRender(unittest.TestCase):
    def test_render_line_produces_a_real_image(self):
        img, used_raqm = S.render_line("পরীক্ষা", _SYSTEM_FONT, height=48)
        self.assertEqual(img.shape[0], 48)
        self.assertGreater(img.shape[1], 4)
        self.assertLess(int(img.min()), 200)   # some dark ink actually drawn, not a blank white image
        self.assertIsInstance(used_raqm, bool)

    def test_generate_font_lines_writes_rows_and_files(self):
        vocab = V.Vocabulary.default()
        with tempfile.TemporaryDirectory() as fonts, tempfile.TemporaryDirectory() as out:
            import shutil
            shutil.copy(_SYSTEM_FONT, Path(fonts) / "Noto.ttf")
            rows, report = S.generate_font_lines(vocab, fonts, out, n=3, seed=0)
            self.assertEqual(len(rows), 3)
            for r in rows:
                self.assertTrue(Path(r["image_path"]).exists())
                self.assertEqual(r["document_type"], "synthetic")
                self.assertTrue(r["writer_id"].startswith("font:"))
            self.assertIn("shaping_support", report)


class DigitCompositor(unittest.TestCase):
    def test_delegates_to_synthgen_and_skips_without_numtadb(self):
        sys.path.insert(0, str(ROOT / "studio"))
        import synthgen
        try:
            synthgen.load_pool()
        except OSError:
            with tempfile.TemporaryDirectory() as out:
                with self.assertRaises(OSError): S.generate_digit_cells("anc", out, n=2)
            self.skipTest("needs the NumtaDB digits in data/bangla_digits (OSError path exercised above)")


if __name__ == "__main__":
    unittest.main()
