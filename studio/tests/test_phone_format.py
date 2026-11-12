"""The app's CorrectionsLog (Dart) writes what Studio's importer (Python) accepts. Needs `dart`; skipped otherwise."""
import shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import corrections as C

CHECK = HERE.parent / "cardpack" / "dart_check"


@unittest.skipUnless(shutil.which("dart") and (CHECK / "lib" / "corrections_log.dart").exists(), "needs dart and the dart_check project")
class DartToPython(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run(["dart", "run", "bin/corrections_check.dart", d], cwd=CHECK, capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.strip(), "ok,ok,null,null,null,null,ok,ok,null")   # only the valid ones recorded; a name as the writer is refused by the app
            import cards
            old = (C.IMPORTS, cards.WORK, cards.CARDS); C.IMPORTS = Path(d) / "imports"; cards.WORK = Path(d) / "work"; cards.CARDS = Path(d) / "cards"
            try:
                rep = C.import_phone_folder(d)
                self.assertEqual((rep["records"], rep["imported"], rep["rejected"]), (4, 4, []))
                got = {(r["row"], r["col"], r["label"], r["pack"], r["pack_version"], r["source"]) for r in C.read_imported()}
                self.assertEqual(got, {("pulse", "c3", "98", "anc", "4", "phone"), ("bp", "c1", "120/80", "anc", "4", "phone"), ("weight_kg", "c1", "57", "anc", "4", "phone"), ("pulse", "c1", "88", "anc", "4", "phone")})
                pages = {(r["row"], r["label"]): r["page"] for r in C.read_imported()}
                self.assertEqual(pages[("weight_kg", "57")], pages[("pulse", "88")])                 # the two cells of one photographed card share a page
                self.assertEqual(cards.writers("anc"), {pages[("weight_kg", "57")]: "W07"})          # and the app's writer code reached the writer table
            finally:
                C.IMPORTS, cards.WORK, cards.CARDS = old


if __name__ == "__main__":
    unittest.main()
