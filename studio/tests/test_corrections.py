"""Label import (phone corrections zip, Label Studio export). Run: python -m unittest discover -s tests"""
import csv, io, json, sys, tempfile, unittest, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import corrections as C

PNG = C.PNG + b"\x00" * 40


def zip_of(records, crops=None, extra=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("corrections.jsonl", "\n".join(json.dumps(r) if isinstance(r, dict) else r for r in records))
        for n, b in (crops if crops is not None else {r["crop"]: PNG for r in records if isinstance(r, dict) and "crop" in r}).items():
            z.writestr(f"crops/{n}", b)
        for n, b in (extra or {}).items(): z.writestr(n, b)
    return buf.getvalue()


def rec(i=1, **kw):
    r = {"id": f"aaaa{i:04d}-bbbb-cccc", "pack_id": "anc", "pack_version": 4, "field": "pulse", "col": "c3", "label": "98", "crop": f"c{i}.png", "created": "2026-11-04T10:00:00"}
    r.update(kw); return r


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.old = C.IMPORTS; C.IMPORTS = Path(self.tmp.name) / "imports"

    def tearDown(self):
        C.IMPORTS = self.old; self.tmp.cleanup()


class Phone(Base):
    def test_good_import_writes_label_and_crop(self):
        rep = C.import_phone_zip(zip_of([rec(1), rec(2, field="weight_kg", col="c1", label="57")]))
        self.assertEqual((rep["imported"], rep["new"], rep["rejected"]), (2, 2, []))
        rows = C.read_imported()
        self.assertEqual({(r["row"], r["col"], r["label"], r["source"], r["pack"], r["pack_version"]) for r in rows},
                         {("pulse", "c3", "98", "phone", "anc", "4"), ("weight_kg", "c1", "57", "phone", "anc", "4")})
        for r in rows: self.assertTrue((C.IMPORTS / r["crop"]).read_bytes().startswith(C.PNG))

    def test_reimport_is_idempotent_and_newest_wins(self):
        C.import_phone_zip(zip_of([rec(1)]))
        rep = C.import_phone_zip(zip_of([rec(1, label="99")]))
        self.assertEqual((rep["new"], rep["changed"]), (0, 1))
        self.assertEqual([r["label"] for r in C.read_imported()], ["99"])

    def test_each_rejection(self):
        bad = [rec(1, id="x"), rec(2, field="../etc"), rec(3, col="3"), rec(4, label=""), rec(5, label="9" * 41), rec(6, crop="../x.png"),
               rec(7, crop="missing.png"), rec(8, pack_version="4"), "not json", rec(9, label=5)]
        rep = C.import_phone_zip(zip_of(bad, crops={"c1.png": PNG, "c2.png": PNG, "c3.png": PNG, "c4.png": PNG, "c5.png": PNG, "c8.png": PNG, "c9.png": PNG}))
        self.assertEqual(rep["imported"], 0, rep)
        self.assertEqual(len(rep["rejected"]), len(bad))
        self.assertEqual(C.read_imported(), [])

    def test_crop_must_be_a_small_png(self):
        rep = C.import_phone_zip(zip_of([rec(1), rec(2)], crops={"c1.png": b"GIF89a....", "c2.png": C.PNG + b"x" * (C.MAX_CROP + 1)}))
        self.assertEqual(rep["imported"], 0)

    def test_not_a_zip_and_missing_jsonl(self):
        with self.assertRaises(ValueError): C.import_phone_zip(b"hello")
        buf = io.BytesIO(); zipfile.ZipFile(buf, "w").writestr("x.txt", "x")
        with self.assertRaises(ValueError): C.import_phone_zip(buf.getvalue())

    def test_nothing_is_written_outside_imports(self):
        before = {p for p in HERE.parent.glob("work/*.csv")}
        C.import_phone_zip(zip_of([rec(1)], extra={"../evil.txt": b"x"}))
        self.assertEqual({p for p in HERE.parent.glob("work/*.csv")}, before)
        self.assertFalse((Path(self.tmp.name) / "evil.txt").exists())

    def test_folder_import(self):
        d = Path(self.tmp.name) / "pulled"; (d / "crops").mkdir(parents=True)
        (d / "corrections.jsonl").write_text(json.dumps(rec(1)) + "\n"); (d / "crops" / "c1.png").write_bytes(PNG)
        self.assertEqual(C.import_phone_folder(d)["imported"], 1)


class LabelStudio(Base):
    def setUp(self):
        super().setUp()
        self.crops = Path(self.tmp.name) / "work_crops"
        for f, c, p in (("pulse", "c3", "p-001"), ("bp", "c2", "p-007")):
            (self.crops / f / c).mkdir(parents=True, exist_ok=True); (self.crops / f / c / f"{p}.png").write_bytes(PNG)

    def task(self, image, text, i=1, cancelled=False):
        return {"id": i, "data": {"image": image}, "annotations": [{"was_cancelled": cancelled, "result": [{"type": "textarea", "value": {"text": [text]}}]}]}

    def test_export_with_prefix_and_skips(self):
        tasks = [self.task("/data/upload/1/ab12cd34-pulse_c3_p-001.png", "88", 1), self.task("/x/bp_c2_p-007.png", "120/80", 2),
                 self.task("/x/pulse_c3_p-999.png", "77", 3),            # crop does not exist
                 self.task("/x/weird.png", "1", 4), self.task("/x/bp_c2_p-007.png", "1", 5, cancelled=True),
                 {"id": 6, "data": {"image": "/x/bp_c2_p-007.png"}, "annotations": []}]
        rep = C.import_label_studio(json.dumps(tasks).encode(), crops_root=self.crops)
        self.assertEqual((rep["imported"], len(rep["skipped"])), (2, 4), rep)
        self.assertEqual({(r["row"], r["col"], r["page"], r["label"], r["source"], r["crop"]) for r in C.read_imported()},
                         {("pulse", "c3", "p-001", "88", "label-studio", ""), ("bp", "c2", "p-007", "120/80", "label-studio", "")})

    def test_bad_export(self):
        with self.assertRaises(ValueError): C.import_label_studio(b'{"not": "a list"}', crops_root=self.crops)


class CardAndWriter(Base):
    def setUp(self):
        super().setUp(); import cards
        self.old_cards = (cards.WORK, cards.CARDS); cards.WORK = Path(self.tmp.name) / "work"; cards.CARDS = Path(self.tmp.name) / "cards"

    def tearDown(self):
        import cards
        cards.WORK, cards.CARDS = self.old_cards; super().tearDown()

    def test_cells_of_one_photographed_card_share_a_page_and_the_writer_is_recorded(self):
        import cards
        rep = C.import_phone_zip(zip_of([rec(1, card="c0ffee0012345678", writer="W07"), rec(2, field="weight_kg", col="c1", label="57", card="c0ffee0012345678", writer="W07"),
                                         rec(3, card="deadbeef87654321", writer="W08"), rec(4, card="feedface11223344")]))
        self.assertEqual((rep["imported"], rep["rejected"], rep["writers_recorded"]), (4, [], 2))
        self.assertEqual(sorted({r["page"] for r in C.read_imported()}), ["ph-c0ffee001234", "ph-deadbeef8765", "ph-feedface1122"])
        self.assertEqual({r["page"] for r in C.read_imported() if r["row"] == "weight_kg"}, {"ph-c0ffee001234"})
        self.assertEqual(cards.writers("anc"), {"ph-c0ffee001234": "W07", "ph-deadbeef8765": "W08"})          # the card with no writer has none

    def test_a_name_instead_of_a_code_or_a_bad_card_id_rejects_that_record_only(self):
        rep = C.import_phone_zip(zip_of([rec(1, writer="Dr Rahman"), rec(2, card="../x"), rec(3, writer="W01")]))
        self.assertEqual((rep["imported"], [b["line"] for b in rep["rejected"]]), (1, [1, 2])); self.assertIn("not a name", rep["rejected"][0]["reason"])

    def test_old_records_without_a_card_still_import_one_page_each(self):
        rep = C.import_phone_zip(zip_of([rec(1), rec(2, field="weight_kg", col="c1", label="57")]))
        self.assertEqual((rep["imported"], len({r["page"] for r in C.read_imported()}), rep["writers_recorded"]), (2, 2, 0))


class MergeIntoTraining(Base):
    def test_trainer_merge_includes_imports_and_never_edits_labels_csv(self):
        sys.path.insert(0, str(HERE.parent / "work_phaseA")); sys.path.insert(0, str(HERE.parent / "cardpack"))
        import trainer
        labels = HERE.parent / "work" / "labels.csv"
        before = labels.read_bytes() if labels.exists() else None
        C.import_phone_zip(zip_of([rec(1, field="pulse", col="c3", label="98")]))
        out = Path(self.tmp.name) / "merged.csv"
        rows = trainer.merge_labels(out)
        mine = [r for r in rows if r["page"].startswith("ph-")]
        self.assertEqual([(r["row"], r["col"], r["label"]) for r in mine], [("pulse", "c3", "98")])
        self.assertTrue(Path(mine[0]["crop"]).exists())
        self.assertEqual(labels.read_bytes() if labels.exists() else None, before)


if __name__ == "__main__":
    unittest.main()
