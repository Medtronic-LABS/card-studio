"""Layout detection: names must be unique so a detected layout can be saved as-is. Run: python -m unittest discover -s tests"""
import sys, unittest
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import detect


def card(boxes_in_cell):
    im = np.full((700, 1000, 3), 250, np.uint8)
    for x in range(100, 901, 160): cv2.line(im, (x, 100), (x, 600), (30, 30, 30), 3)
    for y in range(100, 601, 100): cv2.line(im, (100, y), (900, y), (30, 30, 30), 3)
    x0, y0 = 120, 120                                     # first cell: several hollow checkboxes in one cell
    for i in range(boxes_in_cell): cv2.rectangle(im, (x0 + i * 38, y0), (x0 + i * 38 + 28, y0 + 28), (30, 30, 30), 2)
    return im


class Names(unittest.TestCase):
    def test_detected_names_are_unique(self):
        for n in (0, 2, 3):
            lay = detect.detect_layout(card(n))
            names = [f["name"] for f in lay["fields"]]
            self.assertEqual(len(names), len(set(names)), names)
        lay = detect.detect_layout(card(3))
        names = [f["name"] for f in lay["fields"]]
        self.assertEqual(len(names), len(set(names)), names)

    def test_unique_names_helper(self):
        fs = [{"name": "a_box"}, {"name": "a_box"}, {"name": "a_box"}, {"name": "b"}]
        self.assertEqual(detect.unique_names(fs), 2)
        self.assertEqual([f["name"] for f in fs], ["a_box", "a_box_2", "a_box_3", "b"])
        self.assertEqual(detect.unique_names(fs), 0)           # idempotent


if __name__ == "__main__":
    unittest.main()
