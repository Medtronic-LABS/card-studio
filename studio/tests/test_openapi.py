"""The OpenAPI spec and Swagger UI page (offline: swagger-ui is vendored). Run: python -m unittest discover -s tests"""
import json, subprocess, sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "tools"))
import gen_openapi


class Spec(unittest.TestCase):
    def test_checked_in_spec_matches_the_routes(self):
        self.assertEqual((HERE / "static" / "openapi.json").read_text(), json.dumps(gen_openapi.build(), indent=1) + "\n", "run python tools/gen_openapi.py")

    def test_known_routes_and_params(self):
        paths = gen_openapi.build()["paths"]
        self.assertIn("post", paths["/api/train/start"]); self.assertIn("get", paths["/api/datasets"])
        self.assertEqual([p["name"] for p in paths["/api/card/{card}/split/preview"]["get"]["parameters"]], ["card"])
        self.assertTrue(all("\0" not in k and "{}" not in k for k in paths))

    def test_docs_page_has_no_cdn(self):
        html = (HERE / "static" / "docs.html").read_text()
        self.assertNotIn("http", html.replace("http-equiv", ""))
        for f in ("swagger-ui.css", "swagger-ui-bundle.js"): self.assertTrue((HERE / "static" / "vendor" / f).exists())


if __name__ == "__main__":
    unittest.main()
