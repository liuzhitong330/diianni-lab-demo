"""Fast deterministic formula/export tests; no network, raw data or TF required."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from analysis import BANDS, BUDGETS, SAMPLE_IDS, ARCHIVES, diagnostics, reference_masks

HERE = Path(__file__).resolve().parent


class FormulaTests(unittest.TestCase):
    def test_exhaustive_nonoverlapping_bounds(self):
        # Includes exact boundaries and below-floor values; maximum belongs once.
        y = np.array([[1, .1, .01, .001, .0001, .00001]])
        masks = reference_masks(y)
        self.assertTrue(np.all(np.stack(masks).sum(axis=0) == 1))
        self.assertEqual([int(m.sum()) for m in masks], [1,1,1,1,2])

    def test_identity(self):
        y = np.array([[1, .01], [.0001, .000001]])
        d = diagnostics(y,y)
        self.assertEqual(d["nmse"], 0)
        self.assertEqual(d["whole_bias"], 0)
        self.assertEqual(d["whole_absolute_error_percent"], 0)
        self.assertEqual(d["negative_count"], 0)
        self.assertAlmostEqual(sum(b["power_share_percent"] for b in d["bands"]),100)

    def test_multiplicative_bias(self):
        y = np.array([[1, .1, .01, .001, .0001, .00001]])
        d = diagnostics(y, y*1.2)
        self.assertAlmostEqual(d["nmse"], .04)
        self.assertAlmostEqual(d["whole_bias"], 20)
        for band in d["bands"]:
            self.assertAlmostEqual(band["signed_bias_percent"],20)
            self.assertAlmostEqual(band["normalized_absolute_error_band_percent"],20)
        self.assertAlmostEqual(sum(b["absolute_error_percent_total_reference"] for b in d["bands"]),20)

    def test_signed_cancellation_and_negatives_not_clipped(self):
        y = np.array([[1.,1.]])
        d = diagnostics(y, np.array([[-1.,3.]]))
        self.assertEqual(d["negative_count"],1)
        self.assertEqual(d["whole_bias"],0)
        self.assertEqual(d["whole_absolute_error_percent"],200)
        self.assertEqual(d["nmse"],4)
        self.assertEqual(d["bands"][-1]["normalized_absolute_error_band_percent"],200)

    def test_empty_bands_are_null_not_zero_bias(self):
        d = diagnostics(np.ones((2,2)),np.ones((2,2)))
        for band in d["bands"][:-1]:
            self.assertEqual(band["pixels"],0)
            self.assertIsNone(band["signed_bias_percent"])
            self.assertIsNone(band["normalized_absolute_error_band_percent"])

    def test_invalid_values_rejected(self):
        for y,p in [(np.array([0.]),np.array([1.])),(np.array([1.]),np.array([np.nan])),
                    (np.array([np.inf]),np.array([1.])),(np.ones((2,)),np.ones((3,)))]:
            with self.assertRaises(ValueError): diagnostics(y,p)

    def test_compression_definition(self):
        self.assertEqual([100*(1-k/500) for k in BUDGETS],[75,80,85,90,95])


@unittest.skipUnless((HERE / "analysis.json").exists(), "Generate outputs first for export checks")
class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((HERE / "analysis.json").read_text())

    def test_exact_pair_coverage(self):
        rows = self.data["rows"]
        self.assertEqual(len(rows),200)
        self.assertEqual({(r["id"],r["k"]) for r in rows},{(sid,k) for sid in SAMPLE_IDS for k in BUDGETS})
        for r in rows:
            self.assertEqual(sum(b["pixels"] for b in r["bands"]),9216)
            self.assertAlmostEqual(sum(b["power_share_percent"] for b in r["bands"]),100,places=10)
            self.assertAlmostEqual(sum(b["absolute_error_percent_total_reference"] for b in r["bands"]),r["whole_absolute_error_percent"],places=10)
            reconstructed = sum((b["signed_bias_percent"] or 0)*b["power_share_percent"]/100 for b in r["bands"])
            self.assertAlmostEqual(reconstructed,r["whole_bias"],places=10)

    def test_display_payload_error_tolerance(self):
        # Rounded web arrays are a display derivative, not the source of results.
        for sid in SAMPLE_IDS:
            s = json.loads((HERE / "samples" / f"{sid}.json").read_text())
            self.assertEqual(s["id"],sid)
            self.assertEqual(s["shape"],[96,96])
            y = np.array(s["reference"])
            self.assertEqual(y.size,9216)
            self.assertEqual(set(s["predictions"]),set(map(str,BUDGETS)))
            for k in BUDGETS:
                p = np.array(s["predictions"][str(k)])
                r = next(r for r in self.data["rows"] if r["id"]==sid and r["k"]==k)
                self.assertEqual(p.size,9216)
                self.assertTrue(np.isfinite(p).all())
                # Compare whole-image quantities; rounding can move a boundary pixel.
                self.assertAlmostEqual(float(np.sum((p-y)**2)/np.sum(y*y)),r["nmse"],places=5)
                self.assertAlmostEqual(float(100*np.sum(p-y)/np.sum(y)),r["whole_bias"],places=4)
                self.assertEqual(int((p<0).sum()),r["negative_count"])

    def test_manifest_hashes(self):
        p = json.loads((HERE / "PROVENANCE.json").read_text())
        self.assertEqual(p["archives"],ARCHIVES)
        self.assertEqual(set(p["sample_sha256"]),set(SAMPLE_IDS))
        self.assertEqual(set(p["model_sha256"]),set(map(str,BUDGETS)))
        for group in [p["sample_sha256"],p["model_sha256"],p["upstream_file_sha256"]]:
            self.assertTrue(all(len(v)==64 and int(v,16)>=0 for v in group.values()))
        self.assertFalse(p["trained_here"])
        self.assertFalse(p["independent_validation"])

    def test_published_reproduction(self):
        self.assertTrue(all(r["within_1e_minus4"] for r in self.data["reproduction_checks"]))
        self.assertEqual([round(x["repository_ssim_mean"],4) for x in self.data["summary"]["by_budget"]],
                         [.9166,.9041,.8915,.8619,.8154])


if __name__ == "__main__":
    unittest.main(verbosity=2)
