#!/usr/bin/env python3
"""Reproduce released Deep-fUS inference and audit matched signal-band errors.

Copyright 2026 Cathy Liu. AI-assisted analysis implementation.
Apache-2.0. Preprocessing and repository-defined metrics follow Tommaso Di
Ianni's Apache-2.0 deep-fus code (Copyright 2021 Tommaso Di Ianni), pinned below.
Modified here: explicit source IDs, deterministic full coverage, band diagnostics,
integrity checks, compact web exports, and a reproducible review table.
No training, prospective acquisition, or functional validation is performed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np

COMMIT = "080ed97a521bbebae5bf71b3cc85a2a11c5ba682"
REPO = "https://github.com/todiian/deep-fus"
XMEAN, XSTD = -0.5237595494149918, 131526.6016974602
BUDGETS = [125, 100, 75, 50, 25]
SAMPLE_IDS = [f"fr{i}" for i in range(1, 41)]
BANDS = [
    {"id": "below_40", "lo": None, "hi": -40, "label": "Below −40 dB"},
    {"id": "minus40_minus30", "lo": -40, "hi": -30, "label": "−40 to −30 dB"},
    {"id": "minus30_minus20", "lo": -30, "hi": -20, "label": "−30 to −20 dB"},
    {"id": "minus20_minus10", "lo": -20, "hi": -10, "label": "−20 to −10 dB"},
    {"id": "minus10_zero", "lo": -10, "hi": 0, "label": "−10 to 0 dB"},
]
ARCHIVES = {
    "test_data.zip": {
        "url": "https://ndownloader.figshare.com/files/41528487",
        "sha256": "28b0cba98af87110b5049222da51c33de1973ea1b3ecedea3ac8ea7b60877ed8",
        "bytes": 426605576, "doi": "10.6084/m9.figshare.23663970.v1",
    },
    "pretrained_models.zip": {
        "url": "https://ndownloader.figshare.com/files/41528931",
        "sha256": "918919f517b5ee09b07ed9b8461ac19aae36e5ccead6439ec27187fc24ede287",
        "bytes": 408685445, "doi": "10.6084/m9.figshare.23663979.v2",
    },
}
MODEL_HASHES = {
    125: "5c7957fde0819e45f6329400cd231ef7ec5954166c88d31a9224b878ff811720",
    100: "1a7f3a53ae150a8053dcf64706729deecc18d8856bd5e595c6939b6b06262b6b",
    75: "8fb70ec9942d23aac3447a83a1f7248d0f1b33067e07fa30375fc5dda865e784",
    50: "c46aa5292e92f17adcb28c39a1b61cd6fea2776bec081b2059d0222c6249d2bf",
    25: "46b4791556255cc30181e12c213961562ff4abcd5df7329d9b598e8c3ab96df0",
}
UPSTREAM_FILES = {
    "README.md": "286096e5b678f443d76d72c11916295676c7a28e583596ff757563a0a80dc30c",
    "LICENSE": "1889a215b515a702b7ea141cfb549e7ee4d2843adca57a5caa28300d72f3e8ad",
    "src/predict.py": "6e753cd9f21f579aa30b229250e13e3da9af9feeab9df97f20324b052cfba740",
    "src/utils.py": "2fba07c914f77ffdab6028d9a05eb8bab48379e26f0c3272cb6bb459d7540dc3",
    "src/models.py": "8c8fcd88330f5ec32cb8f9fc21eefd564c78a8837d1d10ed5d94f83bb44c4e2f",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_file(url: str, digest: str, path: Path) -> None:
    """Never silently trust or replace an existing mismatched input."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "Deep-fUS-reanalysis/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f, length=8 * 1024 * 1024)
        if sha256(partial) != digest:
            raise ValueError(f"Download checksum mismatch: {partial}")
        partial.rename(path)
    if sha256(path) != digest:
        raise ValueError(f"Existing input checksum mismatch: {path}; use a fresh cache.")


def extract_member(archive: zipfile.ZipFile, member: str, cache: Path) -> Path:
    """Only explicitly named expected members are extracted; no arbitrary paths."""
    p = cache / member
    raw = archive.read(member)  # ZIP CRC is also checked by zipfile.
    digest = hashlib.sha256(raw).hexdigest()
    if p.exists():
        if sha256(p) != digest:
            raise ValueError(f"Extracted input differs from verified archive: {p}")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    return p


def prepare_inputs(cache: Path) -> None:
    for name, metadata in ARCHIVES.items():
        print(f"Checking {name}", flush=True)
        ensure_file(metadata["url"], metadata["sha256"], cache / name)
    with zipfile.ZipFile(cache / "test_data.zip") as z:
        for sid in SAMPLE_IDS:
            extract_member(z, f"test/{sid}.mat", cache)
    with zipfile.ZipFile(cache / "pretrained_models.zip") as z:
        for k in BUDGETS:
            p = extract_member(z, f"pretrained_models/ResUNet5_{k}/my_model.h5", cache)
            if sha256(p) != MODEL_HASHES[k]:
                raise ValueError(f"Unexpected model weights: {p}")
    for filename, digest in UPSTREAM_FILES.items():
        ensure_file(f"https://raw.githubusercontent.com/todiian/deep-fus/{COMMIT}/{filename}",
                    digest, cache / "upstream-pin" / filename)


def reference_masks(reference: np.ndarray) -> list[np.ndarray]:
    """Masks use the native reference only. All finite positive pixels included."""
    y = np.asarray(reference, dtype=np.float64)
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("Reference must contain finite, strictly positive power.")
    db = 10 * np.log10(y / y.max())
    masks = []
    for i, band in enumerate(BANDS):
        lower = np.ones(y.shape, dtype=bool) if band["lo"] is None else db >= band["lo"]
        upper = db <= band["hi"] if i == len(BANDS) - 1 else db < band["hi"]
        masks.append(lower & upper)
    if not np.all(np.sum(np.stack(masks), axis=0) == 1):
        raise ValueError("Signal bands must partition every reference pixel exactly once.")
    return masks


def diagnostics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    """Unclipped native-power errors; no inference about vessel or neural activity."""
    y, p = np.asarray(reference, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Prediction must be finite and match the reference shape.")
    masks = reference_masks(y)
    total = float(y.sum())
    bands = []
    for definition, mask in zip(BANDS, masks):
        power = float(y[mask].sum())
        absolute = float(np.abs(p[mask] - y[mask]).sum())
        bands.append({
            **{key: definition[key] for key in ("id", "lo", "hi")},
            "pixels": int(mask.sum()),
            "power_share_percent": 100 * power / total,
            "signed_bias_percent": 100 * float((p[mask] - y[mask]).sum()) / power if power else None,
            "absolute_error_percent_total_reference": 100 * absolute / total,
            "normalized_absolute_error_band_percent": 100 * absolute / power if power else None,
        })
    return {
        "nmse": float(np.sum((p-y)**2) / np.sum(y**2)),
        "whole_bias": 100 * float((p-y).sum()) / total,
        "whole_absolute_error_percent": 100 * float(np.abs(p-y).sum()) / total,
        "negative_count": int((p < 0).sum()),
        "bands": bands,
    }


def rounded_flat(array: np.ndarray) -> list[float]:
    # Display payload only. All metrics are calculated before this rounding.
    return [float(f"{float(v):.7g}") for v in np.asarray(array).reshape(-1)]


def summaries(rows: list[dict]) -> dict:
    output = []
    for k in BUDGETS:
        selected = [r for r in rows if r["k"] == k]
        output.append({
            "k": k, "compression": 100 * (1-k/500), "n_pairs": len(selected),
            "repository_ssim_mean": float(np.mean([r["repository_ssim"] for r in selected])),
            "repository_ssim_sd": float(np.std([r["repository_ssim"] for r in selected])),
            "nmse_mean": float(np.mean([r["nmse"] for r in selected])),
            "nmse_sd": float(np.std([r["nmse"] for r in selected])),
            "whole_bias_median": float(np.median([r["whole_bias"] for r in selected])),
            "negative_count": sum(r["negative_count"] for r in selected),
            "bands": [{
                "id": definition["id"],
                "nonempty_pairs": sum(r["bands"][j]["pixels"] > 0 for r in selected),
                "signed_bias_median": float(np.median([r["bands"][j]["signed_bias_percent"] for r in selected if r["bands"][j]["pixels"]])),
                "power_share_median": float(np.median([r["bands"][j]["power_share_percent"] for r in selected])),
            } for j, definition in enumerate(BANDS)],
        })
    return {"by_budget": output, "n_pairs": 40, "n_models": 5, "n_reconstructions": len(rows),
            "not_independent_animals": True, "confidence_intervals": "Not calculated: animal/session mapping unavailable."}


def export_csv(rows: list[dict], path: Path) -> None:
    base = ["id", "k", "compression", "repository_ssim", "nmse", "whole_bias", "whole_absolute_error_percent", "negative_count"]
    bandfields = ["id", "lo", "hi", "pixels", "power_share_percent", "signed_bias_percent",
                  "absolute_error_percent_total_reference", "normalized_absolute_error_band_percent"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base + ["band_" + x for x in bandfields])
        writer.writeheader()
        for row in rows:
            for band in row["bands"]:
                writer.writerow({**{x: row[x] for x in base}, **{"band_" + x: band[x] for x in bandfields}})


def figures(data: dict, references: np.ndarray, predictions: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    series = data["summary"]["by_budget"]
    for ax, name, label in zip(axes, ["repository_ssim", "nmse"], ["Repository-defined SSIM", "NMSE"]):
        ax.errorbar([x["compression"] for x in series], [x[name+"_mean"] for x in series],
                    yerr=[x[name+"_sd"] for x in series], color="#376d68", marker="o", capsize=3)
        ax.set(xlabel="Beamformed-data compression (%)", ylabel=label)
    fig.suptitle("Published test evidence reproduced: 40 matched pairs, five released models")
    fig.text(.5, .01, "Points: pair mean; bars: pair SD, not confidence intervals or independent animals.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95)); fig.savefig(out / "01-reproduction.svg", metadata={"Date": None}); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
    for ax, j in zip(axes, [1, 2]):
        for k, color in [(125, "#376d68"), (25, "#9b5945")]:
            r = [r for r in data["rows"] if r["k"] == k]
            ax.scatter([a["bands"][j]["power_share_percent"] for a in r],
                       [a["bands"][j]["signed_bias_percent"] for a in r], s=24, alpha=.75, color=color, label=f"{k} real frames")
        ax.axhline(0, lw=.7, color="0.4"); ax.set_xscale("log")
        ax.set(title=BANDS[j]["label"], xlabel="Band's share of reference power (%)")
    axes[0].set_ylabel("Signed aggregate band bias (%)"); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Relative bias must be read alongside signal mass")
    fig.text(.5, .01, "Each point is one image pair. Low-power bands are not validated vessels or neural states.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94)); fig.savefig(out / "02-band-bias-context.svg", metadata={"Date": None}); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache"), help="Private raw inputs/model cache (~2GB; do not publish).")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    cache, out = args.cache.resolve(), args.output.resolve()
    cache.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    prepare_inputs(cache)
    import scipy.io as sio
    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    references, input_hashes = [], {}
    for sid in SAMPLE_IDS:
        path = cache / "test" / f"{sid}.mat"
        sample = sio.loadmat(path)
        if sample["x"].shape != (96,96,250) or sample["y"].shape != (96,96):
            raise ValueError(f"Unexpected dimensions: {sid}")
        if np.iscomplexobj(sample["x"]) or not np.isfinite(sample["x"]).all():
            raise ValueError(f"Non-real/nonfinite input: {sid}")
        reference_masks(sample["y"])
        references.append(sample["y"])
        input_hashes[sid] = sha256(path)
    references = np.stack(references)
    predictions = np.zeros((5,40,96,96), dtype=np.float32)
    rows = []
    for model_idx, k in enumerate(BUDGETS):
        model = tf.keras.models.load_model(cache / "pretrained_models" / f"ResUNet5_{k}" / "my_model.h5", compile=False)
        if model.input_shape != (None,96,96,k) or model.output_shape != (None,96,96):
            raise ValueError(f"Model input/output mismatch: k={k}")
        for sample_idx, sid in enumerate(SAMPLE_IDS):
            raw = sio.loadmat(cache / "test" / f"{sid}.mat")["x"]
            x = ((raw[:,:,:k] - XMEAN)/XSTD).astype(np.float32)[None]
            p = model([x], training=False).numpy()[0]
            predictions[model_idx,sample_idx] = p
            y = references[sample_idx]
            # Intentional exact replication of the upstream SSIM definition.
            ssim = float(tf.image.ssim(y[None,:,:,None].astype(np.float32), p[None,:,:,None], max_val=1, filter_size=3))
            rows.append({"id": sid, "k": k, "compression": 100*(1-k/500),
                         "repository_ssim": ssim, **diagnostics(y,p)})
        print(f"Completed {k} real frames: 40 matched predictions", flush=True)
        tf.keras.backend.clear_session()
    summary = summaries(rows)
    expected_ssim = [.9166,.9041,.8915,.8619,.8154]
    expected_nmse = [.0399,.0448,.0506,.0701,.1106]
    checks = []
    for result, ss, nm in zip(summary["by_budget"], expected_ssim, expected_nmse):
        passed = abs(result["repository_ssim_mean"]-ss) < .0001 and abs(result["nmse_mean"]-nm) < .0001
        checks.append({"k": result["k"], "paper_ssim_4dp": ss, "paper_nmse_4dp": nm, "within_1e_minus4": passed})
    if not all(c["within_1e_minus4"] for c in checks):
        raise ValueError("Published aggregate reproduction check failed; inspect before publishing.")
    source = {"repository": REPO, "commit": COMMIT, "paper_doi": "10.1109/TMI.2022.3148728",
              "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9247015/", "data_doi": ARCHIVES["test_data.zip"]["doi"],
              "model_doi": ARCHIVES["pretrained_models.zip"]["doi"], "data_license": "CC BY 4.0", "code_license": "Apache-2.0"}
    data = {"schema_version": "1.0", "source": source, "shape": [96,96], "sample_ids": SAMPLE_IDS,
            "budgets": BUDGETS, "bands": BANDS, "rows": rows, "summary": summary,
            "reproduction_checks": checks,
            "definitions": {
                "units": "Native power Doppler arbitrary units; not calibrated CBV.",
                "band_assignment": "10*log10(reference/max(reference)); all lower bounds inclusive, upper bounds exclusive except the final0dBmaximum.",
                "signed_bias_percent": "100*sum(prediction-reference)/sum(reference) within band; sign can cancel errors.",
                "absolute_error_percent_total_reference": "100*sum(abs(prediction-reference) within band)/sum(reference over whole image).",
                "normalized_absolute_error_band_percent": "100*sum(abs(prediction-reference) within band)/sum(reference within band).",
                "whole_bias": "100*sum(prediction-reference)/sum(reference) over all pixels.",
                "nmse": "sum((prediction-reference)^2)/sum(reference^2), on unclipped native values.",
                "repository_ssim": "TensorFlow image.ssim,filter_size=3,max_val=1,on nativefloat32images; exactly follows released code despite referencevaluesabove1.",
                "review_tolerance": "User-set descriptive cutoff, not validated quality, safety, acquisition or clinical criterion.",
                "model_budget_confound": "Each frame budget uses a separately trained released model; differences combine sampling and model-weight effects, not a causal estimate of framecount alone.",
                "nulls": "Empty reference bands have null signed/normalized errors, zero pixels/power/absolute error. Never replace null with zero bias.",
                "sample_unit": "Matched image pair, not independent animal, neural state, brain region or functional timepoint."
            }}
    (out / "analysis.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    (out / "data.js").write_text("// Generated by analysis.py; metrics use unrounded native arrays.\nwindow.DIIANNI_DATA = "+json.dumps(data,separators=(",", ":"), allow_nan=False)+";\n")
    sample_dir = out / "samples"; sample_dir.mkdir(exist_ok=True)
    for idx, sid in enumerate(SAMPLE_IDS):
        payload = {"id": sid, "shape": [96,96], "reference": rounded_flat(references[idx]),
                   "predictions": {str(k): rounded_flat(predictions[j,idx]) for j,k in enumerate(BUDGETS)}}
        (sample_dir / f"{sid}.json").write_text(json.dumps(payload,separators=(",", ":"),allow_nan=False)+"\n")
    export_csv(rows, out / "metrics.csv")
    manifest = {"source": source, "archives": ARCHIVES, "model_sha256": MODEL_HASHES, "sample_sha256": input_hashes,
                "upstream_file_sha256": UPSTREAM_FILES,
                "versions": {p: importlib.metadata.version(p) for p in ["numpy","scipy","tensorflow","keras","h5py","matplotlib"]},
                "xmean": XMEAN, "xstd": XSTD, "display_precision_significant_digits": 7,
                "trained_here": False, "independent_validation": False, "negative_predictions_clipped_for_metrics": False}
    (out / "PROVENANCE.json").write_text(json.dumps(manifest,indent=2)+"\n")
    np.savez_compressed(cache / "reanalysis-native-arrays.npz", references=references,predictions=predictions,ids=SAMPLE_IDS,real_frames=BUDGETS)
    if not args.no_figures:
        figures(data,references,predictions,out / "figures")
    print(f"Exported {len(rows)} rows, 1000 band records, and40compact samplefiles to {out}", flush=True)


if __name__ == "__main__":
    main()
