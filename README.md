# Sparse sensing, matched review

A small, reproducible reanalysis of the Di Ianni Lab's released **Deep-fUS** models and 40 matched test pairs. It asks where whole-image reconstruction agreement can conceal signal-dependent error, and makes those cases easy to inspect before designing a new paired acquisition.

This is an AI-assisted application project by Cathy Liu, not a lab-endorsed tool. It reproduces existing test evidence and adds a descriptive review workflow; it is **not independent validation, a neural-state decoder, or an acquisition safety test**.

## What is useful here?

- Re-runs all five published models on the same 40 source IDs: 200 reconstructions, with no retraining or cherry-picked pairs.
- Reproduces the published mean SSIM and NMSE, then contrasts global agreement with reference-defined intensity-band bias, absolute error, and signal mass.
- Lets a reviewer change a descriptive tolerance, inspect contributing pixels, and export a matched review shortlist. No threshold is a validated pass/fail standard.
- Preserves provenance and a reusable paired-image QC schema for a future dataset.

## Results in brief

| Real frames | Beamformed-data reduction | Mean repository SSIM | Mean NMSE |
|---:|---:|---:|---:|
| 125 | 75% | 0.9166 | 0.0399 |
| 100 | 80% | 0.9041 | 0.0448 |
| 75 | 85% | 0.8915 | 0.0506 |
| 50 | 90% | 0.8619 | 0.0701 |
| 25 | 95% | 0.8154 | 0.1106 |

These means reproduce Table II of [Di Ianni & Airan, 2022](https://doi.org/10.1109/TMI.2022.3148728) at four decimal places. The same test evidence is reused, not independently collected.

A useful example is **fr11 with 50 real frames**: whole-image signed bias is about **+0.12%**, while the reference-defined −30 to −20 dB band has about **+27.29%** signed bias. That band contains **7.90%** of the reference power; its absolute error contributes **2.70%** of total reference power. Whole-image signed errors can cancel, so this example is not evidence that the global reconstruction is perfect or that a neural response is wrong. It is a reason to inspect the pair and test temporal fidelity in a new experiment.

The lowest-intensity band can show much larger relative errors while carrying very little reference power. The explorer therefore shows absolute error and power share together, rather than using a large percentage as a biological claim. Band error is not uniformly monotonic with frame budget. **Each frame budget uses a separately trained released model.** Differences therefore reflect the input budget and the model weights together, not a controlled causal effect of frame count alone.

## Reproduce

Tested on Python 3.12.14, macOS arm64 CPU. The installed TensorFlow model-loading path requires NumPy 1.x; use the pinned environment rather than upgrading NumPy independently.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python analysis.py --cache .cache --output .
.venv/bin/python -m unittest -v test_analysis.py
.venv/bin/python -m http.server 8000
```

Open `http://localhost:8000`. The script downloads approximately 835 MB of original archives and checks SHA-256 before extraction. Allow several GB for the environment and private cache. The 200 CPU reconstructions take a few minutes with the script's single-thread settings; download time varies. `--no-figures` skips the optional static SVG figures. No GPU or model training is needed.

The original upstream source is pinned to commit `080ed97a521bbebae5bf71b3cc85a2a11c5ba682`; the pipeline also verifies the inspected source files. Never commit `.cache`, raw `.mat` files, `.h5` models, or the virtual environment.

## Files

- `analysis.py`: integrity checks, source-ID-preserving inference, diagnostic calculations, and exports.
- `analysis.json` / `data.js`: full-precision metric records and definitions used by the page.
- `samples/frN.json`: native-unit reference and predictions, flattened row-major and rounded to seven significant digits **only for display**.
- `metrics.csv`: 1,000 sample × model × band records; global metrics repeat across the five bands. Do not treat those repetitions as extra samples.
- `PROVENANCE.json` / `PROVENANCE.md`: archive/model/source hashes, transformations, units, and limitations.
- `test_analysis.py`: partition, boundary, null, negative-value, formula, coverage, export, and reproduction checks.
- `WALKTHROUGH.md`: plain-language explanation and defensible outreach claims.
- `figures/`: two reproducible SVG summaries of aggregate reproduction and signal-band bias versus power share.

## Boundaries and next experiment

The released files lack animal/session mapping, stimulus timing, physiological annotations, anatomical masks, and functional time series. Forty pairs are not 40 independent animals. A static reconstruction error cannot establish a false neural state, altered activation, or neuromodulation efficacy. Reference-defined intensity bands are not brain regions or validated vessels. The reference is conventional processing, not biological ground truth. No conventional sparse-processing comparison is recreated because the public inputs do not include the required complex data and complete matched processing outputs.

**Follow-up:** acquire paired full/sparse time series with matched animal/session IDs, stimulus logs, registration, motion and physiological records. Test concordant and discrepant cases under stable baseline and a controlled stimulation condition. Does a reconstruction bias remain constant, or change with physiological state? This distinguishes a static offset from a distortion of the response the lab intends to measure.

## Sources and reuse

The [official lab resources](https://diiannilab.org/media-resources/) link the [upstream repository](https://github.com/todiian/deep-fus), [test data v1](https://doi.org/10.6084/m9.figshare.23663970.v1), and [models v2](https://doi.org/10.6084/m9.figshare.23663979.v2). Code is Apache-2.0. Derived data retain **CC BY 4.0** attribution to Tommaso Di Ianni; see `NOTICE`. The paper's figures are not redistributed.
