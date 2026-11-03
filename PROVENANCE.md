# Provenance and analysis decisions

## Verified sources

The [Di Ianni Lab resource page](https://diiannilab.org/media-resources/) directly links the target-owned [Deep-fUS repository](https://github.com/todiian/deep-fus). This analysis pins commit `080ed97a521bbebae5bf71b3cc85a2a11c5ba682` (2023-07-11). The publication is [Di Ianni & Airan, IEEE TMI 2022, DOI 10.1109/TMI.2022.3148728](https://doi.org/10.1109/TMI.2022.3148728).

| Input | Version | Size | SHA-256 |
|---|---|---:|---|
| [Test data](https://doi.org/10.6084/m9.figshare.23663970.v1) | 1 | 426,605,576 bytes | `28b0cba98af87110b5049222da51c33de1973ea1b3ecedea3ac8ea7b60877ed8` |
| [Pretrained models](https://doi.org/10.6084/m9.figshare.23663979.v2) | 2 | 408,685,445 bytes | `918919f517b5ee09b07ed9b8461ac19aae36e5ccead6439ec27187fc24ede287` |

Both archival records identify Tommaso Di Ianni and CC BY 4.0. `PROVENANCE.json` includes individual source-file, sample-file and model hashes. The upstream software uses Apache-2.0; its license and attribution are retained.

## What each input is

Each `fr1.mat` through `fr40.mat` has exactly two numeric fields: `x`, a real-valued 96 × 96 × 250 beamformed compound sequence, and `y`, a 96 × 96 conventional full-sequence power-Doppler reference. The public data are not raw channel RF and do not include the imaginary component. The 250 frames are the short within-acquisition input for one Doppler image, not 250 functional-state observations.

Native `y` values are positive and not normalized to [0,1]; the reference maxima range from about 6.73 to 23.66. Units are arbitrary power Doppler, not calibrated blood volume. There is no pixel-spacing or anatomical annotation in the files. The paper describes 40 test pairs from four acquisitions, but the deposit does not map filenames to acquisitions or animals. No biological replicate count is inferred.

The model folders are named `ResUNet5_125`, `ResUNet5_100`, `ResUNet5_75`, `ResUNet5_50`, and `ResUNet5_25`. Their H5 configurations identify the 3D residual U-Net architecture (`ResUNet53D`), despite the folder names. No precomputed prediction images were included in the archives.

## What ran

All five published H5 models were loaded using TensorFlow/Keras with `compile=False` and applied at inference time without training. This pipeline mirrors the inspected upstream `predict.py` / `utils.py` preprocessing and metric definitions; it does not claim the original `predict.py` ran end-to-end unchanged. The original model path differs from the archive's folder name, and its loader permutes sample order. Here each source filename remains explicit throughout.

For each budget `k`, take `x[:, :, :k]` and standardize with the published training constants:

```text
x_standardized = (x - (-0.5237595494149918)) / 131526.6016974602
```

Predictions and reference values remain in native units for all metrics. No negative prediction is silently zeroed, no image is independently rescaled to improve agreement, and no case is excluded. All 200 predictions are finite; negative prediction counts are reported by case and budget.

Compression is `100 × (1 − k/500)` for full spatial retention: the denominator includes 250 complex frames with real and imaginary parts. Thus 125 real frames mean 75% reduction in beamformed sample volume. These percentages do not establish equivalent scanner latency, imaging frame rate, exposure reduction, or prospective performance. Each budget is paired with its own separately trained released model, so between-budget differences combine sampling and model-weight effects. This is not an experiment isolating frame count as a causal variable.

## Metrics and bands

- **Repository SSIM:** `tf.image.ssim(reference, prediction, max_val=1, filter_size=3)` on native float32 images. This deliberately preserves the released code's fixed range, even though some native values exceed 1. It is a reproduction metric, not a universally calibrated acceptance score.
- **NMSE:** `Σ(prediction − reference)² / Σ(reference²)` on unrounded, unclipped arrays.
- **Signed bias:** `100 × Σ(prediction − reference) / Σ(reference)`, either over a band or the whole image. Positive and negative errors can cancel.
- **Band-normalized absolute error:** `100 × Σ|prediction − reference| / Σ(reference)` within the band. This does not cancel signed error, but can be large for a weak denominator.
- **Absolute-error contribution:** `100 × Σ|prediction − reference| within band / Σ(reference) over whole image`.
- **Reference-power share:** `100 × Σ(reference within band) / Σ(reference over whole image)`.

Bands use `10 log10(reference / max(reference))` and partition all pixels into `<−40`, `[−40,−30)`, `[−30,−20)`, `[−20,−10)`, and `[−10,0]` dB. They are not anatomical regions or vessel segmentations. Empty bands have null ratios, not a false zero error. Bands are fixed from the reference, so the prediction cannot redefine its own comparison set.

The page's tolerance slider is exploratory and chosen for inspection, not trained or clinically validated. Showing alternative tolerances is a sensitivity check, not an estimate of statistical uncertainty. Pair means and SDs are descriptive; no confidence intervals or hypothesis tests are offered without the missing grouping metadata.

## Reproduction versus new work

The aggregate SSIM and NMSE reproduce the same released test evidence and Table II at four decimal places. The added work is the explicit matched-ID audit, signal-band sensitivity analysis, signal-mass context and reusable review export. It is not a new model, independent cohort, prospective trial, demonstrated neural-state measurement, or discovery of a biological mechanism.

The numerical arrays used in `samples/` are rounded to seven significant digits for browser display; `analysis.json` and `metrics.csv` were calculated from the unrounded native arrays. Recompute exact metrics using the archival inputs, not the rounded browser derivative.

## What is missing

There are no stimuli, functional time series, animal/session IDs, physiological labels, movement annotations, registration landmarks, or region masks in this deposit. The reference is an existing reconstruction, not direct neuronal truth. Different pairs must not be connected into an invented longitudinal trace. The data cannot determine whether an error changes neural-state estimates or closed-loop treatment decisions.

The current lab programme motivates the question, but no private lab bottleneck or planned experiment is assumed. This package demonstrates AI-assisted analysis and experimental reasoning, not independent hands-on ultrasound acquisition or validation by the laboratory.
