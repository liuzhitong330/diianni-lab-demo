# Verification

Checked on 3 November 2026. This is software and numerical verification of reused public evidence, not independent biological validation.

- The full released-model pipeline ran on all 40 source pairs at all five budgets. Eleven tests cover boundaries, exhaustive bands, null ratios, unclipped negative predictions, formula examples, sample coverage, exports and published aggregate reproduction.
- A separate calculation, without importing the production metric functions, checked all 200 reconstructions against the native arrays. NMSE differences were below 5e-17 and SSIM differences below 2e-7. Band statistics matched; all reference images matched their source MAT files.
- All 1,000 long-form CSV records matched the analysis JSON. Rounded browser arrays preserved the band assignments and negative-pixel counts for these data.
- Browser checks exercised the five budgets, five bands, sample suggestions, selected-band mask, exact pixel values, keyboard-operated tolerance, empty-shortlist case, table-to-image link and CSV download. The downloaded 200-row review CSV matched the full-precision data; default flag counts by 125/100/75/50/25 frames were 0/0/0/3/2.
- Desktop layout was compared with the Ramani reference. Narrow 390px and 320px iframe previews were inspected. The fixed editorial-shell validator passed. No browser console warnings/errors were observed during local interaction checks.

The selected opening example is illustrative, not representative of every pair. The page never labels an unflagged image a validated pass. Different frame budgets also use different pretrained models; observed differences do not isolate the causal effect of adding frames.
