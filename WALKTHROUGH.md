# Understanding this project

## 1. What biological question sits behind it?

Functional ultrasound follows changes in blood-related ultrasound signals as an indirect window onto brain activity. If we want to sense brain states while also delivering ultrasound stimulation, collecting less data could make the measurement easier to fit into the experiment. But a reconstructed image that looks good overall may still behave differently in the particular signal range we want to measure.

The modest question here is: **which existing image pairs deserve a closer check before we change the acquisition budget?** We cannot yet answer whether a neural state or stimulation response is measured correctly.

## 2. What do the files represent?

There are 40 pairs. Each has a stack of 250 real beamformed ultrasound frames and one conventional power-Doppler reference image. Five released neural networks take the first 125, 100, 75, 50 or 25 real frames and predict a Doppler image. We did not collect these data or train these networks.

Think of the reference as the original processing result we can compare against—not a direct measurement of every neuron's activity. The filenames do not tell us which animal or session they belong to.

## 3. What did the analysis do?

First, it checked the download hashes and used the published input scaling. It ran every model on every pair and kept the source IDs. The average similarity and squared-error results reproduce the published table.

Next, it divides each reference image into intensity bands. A band is simply a set of pixels with comparable reference signal strength; it is not a named brain region. For the same pixels it compares reconstructed and reference power, while also showing how much of the total signal that band represents.

This extra context matters. A 50% error in a very small signal can be less consequential in absolute terms than a 10% error in a much larger signal. Conversely, a near-zero whole-image signed bias can conceal local overestimation and underestimation that cancel.

## 4. How should I use the interaction?

Start with `fr11`, 50 real frames and the −30 to −20 dB band. Compare the whole-image signed bias with the band's signed bias. Then check the band's power share and absolute-error contribution. Change the frame budget rather than assuming more frames always improve every local diagnostic. Each setting also uses a different, separately trained model, so we cannot say frame count alone caused a difference.

Change the review tolerance and see which pairs remain flagged. This is a way to choose what to inspect first—not an automatic good/bad verdict. Look at both discrepant and concordant cases, because a useful next experiment needs controls.

## 5. What is the next sensible experiment?

Collect a new paired full/sparse time series with animal and session IDs, stimulus timing, motion and physiological records. Include stable baseline and a controlled stimulation condition. Ask whether a reconstruction offset stays constant or changes with the state we are trying to measure. A constant offset may have different implications from a response-dependent distortion. Only these new matched measurements could test temporal or functional fidelity.

## 6. What can I accurately say in an email or interview?

> I re-ran your released Deep-fUS models on the 40 public test pairs and reproduced the aggregate reconstruction metrics. I added a small review tool that compares whole-image agreement with signal-band bias, absolute error and signal mass, so discrepant cases can be prioritized for a paired-reference check. It is a reanalysis of existing evidence, not a validation of neural-state sensing.

If asked about implementation, explain that the project was built with AI assistance and that the lab supplied the original data, models and methods. The project does not establish that you have independently performed ultrasound acquisition or trained a reconstruction model. The strongest contribution is the question, transparent comparison and practical follow-up—not an unsupported claim of new experimental expertise.
