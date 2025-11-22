# Practical Uncertainty Upgrade: MC Dropout at Inference

This repository currently uses a fixed Gaussian residual noise vector saved to `uncertainty_<model>.npy`. That approach yields a single, input-agnostic uncertainty estimate. To obtain input-dependent (epistemic) uncertainty with minimal code changes, enable **Monte Carlo (MC) dropout at inference**.

## Why MC dropout helps
- **Epistemic coverage:** Running the model with dropout *on* during inference yields slightly different subnetworks each pass. The spread of their predictions reflects how confident the model is around the given feature vector.
- **Minimal change:** You reuse the existing dropout layers—no need for extra heads or retraining. Only the prediction loop changes.

### Intuition: why zeroing neurons reveals uncertainty
- **One model, many plausible subnetworks:** Dropout randomly removes neurons, creating a *family* of lighter networks that all share the same learned weights. Each mask is a hypothesis for how the model might behave if some features or pathways were missing.
- **Agreement vs. disagreement:** If the training data strongly supports a pattern at your current input, most subnetworks will produce nearly the same output; their variance will be low. If the model is unsure (little signal, or input far from the training distribution), different subnetworks will disagree more, inflating the variance. That spread is the uncertainty estimate.
- **Connection to data scarcity:** Zeroing neurons mimics having less evidence. Where the original training never pinned down a unique solution, these masked subnetworks drift apart, revealing the model's epistemic doubt.

## What “Monte Carlo dropout” actually means
- **Dropout as a random subnetwork sampler:** Each dropout mask randomly zeros a subset of neurons, so every forward pass is a different, thinner network. This randomness is the “dropout” part.
- **Monte Carlo = many random trials:** You repeat the forward pass many times with different dropout masks (samples). Collecting these samples is the “Monte Carlo” part. By averaging and measuring their spread, you estimate both the typical prediction and how much it fluctuates for the given input.
- **Approximate Bayesian view (intuition only):** Treating each dropout mask as one plausible model approximates drawing models from a posterior distribution. The variance across samples is then a practical proxy for epistemic uncertainty—how unsure the model is because of limited data or unfamiliar inputs.

## How to implement (step by step)
1. **Switch model to train mode for sampling:** After loading the trained model but before prediction, call `model.train()` instead of `model.eval()`. This keeps dropout active while leaving BatchNorm statistics fixed if you avoid further training steps.
2. **Disable gradient tracking:** Wrap the sampling loop in `with torch.no_grad():` to avoid autograd overhead.
3. **Draw multiple samples:** For each input batch, run `K` stochastic forward passes (e.g., `K=50`). Stack the outputs to shape `(K, batch, horizons)`.
4. **Aggregate statistics:**
   - Undo the target normalization before aggregating so outputs are in the original return scale.
   - Mean prediction: `mean_preds = samples.mean(dim=0)`
   - Predictive std: `std_preds = samples.std(dim=0, unbiased=False)`
5. **Use stats downstream:** If you rank with a gamma-mode, shift the predicted returns into a non-negative multiplier by adding `1` before computing `mode = (mean+1) - var/(mean+1)`. Use `std_preds` as the uncertainty term instead of the static `uncertainty_<model>.npy` values.

## Tips to keep results stable
- Keep dropout rates as trained; do **not** retrain in train mode during sampling.
- Fix a random seed (`torch.manual_seed`) to make experiments reproducible.
- Start with `K=50` samples; increase if the std looks noisy.
- If BatchNorm instability appears, replace BatchNorm with LayerNorm in future training runs, or cache running stats before sampling.

## Expected outcome
Compared to the current single noise vector, MC dropout produces larger uncertainty for out-of-distribution or weak-signal inputs while keeping low uncertainty where the model is confident. This should give more realistic spreads for your gamma-based ranking without major refactoring.
