# Evidence Representation Study

How does evidence geometry affect learning when the encoded scalar information
is held fixed? This exploratory synthetic study compares centered, log and
arcsine representations of the same estimated binary-class evidence across
26 conditions. It includes reproducible PyTorch encoders, logistic regression,
a one-hidden-layer MLP, a decision-tree control, and recorded experimental
findings. The goal is to understand when geometry matters, not to establish a
universal representation winner.

Every evidence transform shares the same fitted equal-prior probability q;
only its geometry changes. See [Methods](#methods) for the estimator and fitting
protocol, or [Reproduce](#reproduce) to run the study.

Explore the experiments and plots in [the study notebook](evidence_experiment.ipynb).

## Methods

### Shared Evidence

For a shared equal-prior probability $q$, the encodings are:

| Name | Formula | Range |
|---|---|---|
| `centered` | $2q-1$ | $(-1,1)$ |
| `log` | $\log(q)-\log(1-q)$ | Real numbers |
| `arcsine` | $\frac{2}{\pi}\arcsin(2q-1)$ | $(-1,1)$ |

These transforms are strictly increasing and invertible in exact arithmetic.
No representation-specific clipping or shrinkage is applied. Inputs are finite
continuous features and binary labels; each fitted fold must contain both classes.

For each feature partition, let $n_{cb}$ be the count in class $c$, bin $b$;
$N_c$ the class count; and $B$ the partition's number of bins. With `alpha > 0`:

$$
m_{cb}=\frac{n_{cb}+\alpha}{N_c+B\alpha},\qquad
q_b=\frac{m_{1b}}{m_{0b}+m_{1b}}.
$$

`Log` is positive-oriented weight of evidence for the class-conditional masses,
not empirical-prevalence posterior log-odds. Equal-prior q is not a calibrated
deployment probability. Uniform smoothing can still bias sparse or imbalanced
estimates. A low-class-support bin has fewer than five observations in either
class. Extreme q means q < 0.05 or q > 0.95; extreme test rows have such a value
in any generating signal feature, not necessarily an oracle rare observation.

[encoders.py](encoders.py) supports percentile-width (`fixed`), historical
raw-width (`raw_fixed`), shifted dual-percentile (`dual_fixed`) and merged
quantile (`dual_adaptive`) partitions. Dual modes average q before transforming.
The fitted encoders are nontrainable PyTorch modules supporting CPU/CUDA and
float32/float64. Univariate encoding can lose information carried only by interactions.

### Binning Evidence

The recorded suite tests `fixed` and `raw_fixed` only. `fixed` uses equal widths
in fitted percentile coordinates, giving approximately equal total training
mass per bin, not equal raw widths or equal class counts. `raw_fixed` uses equal
raw widths over the historical min/max range, so bin masses can be very uneven.
Neither guarantees adequate support within each class. The dual modes are
implemented but untested here; no adaptive minimum-mass threshold was varied.

- **Width and placement:** Cases 6/11 and 9/12 compare percentile and raw-width
  partitions on matched clean and contaminated data. Contamination collapses
  raw-width occupancy, but the weak predictive baseline prevents a general
  accuracy claim for either mode.
- **Resolution and support:** Cases 22/23 increase percentile bins from 20 to
  120 on identical data. Mean centered logistic loss rises from 0.667033 to
  0.685695 while raw predictions stay fixed. Case 26 restores support with more
  data and largely removes geometry differences, yet still trails the coarser
  reference. More bins do not necessarily yield better evidence.
- **Smoothing:** Cases 13/14 change different settings relative to case 9;
  only case 14 isolates alpha (1 to 0.1), with little effect in coarse bins.
  Cases 23/24 isolate alpha from 1 to 0.01 in sparse 120-bin partitions:
  centered mean loss improves from 0.685695 to 0.683909, whereas log worsens
  from 0.685548 to 0.689088. Smoothing changes the representation comparison,
  not just overall estimation quality in a common direction.

These contrasts establish sensitivity to partitioning, support and smoothing,
not an optimal bin width, target mass or alpha. Total bin mass alone is
insufficient: class-specific support and out-of-sample evidence reliability
also matter.

### Fitting and Controls

- **IID data:** Seeded stratified 60/20/20 splits and four-fold out-of-fold
  training encoding. Both partitions and class counts are fitted inside each
  fold; external validation/test encoding uses the full training fit.
- **Forward data:** Chronological splits, encoder-only warm-up, horizon-length
  boundary purges and block fits restricted to
  `row + forecast_horizon < block_start`. Frozen history stays fixed; causal
  rolling/expanding histories admit external labels only after maturation.
  Model weights and training-derived scaling stay fixed during validation/test.
- **Learners:** Logistic regression and a one-hidden-layer MLP use AdamW/BCE,
  paired fresh initialization and batch order, and validation-loss checkpoint
  selection. CART uses a fixed depth/leaf budget. Test labels never select models.
- **Scaling:** `standard` uses per-representation training means and standard
  deviations; `none` retains native scales. Both reuse the same probabilities,
  splits and model hyperparameters. Standardization does not remove all
  regularization or optimization differences.
- **Controls:** The targeted family adds matched raw-feature learners. Validation
  signal checks require standardized raw logistic AUC >= 0.58 and prior-relative
  log-loss gain >= 0.005; extreme-cell checks require at least 2% of informative
  validation cells. Failed checks are retained, not used to filter experiments.

The weak-factor generators use noisy, correlated observations of shared Gaussian
factors; outcomes also depend on an unobserved driver and Bernoulli sampling.
Independent Pareto contamination changes observations, not labels. The targeted
family has 12 features (eight informative), two factors, signal/observation-noise
strengths of 1.5 and hidden-driver strength 1. It is distinct from the original
weak-factor family. Same-size, same-signal estimator contrasts share data/splits;
the null changes labels and therefore stratified split membership.

Forward cases use persistent AR(1) factors and overlapping future-outcome
windows. Covariate interventions change observed location/scale; concept
interventions reverse the factor/outcome relationship between change points.
Covariate controls preserve labels and concept controls preserve observations.
Change points are not model inputs. Full generation and protocol definitions
are in [experiment.py](experiment.py) and [temporal.py](temporal.py).

## Reproduce

From this study directory, install [requirements.txt](requirements.txt) and run
the implementation tests:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Open [evidence_experiment.ipynb](evidence_experiment.ipynb) locally or in Colab.
It runs the cases, plots their data/evidence geometry and learning curves, and
exports a JSON bundle plus a summary CSV. The default suite fits 792 models:
360 original-case fits and 432 targeted fits. `SEEDS`, `TARGET_SEEDS` and
`SCALINGS` control repetition; both native and standardized scaling are included.
CUDA is optional. For a single experiment outside the notebook:

```python
import torch
from encoders import EncoderConfig
from experiment import ExperimentConfig, prepare_dataset, run_experiment

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = ExperimentConfig(n_samples=3000, scenario="additive", scaling="standard")
data = prepare_dataset(config, EncoderConfig(num_bins=20), DEVICE)
results = run_experiment(data)
```

`include_raw_baseline=True` adds three raw-feature fits to the default nine
evidence fits per condition/seed/scaling. Saved exports include configurations,
versions, labels/predictions, histories, log loss/Brier/AUC/accuracy, support and
q diagnostics, prior baselines, and forward-history audits. Plots and fitted
models are not exported. The recorded run used Python 3.13.15, PyTorch
2.11.0+cu130 and scikit-learn 1.6.1; CPU/GPU reruns need not be bitwise identical.

## Case Findings

These findings use the [saved summary](results/20261016T152546370593Z/summary.csv)
and [full results](results/20261016T152546370593Z/results.json): 26 conditions,
792 fits and two scaling modes. Cases 1-20 use seed 42; cases 21-26 report means
over seeds 42, 43 and 44. Unless stated otherwise, losses below are standardized
logistic test log losses; lower is better. `Raw` means unencoded features, not
raw-width evidence bins. The prior baseline predicts a constant historical class
frequency. Comparisons are within a condition, not a ranking of task difficulty.

### 1. Moderate Additive Signal (`moderate_additive`)

**Finding:** Centered and log losses are 0.533879 and 0.533153, both well below
the 0.693098 prior baseline. MLP representation differences are also small.
**Conclusion:** Useful signal does not necessarily produce a meaningful geometry
gap; a tiny representation advantage should not dominate model selection.

### 2. Separated Classes (`separated_classes`)

**Finding:** Log improves loss from 0.032556 to 0.023704 versus centered;
the MLP shows the same ordering. All neural fits perform strongly.
**Conclusion:** Log geometry can make strong, informative evidence more useful
to both learners. Boundedness is not automatically beneficial.

### 3. Informative Rare Regions (`rare_regions`)

**Finding:** Log improves loss from 0.608371 to 0.585038 versus centered;
MLP loss also improves from 0.599389 to 0.582655.
**Conclusion:** Rare observations can carry useful evidence. Their rarity alone
is not a reason to suppress their influence.

### 4. Class-Dependent Variance (`variance_tails`)

**Finding:** Log improves loss from 0.251985 to 0.239042 versus centered;
the MLP again benefits. Class information lies in spread rather than a mean shift.
**Conclusion:** Geometry matters beyond simple location separation; log remains
effective when tail structure is genuinely informative.

### 5. Sparse Imbalanced Mixture (`sparse_imbalanced`)

**Finding:** Even the best neural representation loss exceeds the prior baseline:
logistic log loss is 0.416158 versus prior loss 0.404975. There are 1,194
low-class-support bins out of 1,200.
**Conclusion:** A relative representation win need not be a useful predictive
result. Support and prior baselines take precedence over ranking transforms.

### 6. Weak Correlated Indicators (`weak_factors`)

**Finding:** Centered loss is 0.695283 versus prior loss 0.693036. Standardized
neural fits do not beat the prior; centered/log feature correlation averages 0.999988.
**Conclusion:** The fitted evidence pipeline does not recover useful test gains
here. Almost-linear transformed inputs leave little geometry difference to study.

### 7. Reduced Hidden Variation (`weak_less_hidden`)

**Finding:** Reducing hidden-driver strength from 3 to 0.5 leaves centered loss
at 0.702516, above prior loss 0.693035.
**Conclusion:** Less hidden noise did not translate into better encoded prediction
in this realization. This is not proof that the observed features lack signal.

### 8. Weak-Factor Null (`weak_null`)

**Finding:** Removing observed-factor signal leaves all standardized neural
representation losses above the prior; even logistic log loss is 0.693265
versus prior loss 0.693119.
**Conclusion:** Small metric differences can arise without population signal.
The null is a guard against interpreting every representation win as learning.

### 9. Independent Rare Contamination (`weak_contaminated`)

**Finding:** Adding label-independent Pareto shocks leaves logistic log loss at
0.695566 versus prior loss 0.693036. Standardized geometry differences stay tiny.
**Conclusion:** Contaminated observations alone do not establish a robustness
advantage when the encoded task is already near chance.

### 10. Larger Contamination Magnitudes (`weak_larger_tails`)

**Finding:** Increasing shock scale from 20 to 100 gives exactly identical
predictions and training histories under both scaling modes.
**Conclusion:** Once percentile-bin membership is unchanged, larger raw magnitudes
need not alter evidence. Tail magnitude is not the same as evidence extremeness.

### 11. Raw-Width Bins Without Contamination (`weak_raw_clean`)

**Finding:** Raw-width partitions have 166 low-class-support bins out of 640;
centered loss is 0.698906, worse than the percentile counterpart's 0.695283.
**Conclusion:** Partition construction already affects estimation before any
representation transform is applied, even without added contamination.

### 12. Raw-Width Bins With Contamination (`weak_raw_contaminated`)

**Finding:** There are 305 empty bins out of 640, and feature 0 puts 85.2% of
training observations in one bin. Yet centered loss improves over the clean
raw-width case to 0.697157, while still losing to the prior.
**Conclusion:** Bin collapse is real, but predictive damage cannot be inferred
from the partition diagnostic alone in this weak-signal comparison.

### 13. Finer Percentile Bins (`weak_fine_bins`)

**Finding:** Increasing contaminated-data bins from 20 to 80 worsens centered
logistic loss from 0.695574 to 0.698497, but improves centered MLP loss from
0.696096 to 0.694231. Both remain above the prior.
**Conclusion:** Resolution interacts with learner and estimation noise; there
is no learner-independent gain from using finer bins.

### 14. Lower Smoothing With Coarse Bins (`weak_low_smoothing`)

**Finding:** Reducing alpha from 1 to 0.1 with 20 bins changes centered loss only
from 0.695574 to 0.695524 and produces no extreme informative-feature test evidence.
**Conclusion:** Lower smoothing does not necessarily create extreme estimates.
That intended mechanism must be checked rather than assumed.

### 15. Stable Forward Prediction (`time_stable`)

**Finding:** With causal training and frozen external encoding, centered loss
is 0.697425 versus prior loss 0.693661; the MLP also loses to the prior.
**Conclusion:** The historical evidence pipeline is already weak without drift.
This baseline limits what the shift experiments can demonstrate.

### 16. Covariate Shift (`time_covariate`)

**Finding:** Mean test CDF KS distance reaches 0.566 and about 50% of feature
observations lie outside historical ranges. Centered loss is still near chance
at 0.696255, versus prior loss 0.693661.
**Conclusion:** A large marginal distribution change is not by itself evidence
of predictive deterioration or a particular representation advantage.

### 17. Concept Shift (`time_concept`)

**Finding:** Reversing the factor/outcome relationship while preserving observed
features raises centered loss to 0.703358 from 0.697425 in the stable control.
Representation differences are negligible.
**Conclusion:** Bounded geometry does not repair an outdated predictive
relationship; the historical evidence itself can be wrong.

### 18. Combined Shift With Frozen History (`shift_frozen`)

**Finding:** Under combined covariate/concept shift and contamination, centered
MLP loss is 0.696872 versus prior loss 0.692580. Transform differences are small.
**Conclusion:** Changing geometry is not enough to make this frozen historical
estimate useful; history policy is a separate part of the problem.

### 19. Combined Shift With Rolling Updates (`shift_causal_rolling`)

**Finding:** Causal rolling updates improve centered MLP loss from the frozen
case's 0.696872 to 0.690068, below prior loss 0.692580. Logistic loss improves
slightly but remains worse than the prior.
**Conclusion:** Updating matured evidence helps the MLP here more than changing
representation does, but the gain is learner-dependent and comes from one path.

### 20. Combined Shift With Expanding Updates (`shift_causal_expanding`)

**Finding:** Standardized centered MLP loss is 0.690185, close to rolling updates.
Without scaling, arcsine improves logistic loss from 0.695083 to 0.693318, while
log improves MLP loss from 0.692395 to 0.691831. Logistic epochs also differ.
**Conclusion:** A representation ranking can reverse between learners and change
with scaling. Neither expanding history nor a transform is a general winner.

### 21. Learnable Weak-Signal Reference (`target_weak_reference`)

**Finding:** With 12,000 rows and 20 bins, raw logistic passes the validation
signal check in every seed. Mean raw test loss is 0.652520 versus prior loss
0.693121; centered/log losses are 0.656933/0.656964.
**Conclusion:** Observed signal is learnable, but geometry gaps are tiny under
this well-supported estimator. Raw inputs retain slightly better performance.

### 22. Small-Sample Weak Signal (`target_weak_small`)

**Finding:** At 1,200 rows, mean centered loss is 0.667033 versus matched raw
loss 0.639291 and prior loss 0.692669. All raw validation signal checks pass.
**Conclusion:** The observed signal survives, but the supervised evidence
pipeline loses predictive performance before representation choice becomes decisive.

### 23. Fine Bins With Small Samples (`target_weak_fine`)

**Finding:** Increasing bins from 20 to 120 raises mean centered loss from
0.667033 to 0.685695. Matched raw predictions remain unchanged at loss 0.639291.
**Conclusion:** Estimator resolution causes a much larger performance change
than the differences between transforms in this condition.

### 24. Unreliable Extreme Estimates (`target_weak_extreme`)

**Finding:** With 120 bins and alpha=0.01, extreme validation q values have only
46.85-51.85% direction accuracy. Log raises mean logistic loss from 0.683909
to 0.689088, losing to centered in all three seeds; about 95% of the aggregate
penalty comes from extreme-q test rows. Arcsine's lower mean is not a seed-wise
win: it beats centered in only one seed. MLP log wins on average but loses in two seeds.
**Conclusion:** Unreliable extremes can make geometry consequential, but the
effect depends on the learner. The MLP mean conceals a sign-changing response.

### 25. Sparse Extreme Estimates Under the Null (`target_weak_null`)

**Finding:** Extreme estimates remain despite removing observed-factor signal.
Raw validation learnability checks fail in every seed. Mean standardized neural
representation losses exceed prior loss 0.693127; logistic log loss is 0.701282.
**Conclusion:** An apparent geometry advantage can reflect different overfitting
behavior rather than recovered information. A prior baseline is essential.

### 26. Support Recovery (`target_weak_supported`)

**Finding:** At 12,000 rows with the same 120-bin, alpha=0.01 estimator, low-class
support and extreme validation/test estimates disappear in every seed. Mean
logistic losses are 0.665602/0.665555/0.665522 for centered/arcsine/log, a spread
below 0.0001, but remain worse than the 20-bin reference and raw baseline.
**Conclusion:** Support recovery removes this extreme-estimation problem and
largely collapses the geometry gap; excess resolution still carries a cost.

## Overall Summary

**Geometry matters conditionally, not uniformly.** Strong, informative evidence
favors log in the separation, informative-rare-region and variance controls.
Unreliable extreme estimates instead penalize standardized logistic log inputs
in all three targeted seeds, while MLP responses change sign. Transform choice
must therefore be evaluated together with the learner and estimation conditions.

**Binning and smoothing are part of the result, not incidental settings.**
Raw-width versus percentile-width partitions change occupancy; finer partitions
reduce support per bin; smoothing can change which geometry a learner favors.
The controlled resolution contrast has a much larger effect than transform
choice, while support recovery largely collapses the geometry gap. This supports
reporting partition coordinates, bin count, class-specific support and alpha
together, not a universal width/mass/smoothing prescription. Dual/adaptive modes
were not evaluated.

**Estimation quality often matters more than representation.** Historical
relevance also changes evidence before any transform. Raw-feature and prior
controls reveal when a relative representation win still loses useful signal
or merely changes null-task overfitting.

**Scaling and history policy are substantive controls.** Native and standardized
rankings can differ. In the combined-shift path, updating matured evidence helps
the MLP more than changing representation; this is not a general endorsement of
rolling over expanding history. A better transform cannot guarantee repair of
a stale relationship.

**The transforms change learning geometry, not scalar information.** All share
the same fitted q. Evidence-tree test predictions are identical across transforms
throughout this export, while neural learners respond differently. Boundedness
and unboundedness are neither universally good nor universally bad.

These are exploratory synthetic findings, not a benchmark ranking or a claim
about financial-market performance. Cases 1-20 have one seed; targeted cases have
three. Those seeds do not replicate the temporal paths, whose overlapping
outcomes and four test blocks require time-aware uncertainty assessment. Sample
sizes are separately generated, not nested. Hyperparameters are matched rather
than separately optimized; validation-selected epochs can differ, and some fits
reach the training budget. Mean losses are not confidence intervals and do not
account for multiple comparisons. Further experiments are needed for stronger
population or real-world claims, not to force a common representation winner.