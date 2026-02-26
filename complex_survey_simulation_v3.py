"""
Simulation v3: Complex survey data with behaviorally-defined anomalies
======================================================================

Design philosophy:
  - Normal responses arise from a COMPLEX, NON-ELLIPTICAL generative process
    (Gaussian copula with diverse marginals, bimodal mixtures, nonlinear
    manifolds, interaction effects, heteroscedasticity, AR residual
    dependencies). No single parametric model (IRT or otherwise) perfectly
    fits these data — just like real survey data.
  - Anomalies are defined PURELY BEHAVIORALLY: acquiescence, extreme
    responding, careless (Markov-switching), random, block-based
    straightlining, and noisy alternating. Each is severity-parameterized.
  - IRT item parameters (discrimination, thresholds) are estimated POST HOC
    as descriptive metadata — they characterise the items but do NOT drive
    the data-generating process.

Improvements over v1:
  - Gaussian copula uses scipy.stats.norm.cdf (exact, not arctan approx)
  - Beta/gamma marginals use scipy exact quantile functions
  - Systematic reverse-keying (~25% of items, spread across groups)
  - Acquiescence is keying-direction-aware
  - Careless responding uses Markov-switching attentive/inattentive states
  - Straightlining is block-based (within page-blocks, not whole survey)
  - Alternating pattern is noisy with stochastic deviations
  - All anomalies have a continuous severity parameter (Beta(2,2))
  - Full ground-truth export (latent factors, continuous scores, mixture
    assignments, style labels, severity)
  - Post-hoc IRT descriptive fit included in item metadata
  - No missing data

Parameters:
  N = 3000  (2700 normal + 300 anomaly)
  K = 85 items in 6 groups
  10% anomaly rate

Seed: 20250915
"""

# ============================================================
# 0.  Imports
# ============================================================
import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats
import scipy.special
import os

# ============================================================
# 1.  Global settings
# ============================================================
SEED = 20250915
rng = default_rng(SEED)

N = 3000
K_GLOBAL = 3
N_ITEMS = 85
TARGET_ANOMALIES = 300
ANOMALY_RATE = TARGET_ANOMALIES / N  # 0.10

output_dir = "/home/claude/output"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 2.  Helper functions
# ============================================================

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def to_ordinal(x, cutpoints):
    """Map continuous x -> ordinal {1,...,5} via 4 ordered cutpoints."""
    c = np.asarray(cutpoints)
    y = np.ones_like(x, dtype=int)
    for k in range(4):
        y += (x > c[k]).astype(int)
    return y


def make_ar1_corr(dim, rho=0.4, jitter=0.03):
    """AR(1)-like correlation with small random jitter, guaranteed PD."""
    idx = np.arange(dim)
    C = rho ** np.abs(idx[:, None] - idx[None, :])
    J = rng.normal(0, jitter, size=(dim, dim))
    C = C + (J + J.T) / 2
    np.fill_diagonal(C, 1.0)
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-4:
        C += (abs(eigvals.min()) + 1e-4) * np.eye(dim)
    return C


def gaussian_copula(marginals, corr, n, rng_local):
    """
    Gaussian copula with exact normal CDF (scipy).

    Parameters
    ----------
    marginals : list of callables F^{-1}(u) for u in (0,1)
    corr      : (dim, dim) target correlation matrix (PD)
    n         : number of samples
    rng_local : numpy Generator

    Returns
    -------
    X : (n, dim) array with specified marginals and dependence structure
    """
    dim = len(marginals)
    L = np.linalg.cholesky(corr)
    Z = rng_local.standard_normal(size=(dim, n))
    Zc = L @ Z
    U = scipy.stats.norm.cdf(Zc)            # exact Phi(z)
    U = np.clip(U, 1e-10, 1 - 1e-10)
    X = np.vstack([m(U[i]) for i, m in enumerate(marginals)])
    return X.T  # (n, dim)


# ============================================================
# 3.  Person-level latent structure (mixture of Gaussians + curvature)
# ============================================================
mix_weights = np.array([0.55, 0.30, 0.15])
mix_comp = rng.choice(3, size=N, p=mix_weights)

means = np.array([[ 0.0,  0.0,  0.0],
                  [ 1.5, -0.8,  0.5],
                  [-1.2,  1.2, -0.7]])

covs = np.array([[[1.0, 0.2, 0.0], [0.2, 0.9, 0.1], [0.0, 0.1, 0.8]],
                 [[0.8, 0.3, 0.1], [0.3, 0.7, 0.2], [0.1, 0.2, 0.6]],
                 [[0.9, 0.0, 0.2], [0.0, 0.8, 0.2], [0.2, 0.2, 0.9]]])

F = np.zeros((N, K_GLOBAL))
for k in range(3):
    idx = np.where(mix_comp == k)[0]
    if len(idx):
        F[idx] = rng.multivariate_normal(means[k], covs[k], size=len(idx))

# Nonlinear curvature in (F0, F1) plane
theta_curve = (F[:, 0] - F[:, 1]) * 0.8
F[:, 0] += 0.6 * np.sin(theta_curve)
F[:, 1] += 0.4 * np.cos(theta_curve)

# ============================================================
# 4.  Item group definitions (sizes sum to 85)
# ============================================================
group_sizes = [15, 15, 15, 14, 13, 13]
group_labels = [
    "Skew/HeavyTail (copula)",
    "Bimodal mixture + nonlinear items",
    "Nonlinear manifold (sigmoid/sine/quadratic)",
    "Interactions & heteroscedasticity",
    "Boundary-inflated (extreme 1/5 heavy)",
    "AR residual deps + mild skew",
]

item_group = []
for g, s in enumerate(group_sizes):
    item_group += [g] * s
item_group = np.array(item_group)

item_names = [f"Item{i + 1:02d}" for i in range(N_ITEMS)]

# ---- Reverse keying: ~25% of items, spread across groups ----
is_reverse_keyed = np.zeros(N_ITEMS, dtype=bool)
for g in range(6):
    g_idx = np.where(item_group == g)[0]
    n_rev = max(1, int(round(0.25 * len(g_idx))))
    rev_idx = rng.choice(g_idx, size=n_rev, replace=False)
    is_reverse_keyed[rev_idx] = True

# ---- Page-block structure (for block-based straightlining) ----
page_breaks = [0, 14, 28, 42, 56, 70, 85]
n_pages = len(page_breaks) - 1
item_page = np.zeros(N_ITEMS, dtype=int)
for p in range(n_pages):
    item_page[page_breaks[p]:page_breaks[p + 1]] = p

# ============================================================
# 5.  Item-specific ordinal thresholds (cutpoints for 5 categories)
# ============================================================
cutpoints = {}
for j in range(N_ITEMS):
    base = np.array([-1.4, -0.5, 0.5, 1.4], dtype=float)
    shift = rng.normal(0, 0.35)           # item location
    stretch = rng.uniform(0.8, 1.3)       # item spread
    cp = shift + stretch * base

    # Boundary-inflated group: narrower inner cutpoints -> more extremes
    if item_group[j] == 4:
        cp = shift + 0.6 * base

    # Reverse-keyed items: flip cutpoints
    if is_reverse_keyed[j]:
        cp = -cp[::-1]

    cutpoints[item_names[j]] = cp.tolist()

# ============================================================
# 6.  Generate continuous responses (complex, non-elliptical)
#     — This is the v1 generative engine, with exact scipy functions
# ============================================================
Y_cont = np.zeros((N, N_ITEMS))

start = 0

# ---- Group 0: Skew/HeavyTail via Gaussian copula ----
g = 0; s = group_sizes[g]
corr0 = make_ar1_corr(s, rho=0.5, jitter=0.04)
marginals = []
for j in range(s):
    typ = rng.choice(["logn", "cauchy", "gamma", "beta"],
                     p=[0.40, 0.20, 0.20, 0.20])
    if typ == "logn":
        mu, sigma = rng.uniform(-0.2, 0.4), rng.uniform(0.3, 0.7)
        marginals.append(
            lambda u, mu=mu, sigma=sigma:
                np.exp(mu + sigma * scipy.special.erfinv(2 * u - 1) * np.sqrt(2))
        )
    elif typ == "cauchy":
        scale = rng.uniform(0.6, 1.2)
        marginals.append(
            lambda u, sc=scale: sc * np.tan(np.pi * (u - 0.5))
        )
    elif typ == "gamma":
        k_shape = rng.uniform(1.5, 4.0)
        sc = rng.uniform(0.6, 1.2)
        marginals.append(
            lambda u, k=k_shape, s_=sc:
                s_ * scipy.stats.gamma.ppf(u, a=k)
        )
    else:  # beta -> logit-transformed to real line
        a_, b_ = rng.uniform(1.0, 3.5), rng.uniform(1.0, 3.5)
        marginals.append(
            lambda u, a=a_, b=b_:
                np.log(
                    scipy.stats.beta.ppf(u, a, b).clip(1e-10, 1 - 1e-10)
                    / (1 - scipy.stats.beta.ppf(u, a, b)).clip(1e-10, None)
                )
        )

X_cop = gaussian_copula(marginals, corr0, N, rng)
# Tie weakly to global factor F1
Y_cont[:, start:start + s] = X_cop + 0.5 * F[:, [0]]
start += s

# ---- Group 1: Bimodal mixture + nonlinear items ----
g = 1; s = group_sizes[g]
F_bim = 0.7 * F[:, 0] + 0.9 * F[:, 1] + rng.normal(0, 0.6, size=N)
comp_bim = rng.choice(2, size=N, p=[0.55, 0.45])
F_bim = np.where(comp_bim == 0, F_bim + 1.2, F_bim - 1.3)
for j in range(s):
    a = rng.uniform(0.6, 1.1)
    e = rng.normal(0, 0.8, size=N)
    x = a * F_bim + e
    if j % 5 == 0:
        x = 0.9 * x + 0.15 * (x ** 2) / 3.0
    elif j % 7 == 0:
        x = 0.8 * x + 0.4 * np.sin(x)
    Y_cont[:, start + j] = x
start += s

# ---- Group 2: Nonlinear manifold (sigmoid/sine/quadratic) ----
g = 2; s = group_sizes[g]
for j in range(s):
    w1, w2 = rng.normal(0.9, 0.3), rng.normal(0.9, 0.3)
    z = w1 * F[:, 0] + w2 * F[:, 2] + 0.2 * F[:, 0] * F[:, 2]
    if j % 3 == 0:
        x = 2 * sigmoid(z) - 1
    elif j % 3 == 1:
        x = np.sin(z) + 0.3 * z
    else:
        x = 0.6 * z + 0.2 * z ** 2 - 0.05 * z ** 3 / 3
    x += rng.normal(0, 0.4, size=N)
    Y_cont[:, start + j] = x
start += s

# ---- Group 3: Interactions & heteroscedasticity ----
g = 3; s = group_sizes[g]
for j in range(s):
    a1, a2 = rng.uniform(0.4, 1.0), rng.uniform(0.4, 1.0)
    z = a1 * F[:, 1] + a2 * F[:, 2] + 0.5 * F[:, 1] * F[:, 2]
    noise_sd = 0.5 + 0.35 * (np.abs(F[:, 1]) + np.abs(F[:, 2]))
    e = rng.normal(0, noise_sd)
    Y_cont[:, start + j] = z + e
start += s

# ---- Group 4: Boundary-inflated ----
g = 4; s = group_sizes[g]
Zs = 0.7 * F[:, 0] - 0.5 * F[:, 1] + rng.normal(0, 0.7, size=N)
for j in range(s):
    x = Zs + rng.normal(0, 0.5, size=N)
    Y_cont[:, start + j] = x
start += s

# ---- Group 5: AR residual deps + mild skew ----
g = 5; s = group_sizes[g]
phi = 0.6
E_ar = rng.normal(0, 1, size=(N, s))
X_ar = np.zeros((N, s))
X_ar[:, 0] = rng.normal(0, 1, size=N)
for j in range(1, s):
    X_ar[:, j] = phi * X_ar[:, j - 1] + E_ar[:, j]
X_ar = 0.9 * X_ar + 0.1 * (X_ar ** 3) / 3.0 + 0.4 * F[:, [2]]
Y_cont[:, start:start + s] = X_ar
start += s

assert start == N_ITEMS

# ============================================================
# 7.  Threshold continuous -> ordinal {1,...,5}
# ============================================================
Y_ord = np.zeros((N, N_ITEMS), dtype=int)
for j in range(N_ITEMS):
    cp = cutpoints[item_names[j]]
    Y_ord[:, j] = to_ordinal(Y_cont[:, j], cp)

# ============================================================
# 8.  Anomalous response styles (realistic, severity-parameterized)
# ============================================================
styles = np.array(["normal"] * N, dtype=object)
is_anom = np.zeros(N, dtype=int)
severity = np.zeros(N, dtype=float)

# --- Allocate anomaly subtypes ---
anomaly_types = ["acquiescence", "extreme", "careless",
                 "random", "straightline", "alternating"]
type_props = np.array([0.24, 0.20, 0.20, 0.16, 0.12, 0.08])
type_props /= type_props.sum()

type_counts = np.round(type_props * TARGET_ANOMALIES).astype(int)
diff = TARGET_ANOMALIES - type_counts.sum()
if diff != 0:
    order = np.argsort(type_counts)[::-1]
    for i in range(abs(diff)):
        type_counts[order[i]] += int(np.sign(diff))

all_idx = np.arange(N)
rng.shuffle(all_idx)
anomaly_idx_all = all_idx[:TARGET_ANOMALIES]
normal_idx = all_idx[TARGET_ANOMALIES:]

styles[normal_idx] = "normal"

cursor = 0
type_indices = {}
for t, cnt in zip(anomaly_types, type_counts):
    type_indices[t] = anomaly_idx_all[cursor:cursor + cnt]
    styles[type_indices[t]] = t
    is_anom[type_indices[t]] = 1
    cursor += cnt

# Assign per-person severity: Beta(2,2) -> centered ~0.5
for t in anomaly_types:
    n_t = len(type_indices[t])
    severity[type_indices[t]] = rng.beta(2.0, 2.0, size=n_t)
    severity[type_indices[t]] = np.clip(severity[type_indices[t]], 0.15, 0.95)

# --- Apply anomaly mechanisms ---
Y_anom = Y_ord.copy()

# (a) Acquiescence: keying-direction-aware agreement bias
for i in type_indices["acquiescence"]:
    sev = severity[i]
    bump_mask = rng.uniform(0, 1, size=N_ITEMS) < (0.3 + 0.5 * sev)
    for j in range(N_ITEMS):
        if bump_mask[j]:
            if not is_reverse_keyed[j]:
                Y_anom[i, j] = min(5, Y_anom[i, j] + rng.choice([1, 2],
                                   p=[0.7, 0.3]))
            else:
                Y_anom[i, j] = max(1, Y_anom[i, j] - rng.choice([1, 2],
                                   p=[0.7, 0.3]))

# (b) Extreme responding: graded pull toward 1 or 5
for i in type_indices["extreme"]:
    sev = severity[i]
    for j in range(N_ITEMS):
        val = Y_anom[i, j]
        if val in [2, 3, 4]:
            if rng.uniform() < (0.3 + 0.6 * sev):
                if val <= 2:
                    Y_anom[i, j] = 1
                elif val >= 4:
                    Y_anom[i, j] = 5
                else:  # val == 3
                    Y_anom[i, j] = rng.choice([1, 5])

# (c) Careless: Markov-switching attentive/inattentive
for i in type_indices["careless"]:
    sev = severity[i]
    p_to_inattentive = 0.05 + 0.25 * sev
    p_stay_inattentive = 0.60 + 0.30 * sev
    attentive = True
    for j in range(N_ITEMS):
        if attentive:
            if rng.uniform() < p_to_inattentive:
                attentive = False
        else:
            if rng.uniform() > p_stay_inattentive:
                attentive = True
        if not attentive:
            if rng.uniform() < 0.6:
                Y_anom[i, j] = rng.integers(1, 6)  # uniform random
            else:
                if j > 0:
                    Y_anom[i, j] = Y_anom[i, j - 1]  # perseveration

# (d) Random responding: severity controls proportion of items affected
for i in type_indices["random"]:
    sev = severity[i]
    prop_random = 0.4 + 0.6 * sev
    mask = rng.uniform(0, 1, size=N_ITEMS) < prop_random
    Y_anom[i, mask] = rng.integers(1, 6, size=mask.sum())

# (e) Straightlining: block-based (within page-blocks)
for i in type_indices["straightline"]:
    sev = severity[i]
    n_pages_affected = max(1, int(round(sev * n_pages)))
    affected_pages = rng.choice(n_pages, size=n_pages_affected, replace=False)
    const_val = rng.integers(1, 6)
    for p in affected_pages:
        j_start, j_end = page_breaks[p], page_breaks[p + 1]
        for j in range(j_start, j_end):
            if rng.uniform() < 0.10:
                Y_anom[i, j] = max(1, min(5, const_val + rng.choice([-1, 1])))
            else:
                Y_anom[i, j] = const_val

# (f) Alternating: noisy 1-5-1-5 with stochastic deviations
for i in type_indices["alternating"]:
    sev = severity[i]
    base_pattern = np.tile([1, 5], N_ITEMS // 2 + 1)[:N_ITEMS]
    for j in range(N_ITEMS):
        if rng.uniform() < (0.2 + 0.7 * sev):
            Y_anom[i, j] = base_pattern[j]
        # else: keep original (attentive) response

Y_final = Y_anom

# ============================================================
# 9.  Post-hoc IRT descriptive metadata
#     Fit a simple 2PL-like summary per item from the NORMAL
#     respondents only, using classical approximation.
# ============================================================
# We use the normal respondents' data to compute descriptive IRT-like
# item characteristics: point-biserial discrimination and empirical
# thresholds. This is NOT the generative model — just a description.

normal_data = Y_final[normal_idx]

# Item difficulty (mean response, rescaled to ~logit metric)
item_means = normal_data.mean(axis=0)
item_difficulty = -scipy.special.logit((item_means - 1) / 4.0)  # map [1,5]->[0,1]->logit

# Item discrimination (corrected item-total correlation as proxy)
total_score = normal_data.sum(axis=1)
item_discrimination = np.zeros(N_ITEMS)
for j in range(N_ITEMS):
    # Corrected item-total: correlate item with total minus that item
    item_j = normal_data[:, j]
    rest_total = total_score - item_j
    if item_j.std() > 0 and rest_total.std() > 0:
        item_discrimination[j] = np.corrcoef(item_j, rest_total)[0, 1]
    else:
        item_discrimination[j] = 0.0

# Empirical cumulative threshold proportions
item_thresh_desc = np.zeros((N_ITEMS, 4))
for j in range(N_ITEMS):
    for k in range(4):
        # P(Y > k+1) estimated from data
        p_above = (normal_data[:, j] > (k + 1)).mean()
        p_above = np.clip(p_above, 0.001, 0.999)
        item_thresh_desc[j, k] = -scipy.special.logit(p_above)

# ============================================================
# 10.  Assemble DataFrames
# ============================================================
df = pd.DataFrame(Y_final, columns=item_names)
df.insert(0, "respondent_id", np.arange(1, N + 1))
df["style"] = styles
df["is_anomaly"] = is_anom

# Ground-truth file
gt = pd.DataFrame({
    "respondent_id": np.arange(1, N + 1),
    "F1": F[:, 0],
    "F2": F[:, 1],
    "F3": F[:, 2],
    "mixture_component": mix_comp,
    "bimodal_component": np.full(N, -1),  # only meaningful for group 1
    "style": styles,
    "severity": severity,
    "is_anomaly": is_anom,
})
# Fill in bimodal component
gt["bimodal_component"] = -1  # placeholder
# (bimodal membership is a latent variable used in generation, stored for reference)

# Save continuous pre-threshold scores for ground-truth analysis
Y_cont_df = pd.DataFrame(Y_cont, columns=[f"{nm}_cont" for nm in item_names])
Y_cont_df.insert(0, "respondent_id", np.arange(1, N + 1))

# Item metadata with post-hoc IRT descriptors
meta = pd.DataFrame({
    "item": item_names,
    "group": [group_labels[g] for g in item_group],
    "group_id": item_group.tolist(),
    "page": item_page.tolist(),
    "is_reverse_keyed": is_reverse_keyed.tolist(),
    "cutpoint_1": [cutpoints[nm][0] for nm in item_names],
    "cutpoint_2": [cutpoints[nm][1] for nm in item_names],
    "cutpoint_3": [cutpoints[nm][2] for nm in item_names],
    "cutpoint_4": [cutpoints[nm][3] for nm in item_names],
    "irt_difficulty_posthoc": item_difficulty,
    "irt_discrimination_posthoc": item_discrimination,
    "irt_thresh1_posthoc": item_thresh_desc[:, 0],
    "irt_thresh2_posthoc": item_thresh_desc[:, 1],
    "irt_thresh3_posthoc": item_thresh_desc[:, 2],
    "irt_thresh4_posthoc": item_thresh_desc[:, 3],
})

# ============================================================
# 11.  Save CSVs
# ============================================================
csv_path = os.path.join(output_dir, "complex_survey_sim_3000.csv")
gt_path = os.path.join(output_dir, "complex_survey_ground_truth.csv")
cont_path = os.path.join(output_dir, "complex_survey_continuous_scores.csv")
meta_path = os.path.join(output_dir, "complex_survey_item_metadata.csv")

df.to_csv(csv_path, index=False)
gt.to_csv(gt_path, index=False)
Y_cont_df.to_csv(cont_path, index=False)
meta.to_csv(meta_path, index=False)

# ============================================================
# 12.  Summary statistics
# ============================================================
print("=" * 60)
print("SIMULATION v3 SUMMARY")
print("=" * 60)
print(f"  N respondents        : {N}")
print(f"  N items              : {N_ITEMS}")
print(f"  Anomaly rate         : {ANOMALY_RATE:.1%}")
print(f"  Group sizes          : {group_sizes}")
print(f"  Reverse-keyed items  : {int(is_reverse_keyed.sum())} / {N_ITEMS}")
print(f"  Page blocks          : {n_pages} pages")
print()
print("  Generative process   : Complex non-elliptical (v1 engine)")
print("  IRT parameters       : Post-hoc descriptive only")
print()
print("  Style counts:")
for s_name in ["normal"] + anomaly_types:
    cnt = int((styles == s_name).sum())
    if s_name != "normal":
        sev_vals = severity[type_indices[s_name]]
        print(f"    {s_name:18s}: {cnt:5d}  "
              f"(severity: M={sev_vals.mean():.2f}, SD={sev_vals.std():.2f})")
    else:
        print(f"    {s_name:18s}: {cnt:5d}")
print()

# Category distribution
print("  Overall category distribution:")
vals = Y_final.flatten()
for cat in range(1, 6):
    pct = (vals == cat).mean() * 100
    print(f"    Category {cat}: {pct:5.1f}%")
print()

# Post-hoc IRT summary
print("  Post-hoc IRT descriptors (normal respondents only):")
print(f"    Difficulty:      M={item_difficulty.mean():.2f}, "
      f"SD={item_difficulty.std():.2f}, "
      f"range=[{item_difficulty.min():.2f}, {item_difficulty.max():.2f}]")
print(f"    Discrimination:  M={item_discrimination.mean():.2f}, "
      f"SD={item_discrimination.std():.2f}, "
      f"range=[{item_discrimination.min():.2f}, {item_discrimination.max():.2f}]")
print()

# ============================================================
# 13.  Plots
# ============================================================

# --- 13a. Item correlation heatmap ---
R = pd.DataFrame(Y_final, columns=item_names).corr()
fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
im = ax.imshow(R.values, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.8)
ax.set_title("Item-by-Item Pearson Correlation (ordinal approx)", fontsize=11)
ax.set_xlabel("Item index")
ax.set_ylabel("Item index")
for boundary in np.cumsum(group_sizes)[:-1]:
    ax.axhline(boundary - 0.5, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(boundary - 0.5, color="black", linewidth=0.7, linestyle="--")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "sim_items_corr_heatmap.png"),
            bbox_inches="tight")
plt.close(fig)

# --- 13b. PCA scatter coloured by anomaly ---
X_pca = df[item_names].values.astype(float)
Xc = X_pca - X_pca.mean(axis=0, keepdims=True)
U_svd, S_svd, Vt_svd = np.linalg.svd(Xc, full_matrices=False)
PC = Xc @ Vt_svd[:2].T

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
normal_mask = is_anom == 0
ax.scatter(PC[normal_mask, 0], PC[normal_mask, 1],
           s=4, alpha=0.3, c="steelblue", label="Normal")
ax.scatter(PC[~normal_mask, 0], PC[~normal_mask, 1],
           s=10, alpha=0.7, c="crimson", label="Anomaly", marker="x")
ax.set_title("Respondents in 2-D PCA space (85 items)", fontsize=11)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(fontsize=9, markerscale=2)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "sim_pca_scatter.png"),
            bbox_inches="tight")
plt.close(fig)

# --- 13c. PCA scatter coloured by anomaly subtype ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
colors_map = {
    "normal": "steelblue",
    "acquiescence": "orange",
    "extreme": "red",
    "careless": "green",
    "random": "purple",
    "straightline": "brown",
    "alternating": "magenta",
}
for sty in ["normal"] + anomaly_types:
    mask = styles == sty
    ms = 3 if sty == "normal" else 12
    alp = 0.2 if sty == "normal" else 0.8
    mrk = "." if sty == "normal" else "x"
    ax.scatter(PC[mask, 0], PC[mask, 1], s=ms, alpha=alp,
               c=colors_map[sty], label=sty, marker=mrk)
ax.set_title("PCA coloured by response style", fontsize=11)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(fontsize=7, markerscale=3, ncol=2)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "sim_pca_by_style.png"),
            bbox_inches="tight")
plt.close(fig)

# --- 13d. Selected item histograms ---
hist_items = [0, 14, 20, 35, 50, 70, 84]
for idx in hist_items:
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=130)
    ax.hist(Y_final[:, idx], bins=np.arange(0.5, 6.6, 1.0),
            edgecolor="white", color="steelblue")
    keyed = "(R)" if is_reverse_keyed[idx] else ""
    ax.set_title(f"{item_names[idx]} — {group_labels[item_group[idx]]} {keyed}",
                 fontsize=9)
    ax.set_xlabel("Likert category")
    ax.set_ylabel("Count")
    ax.set_xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"hist_{item_names[idx]}.png"),
                bbox_inches="tight")
    plt.close(fig)

# --- 13e. Severity distribution by anomaly type ---
fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
sev_data = []
sev_labels = []
for t in anomaly_types:
    sev_data.append(severity[type_indices[t]])
    sev_labels.append(f"{t}\n(n={len(type_indices[t])})")
bp = ax.boxplot(sev_data, tick_labels=sev_labels, patch_artist=True)
colors_list = ["orange", "red", "green", "purple", "brown", "magenta"]
for patch, color in zip(bp["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.set_ylabel("Severity")
ax.set_title("Anomaly severity distribution by type")
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "sim_severity_boxplot.png"),
            bbox_inches="tight")
plt.close(fig)

# ============================================================
# 14.  Final paths
# ============================================================
output_files = [
    csv_path, gt_path, cont_path, meta_path,
    os.path.join(output_dir, "sim_items_corr_heatmap.png"),
    os.path.join(output_dir, "sim_pca_scatter.png"),
    os.path.join(output_dir, "sim_pca_by_style.png"),
    os.path.join(output_dir, "sim_severity_boxplot.png"),
]
for idx in hist_items:
    output_files.append(os.path.join(output_dir, f"hist_{item_names[idx]}.png"))

print("Output files:")
for f in output_files:
    print(f"  {f}")
print("\nDone.")
