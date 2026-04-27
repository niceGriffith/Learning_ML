"""
=============================================================================
  Locally Weighted Regression (LWR) — Theory & Applications
=============================================================================
  Module for Instance_Based_Learning.ipynb

  References
  ----------
  - Tom Mitchell, *Machine Learning*, Chapter 8 §8.3 — Locally Weighted
    Regression
  - Lecture 13 notes (CS456, NISER)

  This module implements Locally Weighted Regression from scratch and
  compares it against k-NN regression and standard linear regression
  on three adapted datasets: Iris, Wine, and a 1D synthetic signal.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  PART I — THEORY DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def print_section(title, width=72):
    """Pretty-print a section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def theory_lwr_introduction():
    """Display introductory theory on locally weighted regression."""
    print_section("1. LOCALLY WEIGHTED REGRESSION — OVERVIEW")
    print("""
Locally Weighted Regression (LWR) is an instance-based method for
approximating REAL-VALUED target functions (Mitchell Ch.8 §8.3).

Key difference from k-NN regression:
  ● k-NN regression:  f̂(x_q) = mean of k nearest neighbours' targets.
  ● LWR:  fits a LOCAL function (e.g. a line or quadratic surface)
    to the neighbourhood around x_q, weighted by distance.

Why "locally weighted"?
  ● "Local"    → only nearby training examples matter.
  ● "Weighted" → closer examples have MORE influence on the fit.
  ● A separate fit is constructed for EACH query — no single global
    model is ever stored.

Contrast with standard (global) linear regression:
  ● Standard LR finds ONE set of coefficients θ that minimises the
    error across ALL training examples.
  ● LWR finds a DIFFERENT set of coefficients for every query x_q,
    emphasising training examples near x_q.

This makes LWR an INSTANCE-BASED (lazy) learner: no training phase,
all work is done at prediction time.
""")


def theory_lwr_formulation():
    """Display the mathematical formulation of LWR."""
    print_section("2. MATHEMATICAL FORMULATION")
    print("""
Given: training set  D = { (x_i, y_i) }_{i=1}^n
Query: new point x_q

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Compute weights for each training example:

    w_i  =  K( d(x_q, x_i) / τ )

  where K is a kernel function and τ (tau) is the BANDWIDTH parameter.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Solve the weighted least squares problem:

    θ*  =  argmin_θ  Σᵢ  w_i · (y_i - θᵀ x_i)²

  In matrix form:
    θ*  =  (XᵀWX)⁻¹ XᵀWy

  where W = diag(w₁, w₂, …, wₙ) is the diagonal weight matrix.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Predict:

    ŷ_q  =  θ*ᵀ x_q
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Notes:
  ● θ* is computed FRESH for every query x_q.
  ● Computational cost: O(n·d²) per query (matrix inversion).
  ● This is much more expensive than k-NN for large n.
""")


def theory_kernel_and_bandwidth():
    """Display information on the kernel function and bandwidth."""
    print_section("3. KERNEL FUNCTION & BANDWIDTH PARAMETER τ")
    print("""
The GAUSSIAN (RBF) kernel is most commonly used:

    K(u)  =  exp( -u² / 2 )

So the weight for training example x_i given query x_q is:

    w_i  =  exp( -‖x_q - x_i‖² / (2τ²) )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE BANDWIDTH PARAMETER  τ  (tau)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────┬──────────────────────────────────────────────────────┐
│   τ value    │   Effect                                             │
├──────────────┼──────────────────────────────────────────────────────┤
│  Very small  │ ● Only the closest examples have non-negligible     │
│  τ ≈ 0.01    │   weight → behaves like k-NN with very small k.     │
│              │ ● HIGH VARIANCE, can overfit.                       │
│              │ ● Fit is very wiggly / jagged.                       │
├──────────────┼──────────────────────────────────────────────────────┤
│  Medium τ    │ ● Good balance between local and global fit.        │
│  τ ≈ 0.5–2   │ ● Captures data trends while smoothing noise.       │
├──────────────┼──────────────────────────────────────────────────────┤
│  Very large  │ ● All examples get nearly equal weight →            │
│  τ ≈ 10+     │   approaches GLOBAL linear regression.              │
│              │ ● HIGH BIAS, can underfit.                          │
└──────────────┴──────────────────────────────────────────────────────┘

Choosing τ:
  ● Use cross-validation to tune.
  ● τ is analogous to k in k-NN: controls the bias-variance trade-off.
  ● Small τ ↔ small k (more complex); large τ ↔ large k (simpler).
""")


def theory_comparison_with_knn():
    """Compare LWR with k-NN regression."""
    print_section("4. COMPARISON: LWR vs k-NN REGRESSION")
    print("""
┌──────────────────────┬─────────────────────┬────────────────────────┐
│       Aspect          │   k-NN Regression   │  Locally Weighted Reg. │
├──────────────────────┼─────────────────────┼────────────────────────┤
│ Local model          │ None (just mean)    │ Linear / polynomial    │
│ Smoothness           │ Piecewise constant  │ Piecewise linear       │
│ Hyper-parameter      │ k (# neighbours)    │ τ (bandwidth)          │
│ Fit quality          │ Step-like            │ Smooth curve           │
│ Computational cost   │ O(n) per query      │ O(n·d²) per query      │
│ Extrapolation        │ Poor                │ Better (linear fit)    │
│ Sensitivity to noise │ Moderate             │ Lower (smoothed)       │
│ Storage              │ Full dataset         │ Full dataset           │
│ Implementation       │ Very simple          │ Moderate               │
└──────────────────────┴─────────────────────┴────────────────────────┘

Both are lazy / instance-based: no training, all work at query time.
The choice depends on whether the underlying function is smooth (→ LWR)
or discontinuous (→ k-NN may be simpler and sufficient).
""")


def theory_pros_cons():
    """Display pros / cons for LWR."""
    print_section("5. LWR SUMMARY — PROS & CONS")
    print("""
┌────────────────────────────────┬───────────────────────────────────┐
│        ADVANTAGES              │        DISADVANTAGES              │
├────────────────────────────────┼───────────────────────────────────┤
│ ✓ Fits very complex, non-     │ ✗ Expensive: must solve weighted  │
│   linear functions locally.    │   least squares per query.        │
│                                │                                   │
│ ✓ Smooth predictions (unlike  │ ✗ Must store entire dataset.      │
│   step-like k-NN).             │                                   │
│                                │ ✗ Bandwidth τ must be tuned.     │
│ ✓ Adapts to local structure   │                                   │
│   at every query point.        │ ✗ Suffers from curse of dim.     │
│                                │                                   │
│ ✓ No distributional            │ ✗ Sensitive to feature scaling.   │
│   assumptions.                 │                                   │
└────────────────────────────────┴───────────────────────────────────┘
""")


def display_all_theory():
    """Call all theory display functions in order."""
    theory_lwr_introduction()
    theory_lwr_formulation()
    theory_kernel_and_bandwidth()
    theory_comparison_with_knn()
    theory_pros_cons()


# ═══════════════════════════════════════════════════════════════════════════
#  PART II — LWR IMPLEMENTATION (from scratch)
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_kernel(distances, tau):
    """
    Compute Gaussian kernel weights.
    w_i = exp( -d_i^2 / (2 * tau^2) )
    """
    return np.exp(- distances ** 2 / (2 * tau ** 2))


def locally_weighted_regression(X_train, y_train, X_query, tau=1.0):
    """
    Predict target values for X_query using Locally Weighted Regression.

    Parameters
    ----------
    X_train : ndarray (n, d)    — training features (already scaled)
    y_train : ndarray (n,)      — training targets
    X_query : ndarray (m, d)    — query points
    tau     : float             — bandwidth parameter

    Returns
    -------
    y_pred  : ndarray (m,)
    """
    n = X_train.shape[0]
    m = X_query.shape[0]
    y_pred = np.zeros(m)

    # Add bias column
    X_aug = np.hstack([np.ones((n, 1)), X_train])

    for j in range(m):
        xq = X_query[j]
        # Distances to all training points
        dists = np.linalg.norm(X_train - xq, axis=1)
        # Kernel weights
        W = np.diag(gaussian_kernel(dists, tau))
        # Augment query
        xq_aug = np.hstack([[1.0], xq])
        # Weighted least squares: θ = (X^T W X)^{-1} X^T W y
        XTWX = X_aug.T @ W @ X_aug
        # Regularise for numerical stability
        XTWX += 1e-8 * np.eye(XTWX.shape[0])
        XTWy = X_aug.T @ W @ y_train
        theta = np.linalg.solve(XTWX, XTWy)
        y_pred[j] = xq_aug @ theta

    return y_pred


# ═══════════════════════════════════════════════════════════════════════════
#  PART III — 1D SYNTHETIC DEMO (best for visualising τ effects)
# ═══════════════════════════════════════════════════════════════════════════

def demo_1d_synthetic():
    """
    Demonstrate LWR on a 1D non-linear function with noise.
    Shows effects of different τ values and comparison with k-NN and LR.
    """
    print_section("DEMO 1 — 1D SYNTHETIC DATA")
    np.random.seed(42)
    n = 120
    X = np.sort(np.random.uniform(-3, 3, n))
    y_true = np.sin(X) * X + 0.5 * np.cos(3 * X)
    y = y_true + np.random.normal(0, 0.35, n)

    X_plot = np.linspace(-3, 3, 300)

    # ─── Effect of different τ ──────────────────────────────────────────
    taus = [0.1, 0.3, 0.8, 2.0, 5.0]
    fig, axes = plt.subplots(1, len(taus), figsize=(20, 4))
    fig.suptitle('LWR with Different Bandwidth τ — 1D Synthetic Data',
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, tau in zip(axes, taus):
        y_pred = locally_weighted_regression(
            X.reshape(-1, 1), y, X_plot.reshape(-1, 1), tau=tau)
        ax.scatter(X, y, c='#90CAF9', s=15, alpha=0.7, edgecolors='none',
                   label='Training data')
        ax.plot(X_plot, np.sin(X_plot) * X_plot + 0.5 * np.cos(3 * X_plot),
                'g--', lw=1.5, alpha=0.6, label='True function')
        ax.plot(X_plot, y_pred, 'r-', lw=2, label=f'LWR (τ={tau})')
        ax.set_title(f'τ = {tau}', fontsize=13, fontweight='bold')
        ax.set_ylim(y.min() - 1, y.max() + 1)
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    plt.show()

    print("""
  Observations:
  ● τ = 0.1 : Extremely wiggly — overfits noise.
  ● τ = 0.3 : Captures local trends well, slight overfitting.
  ● τ = 0.8 : Good balance — tracks the true function smoothly.
  ● τ = 2.0 : Starts to oversmooth, misses some bends.
  ● τ = 5.0 : Nearly a straight line — approaches global LR, underfits.
""")

    # ─── Comparison: LWR vs k-NN Regression vs Linear Regression ───────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle('Comparison: LWR vs k-NN Regression vs Linear Regression',
                 fontsize=14, fontweight='bold', y=1.02)

    # LWR with good τ
    y_lwr = locally_weighted_regression(
        X.reshape(-1, 1), y, X_plot.reshape(-1, 1), tau=0.8)
    axes[0].scatter(X, y, c='#90CAF9', s=15, alpha=0.7, edgecolors='none')
    axes[0].plot(X_plot, y_lwr, 'r-', lw=2, label='LWR (τ=0.8)')
    axes[0].plot(X_plot, np.sin(X_plot)*X_plot+0.5*np.cos(3*X_plot),
                 'g--', lw=1.5, alpha=0.6, label='True')
    axes[0].set_title('LWR (τ=0.8)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # k-NN Regression
    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(X.reshape(-1, 1), y)
    y_knn = knn.predict(X_plot.reshape(-1, 1))
    axes[1].scatter(X, y, c='#90CAF9', s=15, alpha=0.7, edgecolors='none')
    axes[1].plot(X_plot, y_knn, '#FF6F00', lw=2, label='k-NN (k=7)')
    axes[1].plot(X_plot, np.sin(X_plot)*X_plot+0.5*np.cos(3*X_plot),
                 'g--', lw=1.5, alpha=0.6, label='True')
    axes[1].set_title('k-NN Regression (k=7)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    # Global Linear Regression
    lr = LinearRegression()
    lr.fit(X.reshape(-1, 1), y)
    y_lr = lr.predict(X_plot.reshape(-1, 1))
    axes[2].scatter(X, y, c='#90CAF9', s=15, alpha=0.7, edgecolors='none')
    axes[2].plot(X_plot, y_lr, '#9C27B0', lw=2, label='Linear Regression')
    axes[2].plot(X_plot, np.sin(X_plot)*X_plot+0.5*np.cos(3*X_plot),
                 'g--', lw=1.5, alpha=0.6, label='True')
    axes[2].set_title('Global Linear Regression', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    plt.show()

    print("""
  Key take-aways:
  ● LWR gives the SMOOTHEST, best-fitting curve for non-linear data.
  ● k-NN regression produces a STEP-LIKE (piecewise constant) fit.
  ● Global linear regression CANNOT capture non-linearity at all.
""")


# ═══════════════════════════════════════════════════════════════════════════
#  PART IV — IRIS REGRESSION DEMO
# ═══════════════════════════════════════════════════════════════════════════

def demo_iris_regression():
    """
    Regression task on Iris: predict petal width from other features.
    Compare LWR vs k-NN regression vs global LR.
    """
    print_section("DEMO 2 — IRIS (Predict Petal Width)")
    iris = load_iris()
    X = iris.data[:, :3]       # sepal length, sepal width, petal length
    y = iris.data[:, 3]        # petal width (target)
    feature_names = iris.feature_names[:3]

    print(f"  Features : {feature_names}")
    print(f"  Target   : petal width (cm)")
    print(f"  Samples  : {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ─── LWR with various τ ────────────────────────────────────────────
    taus = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    lwr_results = {}
    for tau in taus:
        y_pred = locally_weighted_regression(X_train_s, y_train, X_test_s, tau=tau)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        lwr_results[tau] = {'mse': mse, 'r2': r2}
        print(f"    τ = {tau:<5.1f}  MSE = {mse:.4f}   R² = {r2:.4f}")

    # ─── k-NN regression ───────────────────────────────────────────────
    print("\n  k-NN Regression:")
    knn_results = {}
    for k in [3, 5, 7, 9, 11]:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(X_train_s, y_train)
        y_pred = knn.predict(X_test_s)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        knn_results[k] = {'mse': mse, 'r2': r2}
        print(f"    k = {k:<3d}    MSE = {mse:.4f}   R² = {r2:.4f}")

    # ─── Global LR ────────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_lr = lr.predict(X_test_s)
    lr_mse = mean_squared_error(y_test, y_lr)
    lr_r2 = r2_score(y_test, y_lr)
    print(f"\n  Global LR   MSE = {lr_mse:.4f}   R² = {lr_r2:.4f}")

    # ─── Plot comparison ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Iris Regression: Predicted vs Actual Petal Width',
                 fontsize=14, fontweight='bold', y=1.02)

    # Find best τ
    best_tau = min(lwr_results, key=lambda t: lwr_results[t]['mse'])
    y_lwr = locally_weighted_regression(X_train_s, y_train, X_test_s, tau=best_tau)

    for ax, (y_pred, title, color) in zip(axes, [
        (y_lwr, f'LWR (τ={best_tau})', '#e74c3c'),
        (KNeighborsRegressor(n_neighbors=5, weights='distance').fit(
            X_train_s, y_train).predict(X_test_s), 'k-NN (k=5)', '#FF6F00'),
        (y_lr, 'Global LR', '#9C27B0'),
    ]):
        ax.scatter(y_test, y_pred, c=color, alpha=0.7, edgecolors='white',
                   s=50)
        lims = [min(y_test.min(), y_pred.min()) - 0.1,
                max(y_test.max(), y_pred.max()) + 0.1]
        ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.5, label='Perfect')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        r2 = r2_score(y_test, y_pred)
        ax.set_title(f'{title}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual petal width (cm)')
        ax.set_ylabel('Predicted petal width (cm)')
        ax.legend()
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    plt.show()

    # ─── Tau effect plot ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    tau_list = sorted(lwr_results.keys())
    mse_vals = [lwr_results[t]['mse'] for t in tau_list]
    r2_vals  = [lwr_results[t]['r2'] for t in tau_list]

    ax1.plot(tau_list, mse_vals, 'o-', color='#e74c3c', lw=2, markersize=8)
    ax1.axhline(y=lr_mse, color='#9C27B0', ls='--', lw=1.5, label=f'Global LR (MSE={lr_mse:.4f})')
    ax1.set_xlabel('τ (bandwidth)', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('LWR: MSE vs τ — Iris', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(tau_list, r2_vals, 's-', color='#2ecc71', lw=2, markersize=8)
    ax2.axhline(y=lr_r2, color='#9C27B0', ls='--', lw=1.5, label=f'Global LR (R²={lr_r2:.4f})')
    ax2.set_xlabel('τ (bandwidth)', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('LWR: R² vs τ — Iris', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  PART V — WINE REGRESSION DEMO
# ═══════════════════════════════════════════════════════════════════════════

def demo_wine_regression():
    """
    Regression task on Wine: predict alcohol content from other features.
    """
    print_section("DEMO 3 — WINE (Predict Alcohol Content)")
    wine = load_wine()
    # Feature 0 is 'alcohol' — use as target, rest as features
    X = wine.data[:, 1:]
    y = wine.data[:, 0]

    print(f"  Features : {wine.feature_names[1:]} (12 features)")
    print(f"  Target   : {wine.feature_names[0]}")
    print(f"  Samples  : {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ─── LWR with various τ ─────────────────────────────────────────
    taus = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    lwr_results = {}
    print("\n  LWR Results:")
    for tau in taus:
        y_pred = locally_weighted_regression(X_train_s, y_train, X_test_s, tau=tau)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        lwr_results[tau] = {'mse': mse, 'r2': r2}
        print(f"    τ = {tau:<5.1f}  MSE = {mse:.4f}   R² = {r2:.4f}")

    # ─── k-NN regression ─────────────────────────────────────────────
    print("\n  k-NN Regression:")
    for k in [3, 5, 7, 9]:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(X_train_s, y_train)
        y_pred = knn.predict(X_test_s)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"    k = {k:<3d}    MSE = {mse:.4f}   R² = {r2:.4f}")

    # ─── Global LR ──────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_lr = lr.predict(X_test_s)
    lr_mse = mean_squared_error(y_test, y_lr)
    lr_r2 = r2_score(y_test, y_lr)
    print(f"\n  Global LR   MSE = {lr_mse:.4f}   R² = {lr_r2:.4f}")

    # ─── Plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Wine Regression: Predicted vs Actual Alcohol',
                 fontsize=14, fontweight='bold', y=1.02)

    best_tau = min(lwr_results, key=lambda t: lwr_results[t]['mse'])
    y_lwr = locally_weighted_regression(X_train_s, y_train, X_test_s, tau=best_tau)

    knn_best = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn_best.fit(X_train_s, y_train)
    y_knn = knn_best.predict(X_test_s)

    for ax, (y_pred, title, color) in zip(axes, [
        (y_lwr, f'LWR (τ={best_tau})', '#e74c3c'),
        (y_knn, 'k-NN (k=5)', '#FF6F00'),
        (y_lr, 'Global LR', '#9C27B0'),
    ]):
        ax.scatter(y_test, y_pred, c=color, alpha=0.7, edgecolors='white', s=50)
        lims = [min(y_test.min(), y_pred.min()) - 0.3,
                max(y_test.max(), y_pred.max()) + 0.3]
        ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.5, label='Perfect')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        r2 = r2_score(y_test, y_pred)
        ax.set_title(f'{title}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual alcohol')
        ax.set_ylabel('Predicted alcohol')
        ax.legend()
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    plt.show()

    # ─── τ effect plot ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    tau_list = sorted(lwr_results.keys())
    mse_vals = [lwr_results[t]['mse'] for t in tau_list]
    r2_vals  = [lwr_results[t]['r2'] for t in tau_list]

    ax1.plot(tau_list, mse_vals, 'o-', color='#e74c3c', lw=2, markersize=8)
    ax1.axhline(y=lr_mse, color='#9C27B0', ls='--', lw=1.5,
                label=f'Global LR (MSE={lr_mse:.4f})')
    ax1.set_xlabel('τ', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('LWR: MSE vs τ — Wine', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(tau_list, r2_vals, 's-', color='#2ecc71', lw=2, markersize=8)
    ax2.axhline(y=lr_r2, color='#9C27B0', ls='--', lw=1.5,
                label=f'Global LR (R²={lr_r2:.4f})')
    ax2.set_xlabel('τ', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('LWR: R² vs τ — Wine', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  PART VI — RUN ALL DEMOS
# ═══════════════════════════════════════════════════════════════════════════

def demo_all():
    """Run all LWR demonstrations."""
    demo_1d_synthetic()
    demo_iris_regression()
    demo_wine_regression()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    display_all_theory()
    demo_all()
    print("\n✅  Locally Weighted Regression analysis complete.\n")
