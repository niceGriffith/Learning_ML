"""
=============================================================================
  k-Nearest Neighbours (k-NN) — Theory & Applications
=============================================================================
  Module for Instance_Based_Learning.ipynb

  References
  ----------
  - Tom Mitchell, *Machine Learning*, Chapter 8 — Instance-Based Learning
  - Lecture 13 notes (CS456, NISER)
  - Peter Harrington, *Machine Learning in Action*, Chapter 2

  Datasets (from scikit-learn):
      1. Iris          — 4 features, 3 classes
      2. Wine          — 13 features, 3 classes
      3. Digits        — 64 features (8×8 pixel images), 10 classes
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)
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


def theory_introduction():
    """Display introductory theory on instance-based / lazy learning."""
    print_section("1. INSTANCE-BASED LEARNING — INTRODUCTION")
    print("""
Instance-based learning (also called *lazy learning* or *memory-based
learning*) defers generalisation until a new query is received.

Key idea (Mitchell Ch.8):
  ● Instead of building an explicit model during training, the learner
    simply STORES all training examples.
  ● When a new query instance x_q arrives, the stored examples are used
    to construct a LOCAL approximation to the target function.
  ● Only the region of the instance space surrounding x_q is examined.

Why "lazy"?
  ● No computation is performed at training time.
  ● All computation is deferred to classification / prediction time.
  ● Contrast with "eager" learners (decision trees, neural nets, SVMs)
    that build a global model before seeing any query.

Advantages of lazy learning:
  ✓ The hypothesis can be different for every query → can represent
    very complex (even discontinuous) target functions.
  ✓ No information is discarded during training.

Disadvantages:
  ✗ Classification is expensive (must scan stored examples).
  ✗ All examples must be kept in memory.
  ✗ Distance metrics may be misleading in high dimensions
    (the *curse of dimensionality*).
""")


def theory_knn_algorithm():
    """Display k-NN algorithm description and pseudocode."""
    print_section("2. THE k-NN ALGORITHM")
    print("""
k-Nearest Neighbours (k-NN) is the simplest instance-based learner.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM: k-Nearest Neighbour Classification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Training:   Store all training examples  { (x_i, y_i) }

  Classification of a new query x_q:

      1. Compute  d(x_q, x_i)  for every stored example x_i.
      2. Sort examples by distance (ascending).
      3. Select the k nearest neighbours  N_k(x_q).
      4. Return the majority class among  N_k(x_q):

              ŷ = argmax_v  Σ_{(x_i,y_i) ∈ N_k}  δ(v, y_i)

         where δ(v, y_i) = 1  if  v = y_i,  else 0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PSEUDOCODE  (from lec13 notes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FOR every point in the dataset:
      calculate the distance between x_q and the current point
  SORT the distances in increasing order
  TAKE the k items with the lowest distances
  FIND the majority class among these k items
  RETURN the majority class as the prediction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOR REAL-VALUED TARGET FUNCTIONS (Regression):

      f̂(x_q)  =  (1/k)  Σ_{(x_i,y_i) ∈ N_k}  y_i

  i.e., take the MEAN of the k neighbours' target values.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Special notes:
  • k=1:  Nearest-neighbour rule.  The Voronoi diagram of the training
    set defines the decision boundaries exactly.
  • Odd k is preferred so that majority voting never ties.
  • If a tie occurs, break by distance-weighting or random selection.
""")


def theory_distance_metrics():
    """Display distance metric formulas and comparisons."""
    print_section("3. DISTANCE METRICS")
    print("""
The choice of distance metric profoundly affects k-NN performance.
Below are four widely-used metrics.

┌────────────────────────────────────────────────────────────────────┐
│  EUCLIDEAN DISTANCE  (L₂ norm)                                    │
│                                                                    │
│      d(x, y) = √( Σᵢ (xᵢ - yᵢ)² )                               │
│                                                                    │
│  • Default metric for k-NN.                                       │
│  • Sensitive to feature scaling → always normalise features first. │
│  • Works well when all features are equally important.             │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│  MANHATTAN DISTANCE  (L₁ norm)                                    │
│                                                                    │
│      d(x, y) = Σᵢ |xᵢ - yᵢ|                                      │
│                                                                    │
│  • "City-block" distance.                                          │
│  • More robust to outliers than Euclidean.                         │
│  • Preferred in high-dimensional spaces (less affected by curse   │
│    of dimensionality).                                             │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│  MINKOWSKI DISTANCE  (Lₚ norm)                                    │
│                                                                    │
│      d(x, y) = ( Σᵢ |xᵢ - yᵢ|ᵖ )^(1/p)                          │
│                                                                    │
│  • Generalisation: p=1 → Manhattan, p=2 → Euclidean.              │
│  • p=3 provides a blend of both.                                   │
│  • The parameter p controls sensitivity to large differences.      │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│  CHEBYSHEV DISTANCE  (L∞ norm)                                    │
│                                                                    │
│      d(x, y) = maxᵢ |xᵢ - yᵢ|                                    │
│                                                                    │
│  • The limiting case of Minkowski as p → ∞.                       │
│  • Only the single largest feature difference matters.             │
│  • Useful when any single large deviation should dominate.         │
└────────────────────────────────────────────────────────────────────┘

FEATURE SCALING is critical:
  If features have vastly different scales (e.g., age ∈ [0, 100] vs
  salary ∈ [30000, 200000]), the high-magnitude feature will dominate
  the distance.  Always apply StandardScaler or MinMaxScaler first.
""")


def theory_choosing_k():
    """Display theory on the effect of k."""
    print_section("4. EFFECT OF THE HYPER-PARAMETER k")
    print("""
The value of k controls the INDUCTIVE BIAS of k-NN.

┌──────────────┬─────────────────────────────────────────────────────┐
│   k value    │   Effect                                            │
├──────────────┼─────────────────────────────────────────────────────┤
│   k = 1      │ ● Decision boundary perfectly conforms to training  │
│              │   data (Voronoi tessellation).                      │
│              │ ● HIGH VARIANCE, LOW BIAS.                          │
│              │ ● Sensitive to noise / outliers.                    │
│              │ ● Can overfit.                                      │
├──────────────┼─────────────────────────────────────────────────────┤
│  Small k     │ ● Complex, jagged decision boundaries.             │
│  (3–5)       │ ● Good at capturing local structure.                │
│              │ ● Still somewhat sensitive to noise.                │
├──────────────┼─────────────────────────────────────────────────────┤
│  Medium k    │ ● Smoother decision boundary.                      │
│  (7–15)      │ ● Balances bias and variance.                      │
│              │ ● Often the best generalisation.                    │
├──────────────┼─────────────────────────────────────────────────────┤
│  Large k     │ ● Very smooth (nearly linear) boundary.            │
│  (≫ √n)      │ ● HIGH BIAS, LOW VARIANCE.                         │
│              │ ● Under-fits: ignores local structure.              │
│              │ ● In the limit k = n, predicts the majority class.  │
└──────────────┴─────────────────────────────────────────────────────┘

Rule of thumb:  k ≈ √n   (n = number of training examples).

Bias–Variance trade-off:
  ● Increasing k  → increases bias, decreases variance
  ● Decreasing k  → decreases bias, increases variance
  ● Choose k via cross-validation (see application section).
""")


def theory_distance_weighted_knn():
    """Display theory on distance-weighted k-NN."""
    print_section("5. DISTANCE-WEIGHTED k-NN")
    print("""
Standard k-NN treats all k neighbours equally.  This can be improved
by giving closer neighbours MORE influence (Mitchell Ch.8 §8.2.1).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DISTANCE-WEIGHTED CLASSIFICATION:

      ŷ = argmax_v  Σ_{(x_i,y_i) ∈ N_k}  w_i · δ(v, y_i)

      where  w_i  =  1 / d(x_q, x_i)²

  If d(x_q, x_i) = 0  →  just assign y_i immediately.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DISTANCE-WEIGHTED REGRESSION (real-valued):

      f̂(x_q)  =  Σ w_i · y_i  /  Σ w_i
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Benefits:
  ✓ Reduces sensitivity to the choice of k.
  ✓ Even with k = n (ALL examples), only nearby points contribute
    meaningfully because distant points have negligible weight.
  ✓ Often yields higher accuracy than uniform weighting.

In scikit-learn this is activated by setting  weights='distance'.
""")


def theory_curse_of_dimensionality():
    """Display theory on curse of dimensionality."""
    print_section("6. CURSE OF DIMENSIONALITY")
    print("""
As the number of features (dimensions) d grows, several problems arise
for instance-based learners (Mitchell Ch.8 §8.5):

  1. DISTANCE CONCENTRATION:
     In high-d spaces, the ratio of nearest-to-farthest neighbour
     distance approaches 1 — all points appear equally far apart.

  2. EXPONENTIAL DATA REQUIREMENT:
     To maintain the same local density, the number of training
     examples must grow exponentially with d.

  3. IRRELEVANT FEATURES:
     If many features are irrelevant, they add noise to the distance
     and degrade k-NN performance severely.

Mitigations:
  → Feature selection / dimensionality reduction (PCA, LDA).
  → Feature weighting in the distance metric.
  → Use Manhattan distance (less affected than Euclidean).
""")


def theory_pros_cons():
    """Display pros / cons summary."""
    print_section("7. k-NN SUMMARY — PROS & CONS")
    print("""
┌────────────────────────────────┬───────────────────────────────────┐
│        ADVANTAGES              │        DISADVANTAGES              │
├────────────────────────────────┼───────────────────────────────────┤
│ ✓ Simple to understand and     │ ✗ Slow at prediction time —      │
│   implement.                   │   must compute distance to ALL   │
│                                │   training examples.             │
│ ✓ No training phase.           │                                   │
│                                │ ✗ Must store entire dataset      │
│ ✓ Naturally handles multi-     │   (high memory).                  │
│   class problems.              │                                   │
│                                │ ✗ Sensitive to irrelevant        │
│ ✓ Can learn very complex       │   features and feature scaling.  │
│   decision boundaries.         │                                   │
│                                │ ✗ No interpretable model — no    │
│ ✓ No assumptions about data    │   insight into data structure.    │
│   distribution.                │                                   │
│                                │ ✗ Performance degrades in high   │
│ ✓ Insensitive to outliers      │   dimensions.                     │
│   (for larger k).              │                                   │
└────────────────────────────────┴───────────────────────────────────┘

Works with:  Numeric AND nominal features (with appropriate metric).
Typical use: Classification, regression, anomaly detection, imputation.
""")


def display_all_theory():
    """Call all theory display functions in order."""
    theory_introduction()
    theory_knn_algorithm()
    theory_distance_metrics()
    theory_choosing_k()
    theory_distance_weighted_knn()
    theory_curse_of_dimensionality()
    theory_pros_cons()


# ═══════════════════════════════════════════════════════════════════════════
#  PART II — HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def load_and_describe(loader, name):
    """Load a sklearn dataset, print description, return X, y, names."""
    data = loader()
    X, y = data.data, data.target
    target_names = data.target_names
    feature_names = (data.feature_names if hasattr(data, 'feature_names')
                     and data.feature_names is not None
                     else [f"f{i}" for i in range(X.shape[1])])
    print(f"\n{'─'*60}")
    print(f"  Dataset : {name}")
    print(f"  Samples : {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes : {len(target_names)}  →  {list(target_names)}")
    print(f"{'─'*60}")
    return X, y, target_names, feature_names


# ═══════════════════════════════════════════════════════════════════════════
#  PART III — APPLICATION: EFFECT OF k
# ═══════════════════════════════════════════════════════════════════════════

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21]
METRICS = {
    'Euclidean': {'metric': 'minkowski', 'p': 2},
    'Manhattan': {'metric': 'minkowski', 'p': 1},
    'Minkowski (p=3)': {'metric': 'minkowski', 'p': 3},
    'Chebyshev': {'metric': 'chebyshev'},
}


def run_knn_k_analysis(X_train, X_test, y_train, y_test, k_values=None,
                        dataset_name="Dataset"):
    """
    Train k-NN for multiple k values (Euclidean, uniform weighting).
    Returns dict {k: accuracy}.
    """
    if k_values is None:
        k_values = K_VALUES
    results = {}
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results[k] = acc
    return results


def plot_accuracy_vs_k(results_dict, dataset_name="Dataset"):
    """
    Plot accuracy vs k for one or more settings.
    results_dict: {label: {k: accuracy}}
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = plt.cm.Set2.colors
    for idx, (label, results) in enumerate(results_dict.items()):
        ks = sorted(results.keys())
        accs = [results[k] for k in ks]
        colour = colours[idx % len(colours)]
        ax.plot(ks, accs, 'o-', color=colour, linewidth=2, markersize=7,
                label=label)
        # annotate best k
        best_k = max(results, key=results.get)
        best_acc = results[best_k]
        ax.annotate(f'k={best_k}\n{best_acc:.3f}',
                    xy=(best_k, best_acc),
                    xytext=(best_k + 0.5, best_acc - 0.015),
                    fontsize=8, color=colour,
                    arrowprops=dict(arrowstyle='->', color=colour, lw=1.2))
    ax.set_xlabel('k (number of neighbours)', fontsize=12)
    ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_title(f'k-NN Accuracy vs k — {dataset_name}', fontsize=14,
                 fontweight='bold')
    ax.set_xticks(K_VALUES)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  PART IV — APPLICATION: DISTANCE METRIC COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def run_metric_comparison(X_train, X_test, y_train, y_test,
                          k_values=None, dataset_name="Dataset"):
    """
    Compare k-NN with different distance metrics across multiple k values.
    Returns nested dict  {metric_name: {k: accuracy}}.
    """
    if k_values is None:
        k_values = K_VALUES
    all_results = {}
    for mname, mparams in METRICS.items():
        results = {}
        for k in k_values:
            clf = KNeighborsClassifier(n_neighbors=k, **mparams)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            results[k] = acc
        all_results[mname] = results
    return all_results


def plot_metric_comparison(all_results, dataset_name="Dataset"):
    """Bar chart comparing metrics at each k."""
    metric_names = list(all_results.keys())
    k_values = sorted(list(all_results[metric_names[0]].keys()))
    n_metrics = len(metric_names)
    x = np.arange(len(k_values))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(12, 5))
    colours = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for i, mname in enumerate(metric_names):
        accs = [all_results[mname][k] for k in k_values]
        ax.bar(x + i * width, accs, width, label=mname, color=colours[i],
               alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_title(f'Distance Metric Comparison — {dataset_name}', fontsize=14,
                 fontweight='bold')
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  PART V — APPLICATION: UNIFORM vs DISTANCE-WEIGHTED
# ═══════════════════════════════════════════════════════════════════════════

def run_weighting_comparison(X_train, X_test, y_train, y_test,
                              k_values=None, dataset_name="Dataset"):
    """Compare uniform vs distance-weighted k-NN."""
    if k_values is None:
        k_values = K_VALUES
    results = {'Uniform': {}, 'Distance-weighted': {}}
    for k in k_values:
        for wt, label in [('uniform', 'Uniform'), ('distance', 'Distance-weighted')]:
            clf = KNeighborsClassifier(n_neighbors=k, weights=wt)
            clf.fit(X_train, y_train)
            results[label][k] = accuracy_score(y_test, clf.predict(X_test))
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PART VI — CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(X_train, X_test, y_train, y_test,
                          target_names, best_k=5, dataset_name="Dataset"):
    """Train on best_k, plot confusion matrix."""
    clf = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(target_names)), max(5, len(target_names) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=target_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(f'Confusion Matrix (k={best_k}, dist-weighted)\n'
                 f'{dataset_name} — Accuracy: {acc:.4f}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.show()

    print(f"\n  Classification Report  (k={best_k}, {dataset_name}):")
    print(classification_report(y_test, y_pred, target_names=[str(n) for n in target_names]))
    return acc


# ═══════════════════════════════════════════════════════════════════════════
#  PART VII — DECISION BOUNDARY VISUALISATION (2D — Iris only)
# ═══════════════════════════════════════════════════════════════════════════

def plot_decision_boundaries_iris():
    """
    Visualise 2D decision boundaries of k-NN on Iris dataset using
    petal length & petal width (features 2 and 3).
    Shows boundaries for k = 1, 3, 7, 15 side by side.
    """
    iris = load_iris()
    X = iris.data[:, 2:4]  # petal length, petal width
    y = iris.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    k_list = [1, 3, 7, 15]
    cmap_bg = ListedColormap(['#FFDDC1', '#C1E1FF', '#C1FFD7'])
    cmap_pts = ListedColormap(['#FF6F00', '#1565C0', '#2E7D32'])

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle('k-NN Decision Boundaries — Iris (Petal Length vs Petal Width)',
                 fontsize=14, fontweight='bold', y=1.02)

    h = 0.02  # mesh step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    for ax, k in zip(axes, k_list):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.6)
        ax.contour(xx, yy, Z, colors='grey', linewidths=0.5, alpha=0.5)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts,
                   edgecolors='k', s=30, linewidths=0.5)
        ax.set_title(f'k = {k}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Petal length (scaled)')
        ax.set_ylabel('Petal width (scaled)')

    fig.tight_layout()
    plt.show()

    print("""
  Observation:
  ● k=1 produces very complex, jagged boundaries that overfit noise.
  ● k=3 smooths out small anomalies; still captures local structure.
  ● k=7 gives a cleaner boundary with good generalisation.
  ● k=15 is very smooth — starts to under-fit subtle patterns.
""")


# ═══════════════════════════════════════════════════════════════════════════
#  PART VIII — FULL DATASET DEMO (callable from notebook)
# ═══════════════════════════════════════════════════════════════════════════

def demo_dataset(loader, name, test_size=0.25, random_state=42):
    """
    Run the complete k-NN analysis pipeline on a sklearn dataset.

    Steps:
      1. Load and describe
      2. Train–test split  +  feature scaling
      3. Accuracy vs k  (Euclidean, uniform)
      4. Distance-metric comparison
      5. Uniform vs distance-weighted comparison
      6. Confusion matrix for best k
    """
    print_section(f"DATASET DEMO — {name.upper()}", width=72)

    # 1. Load
    X, y, target_names, feature_names = load_and_describe(loader, name)

    # 2. Split & scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Accuracy vs k
    print("\n▶ 3a. Accuracy vs k (Euclidean, uniform weighting):")
    k_results = run_knn_k_analysis(X_train, X_test, y_train, y_test,
                                    dataset_name=name)
    for k, acc in sorted(k_results.items()):
        print(f"    k={k:>2d}  →  accuracy = {acc:.4f}")
    plot_accuracy_vs_k({f'{name} (Euclidean)': k_results}, dataset_name=name)

    # 4. Metric comparison
    print("\n▶ 3b. Distance metric comparison:")
    metric_results = run_metric_comparison(X_train, X_test, y_train, y_test,
                                            dataset_name=name)
    for mname, mres in metric_results.items():
        best_k = max(mres, key=mres.get)
        print(f"    {mname:<18s} best: k={best_k:>2d}  acc={mres[best_k]:.4f}")
    plot_metric_comparison(metric_results, dataset_name=name)

    # 5. Weighting comparison
    print("\n▶ 3c. Uniform vs Distance-weighted:")
    wt_results = run_weighting_comparison(X_train, X_test, y_train, y_test,
                                           dataset_name=name)
    for label, res in wt_results.items():
        best_k = max(res, key=res.get)
        print(f"    {label:<20s} best: k={best_k:>2d}  acc={res[best_k]:.4f}")
    plot_accuracy_vs_k(wt_results, dataset_name=name)

    # 6. Confusion matrix at best k
    best_k = max(k_results, key=k_results.get)
    print(f"\n▶ 3d. Confusion matrix at best k={best_k}:")
    plot_confusion_matrix(X_train, X_test, y_train, y_test,
                          target_names, best_k=best_k, dataset_name=name)


def demo_all_datasets():
    """Run demos for Iris, Wine, and Digits."""
    demo_dataset(load_iris,   "Iris")
    demo_dataset(load_wine,   "Wine")
    demo_dataset(load_digits, "Digits")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN — run when script is executed directly
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    display_all_theory()
    demo_all_datasets()
    plot_decision_boundaries_iris()
    print("\n✅  k-NN analysis complete.\n")
