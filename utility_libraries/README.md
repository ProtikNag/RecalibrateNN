# Bottleneck report: mathematical formulation

This document describes the quantities computed in `bottlenexck_report.py` (baseline) and `bottlenexck_report_after.py` (after). The core formulas are the same; the two scripts differ in **which sensitivity columns** are loaded and in **how the alignment matrix is stored** in the "after" script.

---

## 1. Data matrix

For each class, the pipeline builds a sample-by-layer matrix

$$S \in \mathbb{R}^{N \times d}$$

where:

- $N$ is the number of rows (samples / observations in the CSV for that class),
- $d$ is the number of layer-related columns after preprocessing,
- $S_{ij}$ is the sensitivity score for sample $i$ and layer (or concept column) $j$.

Layers are identified with column order: $j = 0, 1, \ldots, d-1$.

### Baseline (`bottlenexck_report.py`)

- Drops non-feature columns such as `Full filepath`.
- Removes columns whose names start with `sensitivityscore_After_`.
- Groups rows by `Full Class Index`.
- Strips the prefix `sensitivityscore_before_` from remaining sensitivity column names.

### After (`bottlenexck_report_after.py`)

- Drops `Full filepath`.
- Removes columns whose names start with `sensitivityscore_before_`.
- Keeps only columns whose names end with `_0.7` (threshold tag in the column name).
- Groups by `Full Class Index`.
- Strips the prefix `sensitivityscore_After_` from those column names.

So the **same formulas** below apply to $S$; only the construction of $S$ differs between the two entry points.

---

## 2. Standard Pearson correlation matrix (`df.corr()`)

Pandas `DataFrame.corr()` (default `method='pearson'`) builds a symmetric matrix $R \in \mathbb{R}^{d \times d}$ with entries

$$R_{jk} = \frac{\sum_{i=1}^{N}(S_{ij}-\bar{S}_j)(S_{ik}-\bar{S}_k)}{\sqrt{\sum_{i=1}^{N}(S_{ij}-\bar{S}_j)^2}\;\sqrt{\sum_{i=1}^{N}(S_{ik}-\bar{S}_k)^2}}$$

where $\bar{S}_j = \frac{1}{N}\sum_i S_{ij}$.

This matrix is used for heatmaps and for the "top subset" plot (Section 6). Excel export masks the **upper triangle** (including the diagonal) with `NaN` so only the lower triangle is written as distinct cells (duplicate symmetric information is hidden for display).

---

## 3. Gradient alignment matrix (explicit construction)

The function `compute_gradient_alignment` builds a matrix $A \in \mathbb{R}^{d \times d}$ from the **raw** (uncentered) columns of $S$.

Let $\mathbf{s}_j \in \mathbb{R}^N$ be column $j$ of $S$. Define

$$M_{jk} = \frac{1}{N}\sum_{i=1}^{N} S_{ij} S_{ik} = \frac{1}{N}\, \mathbf{s}_j^\top \mathbf{s}_k$$

$$\mu^{(2)}_j = \frac{1}{N}\sum_{i=1}^{N} S_{ij}^2 = \frac{1}{N}\, \mathbf{s}_j^\top \mathbf{s}_j$$

Then

$$A_{jk} = \frac{M_{jk}}{\sqrt{\mu^{(2)}_j\,\mu^{(2)}_k}}$$

Equivalently, $A_{jk}$ is the **cosine similarity** between the column vectors $\mathbf{s}_j$ and $\mathbf{s}_k$:

$$A_{jk} = \frac{\mathbf{s}_j^\top \mathbf{s}_k}{\|\mathbf{s}_j\|_2\,\|\mathbf{s}_k\|_2}$$

**Relation to Pearson correlation:** If each column of $S$ is **mean-centered**, then $A_{jk}$ coincides with the Pearson correlation $R_{jk}$, because the numerator becomes the sample covariance (up to the $N$ vs $N-1$ convention pandas may use) and the denominator becomes the product of sample standard deviations. Without centering, $A$ is generally **not** identical to `df.corr()`.

### Baseline vs after for $A$

- **Baseline:** `compute_gradient_alignment` returns the **full** matrix $A$ (all entries).
- **After:** The same $A$ is computed, then entries on and **above** the main diagonal are replaced by `NaN` (strict upper triangle + diagonal masked). The lower triangle still holds $A_{jk}$ for $j > k$ (row-major convention: kept where column index $<$ row index in the code's mask). CSPI and bottleneck detection use this possibly masked object; any statistic that averages only "row $i$, columns $> i$" should be checked against that masking (those positions are in the masked upper triangle).

---

## 4. Concept Sensitivity Propagation Index (CSPI)

Let the layers in column order be $L_0, \ldots, L_{d-1}$. For each row index $i \in \{0,\ldots,d-1\}$, define the **downstream** set of column indices $J_i = \{i+1, \ldots, d-1\}$.

The code defines

$$\text{CSPI}_i = \begin{cases} \displaystyle \frac{1}{|J_i|} \sum_{j \in J_i} \bigl|A_{ij}\bigr|, & J_i \neq \emptyset \\ 0, & J_i = \emptyset \end{cases}$$

Interpretation (when entries $A_{ij}$ for $j > i$ are available): $\text{CSPI}_i$ is the mean absolute alignment of layer $i$ with all **later** layers in the table ordering—treated as a summary of how strongly sensitivity at $i$ co-varies (in the cosine-similarity sense) with downstream layers.

Results are sorted by `CSPI` descending for reporting.

---

## 5. Bottleneck detection

Given a threshold $\tau$ (default $\tau = 0.2$ in code), a layer $L_i$ is flagged when:

- it has at least one downstream layer ($i < d-1$), and
- $\displaystyle \frac{1}{|J_i|} \sum_{j \in J_i} \bigl|A_{ij}\bigr| \ge \tau$.

So bottlenecks use the **same** mean-absolute-downstream statistic as CSPI, then filter by $\tau$. The output records the layer name and that mean value (`Mean_Alignment`).

---

## 6. Top correlation subset (for subset heatmap)

Using the Pearson matrix $R$:

1. Take the **last row** of $R$ (last column index in frame order).
2. Keep only **positive** correlations in that row; if there are at most one such entry, skip the subset plot.
3. Sort those remaining values descending and take the top `top_k` (default 10) column indices.
4. Extract the submatrix $R$ restricted to those rows and columns and plot it as a heatmap.

So the subset highlights variables most strongly and positively correlated with the **last** layer's sensitivity series, within the correlation—not alignment—matrix.

---

## 7. Pipeline summary

For each class:

1. Build $S$ (per-script column rules).
2. Compute $R = \text{Pearson}(S)$ and $A = \text{alignment}(S)$ (cosine similarity of columns).
3. Compute $\text{CSPI}$ and bottleneck flags from $A$ (mean $|\cdot|$ over downstream indices in row order).
4. Persist masked $R$ and $A$, CSPI table, heatmaps, and concatenate bottleneck rows into `bottleneck_report.csv`.

---

## 8. Reference: code-to-symbol map

| Symbol | Code (conceptual) |
|--------|-------------------|
| $S$ | `df.values` after group/prep |
| $R$ | `df.corr()` |
| $A$ | `compute_gradient_alignment(df)` |
| $A_{jk}$ cosine form | $(S^\top S / N)_{jk} / \sqrt{\mu^{(2)}_j \mu^{(2)}_k}$ |
| CSPI | `compute_cspi` |
| Bottlenecks | `detect_bottlenecks` with $\tau$ |

