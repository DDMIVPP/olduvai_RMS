"""
Requirements:
    pip install -r requirements.txt

Default input:
    olduvai_data.xlsx in the same folder as this script

Default output root:
    analysis/ subfolder in the same folder as this script

Subfolders created automatically:
    01_PCA
    02_MCA
    03_Logistic_AME
    04_Density_RawMaterial
    05_Density_WithinSite_ByCategory
    06_Density_AmongSites
    07_Shape_ChiSquare
    08_Dimension_CategoryStats
"""

from __future__ import annotations

import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from scipy.stats import (
    chi2,
    chi2_contingency,
    gaussian_kde,
    kruskal,
    mannwhitneyu,
    norm,
    rankdata,
    t,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_XLSX = BASE_DIR / "olduvai_data.xlsx"
OUTPUT_ROOT = BASE_DIR / "analysis"
SHEET_NAME = 0  # use 0 to read the first sheet; change if needed

SITES = ["HWK EE", "MNK Skull", "EF-HR"]
CATEGORIES = ["Core", "Hammerstone", "Unmodified"]
NUMERIC_VARS_PCA = ["density", "length", "width", "thickness", "weight"]
DIMENSION_VARS = ["length", "width", "thickness", "weight"]
ACTIVE_MCA_VARS = ["sphericity", "angularity", "raw material"]
RAW_MATERIAL_ORDER = ["Phonolite", "Trachyte", "Basalt", "Lava Indet"]
ANGULARITY_MAP = {
    "Angular": 1,
    "Subangular": 2,
    "Sub-Angular": 2,
    "Sub-rounded": 3,
    "Sub-Rounded": 3,
    "Rounded": 4,
    "Well-rounded": 5,
    "Well-Rounded": 5,
}

OUTDIRS = {
    "pca": OUTPUT_ROOT / "01_PCA",
    "mca": OUTPUT_ROOT / "02_MCA",
    "logit": OUTPUT_ROOT / "03_Logistic_AME",
    "dens_rm": OUTPUT_ROOT / "04_Density_RawMaterial",
    "dens_within": OUTPUT_ROOT / "05_Density_WithinSite_ByCategory",
    "dens_among": OUTPUT_ROOT / "06_Density_AmongSites",
    "shape": OUTPUT_ROOT / "07_Shape_ChiSquare",
    "dimension": OUTPUT_ROOT / "08_Dimension_CategoryStats",
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
def make_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for p in OUTDIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def find_col_exact(cols: Iterable[str], candidates_lower: Iterable[str]) -> str | None:
    mapping = {str(c).strip().lower(): c for c in cols}
    for key, original in mapping.items():
        if key in set(candidates_lower):
            return original
    return None


def find_col_contains(cols: Iterable[str], keyword: str) -> str | None:
    for c in cols:
        if keyword.lower() in str(c).lower():
            return c
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    mapping = {
        "site": find_col_exact(cols, ["site", "locality", "assemblage", "site_name"]),
        "category": find_col_exact(cols, ["category", "categoty", "type", "artifact_type"]),
        "density": find_col_contains(cols, "density"),
        "length": find_col_exact(cols, ["length", "l"]),
        "width": find_col_exact(cols, ["width", "w"]),
        "thickness": find_col_exact(cols, ["thickness", "t"]),
        "weight": find_col_exact(cols, ["weight", "mass"]),
        "raw material": find_col_exact(cols, ["raw material", "raw_material", "rawmaterial"]),
        "sphericity": find_col_exact(cols, ["sphericity"]),
        "angularity": find_col_exact(cols, ["angularity"]),
        "technological class": find_col_exact(cols, ["technological class", "technological_class"]),
    }

    if mapping["site"] is None:
        for c in cols:
            values = df[c].astype(str).str.upper()
            if values.str.contains("HWK|MNK|EF", regex=True, na=False).any():
                mapping["site"] = c
                break

    if mapping["category"] is None:
        for c in cols:
            values = df[c].astype(str).str.strip().str.lower()
            if values.isin(["core", "hammerstone", "unmodified"]).sum() >= 5:
                mapping["category"] = c
                break

    required = [
        "site", "category", "density", "length", "width", "thickness", "weight",
        "raw material", "sphericity", "angularity", "technological class"
    ]
    missing = [k for k in required if mapping[k] is None]
    if missing:
        raise ValueError(f"Could not detect required columns: {missing}\nDetected mapping: {mapping}")
    return mapping


def read_data(path: Path, sheet_name=0) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    colmap = detect_columns(df)
    return df, colmap


def canon_site(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip().upper().replace("–", "-").replace("—", "-")
    x = re.sub(r"\s+", " ", x)
    return x


def site_pattern(site: str) -> re.Pattern:
    tokens = re.split(r"[\s\-]+", canon_site(site))
    return re.compile(r"^" + r"[- ]*".join(map(re.escape, tokens)) + r"$")


def clean_category(x: str) -> str | float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == "core":
        return "Core"
    if s == "hammerstone":
        return "Hammerstone"
    if s == "unmodified":
        return "Unmodified"
    return str(x).strip()


def site_subset(df: pd.DataFrame, colmap: Dict[str, str], site: str) -> pd.DataFrame:
    mask = df[colmap["site"]].astype(str).map(canon_site).str.match(site_pattern(site))
    sub = df.loc[mask].copy()
    sub[colmap["category"]] = sub[colmap["category"]].apply(clean_category)
    return sub[sub[colmap["category"]].isin(CATEGORIES)].copy()


def format_p(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.3e}" if x < 0.001 else f"{x:.5f}"


def save_excel(sheets: Dict[str, pd.DataFrame], path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


# ---------------------------------------------------------------------
# Shared statistics helpers
# ---------------------------------------------------------------------
def add_conf_ellipse(ax, x, y, conf: float = 0.95) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    mean = np.array([x.mean(), y.mean()])
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    scale = np.sqrt(chi2.ppf(conf, df=2))
    width, height = 2 * scale * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(mean, width, height, angle=angle, alpha=0.18))
    ax.add_patch(Ellipse(mean, width, height, angle=angle, fill=False, linewidth=1.0))


def dunn_test_holm(df_in: pd.DataFrame, group_col: str, value_col: str, order: List[str]) -> pd.DataFrame:
    d = df_in[[group_col, value_col]].dropna().copy()
    values = d[value_col].to_numpy(dtype=float)
    d["rank"] = rankdata(values)
    N = len(d)
    _, tie_counts = np.unique(values, return_counts=True)
    tie_term = np.sum(tie_counts**3 - tie_counts)
    C = 1.0 - tie_term / (N**3 - N) if N > 1 else 1.0
    rank_sums = d.groupby(group_col)["rank"].sum().reindex(order)
    ns = d.groupby(group_col).size().reindex(order)

    rows = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            g1, g2 = order[i], order[j]
            n1, n2 = ns[g1], ns[g2]
            R1, R2 = rank_sums[g1], rank_sums[g2]
            z = (R1 / n1 - R2 / n2) / np.sqrt((N * (N + 1) / 12.0) * C * (1 / n1 + 1 / n2))
            p_raw = 2 * (1 - norm.cdf(abs(z)))
            rows.append({"Comparison": f"{g1} vs. {g2}", "n1": int(n1), "n2": int(n2), "Z": z, "p_raw": p_raw})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    m = len(out)
    idx = np.argsort(out["p_raw"].to_numpy())
    p_sorted = out["p_raw"].to_numpy()[idx]
    p_holm_sorted = np.maximum.accumulate((m - np.arange(m)) * p_sorted)
    p_holm_sorted = np.clip(p_holm_sorted, 0, 1)
    p_holm = np.empty(m)
    p_holm[idx] = p_holm_sorted
    out["p_holm"] = p_holm
    return out


def mean_ci95(x: pd.Series) -> Tuple[float, float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = x.mean()
    if n == 1:
        return mean, np.nan, np.nan
    sd = x.std(ddof=1)
    se = sd / math.sqrt(n)
    h = t.ppf(0.975, df=n - 1) * se
    return mean, mean - h, mean + h


def mannwhitney_with_z(x: pd.Series, y: pd.Series, label1: str, label2: str) -> Dict[str, float | str]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    U = res.statistic
    p = res.pvalue
    combined = np.concatenate([x, y])
    ranks = rankdata(combined)
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    N = n1 + n2
    mean_U = n1 * n2 / 2
    _, counts = np.unique(combined, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    var_U = (n1 * n2 / 12) * ((N + 1) - tie_term / (N * (N - 1)))
    Z = (U1 - mean_U - 0.5) / np.sqrt(var_U) if U1 > mean_U else (U1 - mean_U + 0.5) / np.sqrt(var_U)
    return {"Comparison": f"{label1} vs. {label2}", "n1": n1, "n2": n2, "U": U, "Z": Z, "p": p}


# ---------------------------------------------------------------------
# 01 PCA
# ---------------------------------------------------------------------
def run_pca(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["pca"]
    workbook = outdir / "PCA_results_3sites.xlsx"
    sheets = {}

    for site in SITES:
        sub = site_subset(df, colmap, site)
        use_cols = [colmap[v] for v in ["category", *NUMERIC_VARS_PCA]]
        sub = sub[use_cols].copy()
        rename_map = {colmap["category"]: "Category", **{colmap[v]: v for v in NUMERIC_VARS_PCA}}
        sub = sub.rename(columns=rename_map)
        for v in NUMERIC_VARS_PCA:
            sub[v] = pd.to_numeric(sub[v], errors="coerce")
        sub = sub.dropna(subset=NUMERIC_VARS_PCA)
        if sub.empty:
            continue

        Xz = StandardScaler().fit_transform(sub[NUMERIC_VARS_PCA])
        pca = PCA(n_components=len(NUMERIC_VARS_PCA), svd_solver="full")
        scores = pca.fit_transform(Xz)
        eig = pca.explained_variance_
        ratio = pca.explained_variance_ratio_
        loadings = pca.components_.T * np.sqrt(eig)

        explained = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(len(NUMERIC_VARS_PCA))],
            "Eigenvalue": eig,
            "Explained_%": ratio * 100,
            "Cumulative_%": np.cumsum(ratio) * 100,
        })
        loading_df = pd.DataFrame(loadings, index=NUMERIC_VARS_PCA, columns=[f"PC{i+1}" for i in range(len(NUMERIC_VARS_PCA))]).reset_index().rename(columns={"index": "variable"})
        score_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(len(NUMERIC_VARS_PCA))])
        score_df.insert(0, "Category", sub["Category"].values)

        tag = canon_site(site).replace(" ", "_").replace("-", "_")
        sheets[f"{tag}_explained"] = explained
        sheets[f"{tag}_loadings"] = loading_df
        sheets[f"{tag}_scores"] = score_df

        # Loadings plot
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for v in NUMERIC_VARS_PCA:
            x = loading_df.loc[loading_df["variable"] == v, "PC1"].iloc[0]
            y = loading_df.loc[loading_df["variable"] == v, "PC2"].iloc[0]
            ax.arrow(0, 0, x, y, head_width=0.05, length_includes_head=True)
            ax.text(x * 1.05, y * 1.05, v, fontsize=9)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
        ax.set_title(f"{site} PCA Loadings")
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(outdir / f"{tag}_PCA_loadings.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Scores + ellipses
        fig, ax = plt.subplots(figsize=(6, 4.5))
        markers = {"Core": "o", "Hammerstone": "^", "Unmodified": "s"}
        for cat in CATEGORIES:
            m = score_df["Category"] == cat
            ax.scatter(score_df.loc[m, "PC1"], score_df.loc[m, "PC2"], s=18, marker=markers[cat], label=cat)
            add_conf_ellipse(ax, score_df.loc[m, "PC1"], score_df.loc[m, "PC2"])
        ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
        ax.set_title(f"{site} PCA Scores")
        ax.legend(frameon=True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(outdir / f"{tag}_PCA_scores_ellipses.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Biplot
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for cat in CATEGORIES:
            m = score_df["Category"] == cat
            ax.scatter(score_df.loc[m, "PC1"], score_df.loc[m, "PC2"], s=16, marker=markers[cat], label=cat)
        scale = 0.35 * min(score_df["PC1"].max() - score_df["PC1"].min(), score_df["PC2"].max() - score_df["PC2"].min())
        for v in NUMERIC_VARS_PCA:
            x = loading_df.loc[loading_df["variable"] == v, "PC1"].iloc[0] * scale
            y = loading_df.loc[loading_df["variable"] == v, "PC2"].iloc[0] * scale
            ax.arrow(0, 0, x, y, head_width=0.08, length_includes_head=True)
            ax.text(x * 1.05, y * 1.05, v, fontsize=9)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
        ax.set_title(f"{site} PCA Biplot")
        ax.legend(frameon=True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(outdir / f"{tag}_PCA_biplot.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if sheets:
        save_excel(sheets, workbook)


# ---------------------------------------------------------------------
# 02 MCA
# ---------------------------------------------------------------------
def prep_mca_main(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    d = df.copy()
    for v in ACTIVE_MCA_VARS + [category_col]:
        d[v] = d[v].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
    for v in ACTIVE_MCA_VARS:
        for cat, g in d.groupby(category_col):
            m = (d[category_col] == cat) & d[v].isna()
            mode = g[v].dropna().mode()
            if m.any() and len(mode):
                d.loc[m, v] = mode.iloc[0]
        if d[v].isna().any():
            mode = d[v].dropna().mode()
            if len(mode):
                d.loc[d[v].isna(), v] = mode.iloc[0]
    return d


def prep_mca_sensitivity(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    d = df.copy()
    for v in ACTIVE_MCA_VARS + [category_col]:
        d[v] = d[v].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
    for v in ACTIVE_MCA_VARS:
        d[v] = d[v].fillna("S.Def")
    return d


def disjunctive_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([pd.get_dummies(df[v].astype("category"), prefix=v) for v in ACTIVE_MCA_VARS], axis=1).astype(float)


def run_mca_math(X: pd.DataFrame, k_vars: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    N = n * k_vars
    r = np.full(n, 1 / n)
    c = X.sum(0).values / N
    P = X.values / N
    S = np.diag(1 / np.sqrt(r)) @ (P - np.outer(r, c)) @ np.diag(1 / np.sqrt(c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    F = np.diag(1 / np.sqrt(r)) @ U @ np.diag(s)
    G = np.diag(1 / np.sqrt(c)) @ Vt.T @ np.diag(s)
    eig = s**2
    ctr = (c[:, None] * G**2) / eig[None, :] * 100
    cos2 = (G**2) / (G**2).sum(1, keepdims=True)
    return eig, F, G, ctr, cos2


def mca_eigen_table(eig: np.ndarray, k_vars: int) -> pd.DataFrame:
    adj = (k_vars / (k_vars - 1)) * np.maximum(0, eig - 1 / k_vars)
    out = pd.DataFrame({
        "dim": [f"Dim{i+1}" for i in range(len(eig))],
        "eigenvalue": eig,
        "var_%": eig / eig.sum() * 100,
        "adj_linear": adj,
    })
    out["adj_linear_var_%"] = adj / adj.sum() * 100 if adj.sum() > 0 else 0
    return out


def cramers_v_table(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    rows = []
    for v in ACTIVE_MCA_VARS:
        tab = pd.crosstab(df[category_col], df[v])
        chi2_stat, p, dof, _ = chi2_contingency(tab, correction=False)
        n = tab.values.sum()
        V = math.sqrt(chi2_stat / (n * (min(tab.shape) - 1)))
        rows.append({"variable": v, "chi2": chi2_stat, "df": dof, "p_value": p, "cramers_v": V})
    return pd.DataFrame(rows)


def repel_text(ax, texts, iterations: int = 200) -> None:
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    for _ in range(iterations):
        moved = 0
        boxes = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.15) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if boxes[i].overlaps(boxes[j]):
                    ci = ((boxes[i].x0 + boxes[i].x1) / 2, (boxes[i].y0 + boxes[i].y1) / 2)
                    cj = ((boxes[j].x0 + boxes[j].x1) / 2, (boxes[j].y0 + boxes[j].y1) / 2)
                    dx, dy = ci[0] - cj[0], ci[1] - cj[1]
                    if dx == 0 and dy == 0:
                        dx = 1
                    norm_xy = (dx * dx + dy * dy) ** 0.5
                    px, py = 2 * dx / norm_xy, 2 * dy / norm_xy
                    inv = ax.transData.inverted()
                    for t_obj, s in [(texts[i], 1), (texts[j], -1)]:
                        x, y = t_obj.get_position()
                        d = ax.transData.transform((x, y))
                        x2, y2 = inv.transform((d[0] + s * px, d[1] + s * py))
                        t_obj.set_position((x2, y2))
                        moved += abs(x2 - x) + abs(y2 - y)
        if moved < 1e-3:
            break


def run_mca(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["mca"]
    summary_rows = []

    for site in SITES:
        sub = site_subset(df, colmap, site)
        use = sub[[colmap["category"], colmap["sphericity"], colmap["angularity"], colmap["raw material"]]].copy()
        use.columns = ["category", "sphericity", "angularity", "raw material"]
        site_dir = outdir / canon_site(site).replace(" ", "_").replace("-", "_")
        site_dir.mkdir(exist_ok=True)
        use.to_csv(site_dir / "subset_used.csv", index=False)
        cramers_v_table(use, "category").to_csv(site_dir / "cramersV_category_vs_vars.csv", index=False)

        for workflow_name, prepared in {
            "main": prep_mca_main(use, "category"),
            "sensitivity": prep_mca_sensitivity(use, "category"),
        }.items():
            wdir = site_dir / workflow_name
            wdir.mkdir(exist_ok=True)
            X = disjunctive_table(prepared)
            eig, F, G, ctr, cos2 = run_mca_math(X, len(ACTIVE_MCA_VARS))
            et = mca_eigen_table(eig, len(ACTIVE_MCA_VARS))
            labels = X.columns.tolist()

            pd.DataFrame({"modality": labels, **{f"Dim{i+1}": ctr[:, i] for i in range(ctr.shape[1])}}).to_csv(wdir / "modality_contrib.csv", index=False)
            pd.DataFrame({"modality": labels, **{f"Dim{i+1}": cos2[:, i] for i in range(cos2.shape[1])}}).to_csv(wdir / "modality_cos2.csv", index=False)
            et.to_csv(wdir / "eigenvalues.csv", index=False)

            centroids = pd.DataFrame(F[:, :2], columns=["Dim1", "Dim2"]).join(prepared[["category"]].reset_index(drop=True)).groupby("category")[["Dim1", "Dim2"]].mean().reset_index()
            centroids.to_csv(wdir / "type_centroids.csv", index=False)
            coords = pd.DataFrame(G[:, :2], columns=["Dim1", "Dim2"])
            coords["modality"] = labels

            fig, ax = plt.subplots(figsize=(7.5, 6.2))
            ax.scatter(coords["Dim1"], coords["Dim2"], s=18)
            texts = [ax.text(r.Dim1, r.Dim2, r.modality, fontsize=8) for _, r in coords.iterrows()]
            ax.scatter(centroids["Dim1"], centroids["Dim2"], marker="D", s=30)
            for _, r in centroids.iterrows():
                ax.text(r.Dim1, r.Dim2, r["category"], fontsize=9)
            repel_text(ax, texts)
            ax.set_xlabel(f"Dim 1 ({et.loc[0, 'var_%']:.1f}%)")
            ax.set_ylabel(f"Dim 2 ({et.loc[1, 'var_%']:.1f}%)")
            ax.set_title(f"MCA — {site} ({workflow_name})")
            fig.tight_layout()
            fig.savefig(wdir / "biplot.png", dpi=300)
            fig.savefig(wdir / "biplot.pdf")
            plt.close(fig)

            summary_rows.append({
                "site": site,
                "workflow": workflow_name,
                "n_rows": len(prepared),
                "Dim1_raw_%": et.loc[0, "var_%"],
                "Dim2_raw_%": et.loc[1, "var_%"],
                "Dim1_adj_%": et.loc[0, "adj_linear_var_%"],
                "Dim2_adj_%": et.loc[1, "adj_linear_var_%"],
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(outdir / "MCA_summary.csv", index=False)


# ---------------------------------------------------------------------
# 03 Logistic regression + AME
# ---------------------------------------------------------------------
def add_size_pc1(site_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    d = site_df.copy()
    size_vars = ["length", "width", "thickness"]
    valid = d[size_vars].dropna().index
    d["pc1"] = np.nan
    if len(valid) < 2:
        return d, None
    X = StandardScaler().fit_transform(d.loc[valid, size_vars])
    pca = PCA(n_components=1)
    d.loc[valid, "pc1"] = pca.fit_transform(X)[:, 0]
    pca_info = pd.DataFrame({
        "variable": size_vars,
        "loading_pc1": pca.components_[0],
        "explained_variance_ratio": pca.explained_variance_ratio_[0],
    })
    return d, pca_info


def fit_logit(data: pd.DataFrame, target: str, predictors: List[str], categorical: Dict[str, List[str]] | None = None):
    d = data[data["category"].isin([target, "Unmodified"])].copy()
    d["y"] = (d["category"] == target).astype(int)
    d = d[["y"] + predictors].dropna().copy()
    if d.empty or d["y"].nunique() < 2:
        return None, None, None, None
    X = d[predictors].copy()
    if categorical:
        for col, levels in categorical.items():
            X[col] = pd.Categorical(X[col], categories=levels)
        X = pd.get_dummies(X, columns=list(categorical.keys()), drop_first=True)
    X = sm.add_constant(X, has_constant="add").astype(float)
    y = d["y"].astype(int)
    try:
        model = sm.Logit(y, X).fit(disp=False)
    except Exception:
        return d, X, None, None

    ci = model.conf_int()
    ci.columns = ["ci_low", "ci_high"]
    res = pd.DataFrame({
        "predictor": model.params.index,
        "coef": model.params.values,
        "p_value": model.pvalues.values,
        "OR": np.exp(model.params.values),
        "OR_95CI_low": np.exp(ci["ci_low"].values),
        "OR_95CI_high": np.exp(ci["ci_high"].values),
    })
    try:
        ame = model.get_margeff(at="overall").summary_frame().reset_index().rename(columns={"index": "predictor", "dy/dx": "AME", "Pr(>|z|)": "AME_p_value"})
        res = res.merge(ame[["predictor", "AME", "AME_p_value"]], on="predictor", how="left")
    except Exception:
        res["AME"] = np.nan
        res["AME_p_value"] = np.nan
    return d, X, model, res


def clean_predictor_names(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    out["predictor"] = out["predictor"].replace({
        "const": "Intercept",
        "density": "Density",
        "pc1": "PC1",
        "angularity_ord": "Angularity (1=Angular ... 5=Well-rounded)",
        "sphericity_High": "Sphericity: High vs Low",
    })
    return out


def plot_predictions(site: str, d_h, m_h, d_c, m_c, outfile: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    if d_h is not None and m_h is not None:
        grid_h = pd.DataFrame({
            "const": 1.0,
            "density": np.linspace(d_h["density"].min(), d_h["density"].max(), 200),
            "pc1": d_h["pc1"].mean(),
            "angularity_ord": d_h["angularity_ord"].mean(),
            "sphericity_High": 0.0,
        })
        axes[0].scatter(d_h["density"], d_h["y"], s=18, alpha=0.55)
        axes[0].plot(grid_h["density"], m_h.predict(grid_h), lw=2)
        axes[0].set_title(site)
        axes[0].set_xlabel("Density (g/cm³)")
        axes[0].set_ylabel("Probability of being hammerstone")
    else:
        axes[0].text(0.5, 0.5, "Model unavailable", ha="center", va="center")
        axes[0].set_axis_off()

    if d_c is not None and m_c is not None:
        grid_c = pd.DataFrame({
            "const": 1.0,
            "density": np.linspace(d_c["density"].min(), d_c["density"].max(), 200),
            "pc1": d_c["pc1"].mean(),
        })
        axes[1].scatter(d_c["density"], d_c["y"], s=18, alpha=0.55)
        axes[1].plot(grid_c["density"], m_c.predict(grid_c), lw=2)
        axes[1].set_title(site)
        axes[1].set_xlabel("Density (g/cm³)")
        axes[1].set_ylabel("Probability of being core")
    else:
        axes[1].text(0.5, 0.5, "Model unavailable", ha="center", va="center")
        axes[1].set_axis_off()

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_logistic(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["logit"]
    all_results, all_pca = [], []

    base = df[[colmap[k] for k in ["site", "category", "density", "length", "width", "thickness", "sphericity", "angularity"]]].copy()
    base.columns = ["site", "category", "density", "length", "width", "thickness", "sphericity", "angularity"]
    base["category"] = base["category"].apply(clean_category)
    for v in ["density", "length", "width", "thickness"]:
        base[v] = pd.to_numeric(base[v], errors="coerce")
    base = base.replace({"NA": np.nan, "nan": np.nan, "": np.nan})

    for site in SITES:
        site_df = base[base["site"].astype(str).map(canon_site).str.match(site_pattern(site))].copy()
        site_df, pca_info = add_size_pc1(site_df)
        site_df["angularity_ord"] = site_df["angularity"].map(ANGULARITY_MAP)

        if pca_info is not None:
            pca_info.insert(0, "site", site)
            all_pca.append(pca_info)

        d_h, X_h, m_h, res_h = fit_logit(site_df, "Hammerstone", ["density", "pc1", "sphericity", "angularity_ord"], categorical={"sphericity": ["Low", "High"]})
        res_h = clean_predictor_names(res_h)
        if res_h is not None:
            res_h.insert(0, "site", site)
            res_h.insert(1, "model", "Hammerstone vs Unmodified")
            all_results.append(res_h)

        d_c, X_c, m_c, res_c = fit_logit(site_df, "Core", ["density", "pc1"])
        res_c = clean_predictor_names(res_c)
        if res_c is not None:
            res_c.insert(0, "site", site)
            res_c.insert(1, "model", "Core vs Unmodified")
            all_results.append(res_c)

        tag = canon_site(site).replace(" ", "_").replace("-", "_")
        plot_predictions(site, d_h, m_h, d_c, m_c, outdir / f"{tag}_probability_plot.png")

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    pca_df = pd.concat(all_pca, ignore_index=True) if all_pca else pd.DataFrame()

    if not results_df.empty:
        results_df["OR (95% CI)"] = (
            results_df["OR"].round(3).astype(str) + " (" +
            results_df["OR_95CI_low"].round(3).astype(str) + "–" +
            results_df["OR_95CI_high"].round(3).astype(str) + ")"
        )

    save_excel({
        "Logistic_results": results_df,
        "PCA_loadings": pca_df,
    }, outdir / "Olduvai_logistic_results_all_sites.xlsx")


# ---------------------------------------------------------------------
# 04 Density by raw material
# ---------------------------------------------------------------------
def run_density_raw_material(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["dens_rm"]
    data = df[[colmap["raw material"], colmap["density"]]].copy()
    data.columns = ["raw material", "density"]
    data["raw material"] = data["raw material"].astype(str).str.strip()
    data["density"] = pd.to_numeric(data["density"], errors="coerce")
    data = data.dropna(subset=["density"]).copy()
    data = data[data["raw material"].isin(RAW_MATERIAL_ORDER)].copy()

    desc = data.groupby("raw material")["density"].agg(N="count", Mean="mean", SD="std", Min="min", Max="max").reindex(RAW_MATERIAL_ORDER).reset_index()
    all_row = pd.DataFrame([{
        "raw material": "All samples",
        "N": data["density"].count(),
        "Mean": data["density"].mean(),
        "SD": data["density"].std(),
        "Min": data["density"].min(),
        "Max": data["density"].max(),
    }])
    desc = pd.concat([desc, all_row], ignore_index=True)

    comparisons = [("Phonolite", "Basalt"), ("Phonolite", "Trachyte"), ("Basalt", "Trachyte")]
    mw = pd.DataFrame([
        mannwhitney_with_z(
            data.loc[data["raw material"] == a, "density"],
            data.loc[data["raw material"] == b, "density"],
            a, b,
        )
        for a, b in comparisons
    ])

    desc.loc[:, ["Mean", "SD", "Min", "Max"]] = desc[["Mean", "SD", "Min", "Max"]].round(3)
    mw["U"] = mw["U"].round(3)
    mw["Z"] = mw["Z"].round(3)
    mw["p_numeric"] = mw["p"]
    mw["p"] = mw["p"].apply(format_p)

    save_excel({"Descriptive_stats": desc, "Mann_Whitney": mw}, outdir / "density_raw_material_statistics.xlsx")


# ---------------------------------------------------------------------
# 05 Density within each site by category
# ---------------------------------------------------------------------
def run_density_within_site(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["dens_within"]
    data = df[[colmap["site"], colmap["category"], colmap["density"]]].copy()
    data.columns = ["site", "category", "density"]
    data["site"] = data["site"].astype(str).str.strip()
    data["category"] = data["category"].apply(clean_category)
    data["density"] = pd.to_numeric(data["density"], errors="coerce")
    data = data[data["category"].isin(CATEGORIES)].dropna(subset=["density"]).copy()

    desc_list, kw_list, dunn_list = [], [], []

    for site in SITES:
        sub = data[data["site"].astype(str).map(canon_site).str.match(site_pattern(site))].copy()
        if sub.empty:
            continue

        desc = sub.groupby("category")["density"].agg(N="count", Mean="mean", SD="std", Min="min", Max="max").reindex(CATEGORIES).reset_index()
        desc.insert(0, "Site", site)
        desc_list.append(desc)

        groups = [sub.loc[sub["category"] == c, "density"].to_numpy() for c in CATEGORIES]
        H, p = kruskal(*groups)
        kw_list.append(pd.DataFrame({"Site": [site], "H": [H], "df": [len(CATEGORIES) - 1], "p": [p]}))

        dunn = dunn_test_holm(sub, "category", "density", CATEGORIES)
        dunn.insert(0, "Site", site)
        dunn_list.append(dunn)

        fig, ax = plt.subplots(figsize=(8, 5.2))
        x_all = sub["density"].to_numpy()
        x_grid = np.linspace(x_all.min() - 0.05, x_all.max() + 0.05, 400)
        for cat in CATEGORIES:
            vals = sub.loc[sub["category"] == cat, "density"].to_numpy()
            ax.hist(vals, bins=10, density=True, alpha=0.28, label=cat)
            if len(vals) > 1 and np.std(vals, ddof=1) > 0:
                ax.plot(x_grid, gaussian_kde(vals)(x_grid), linewidth=1.5)
        ax.set_title(site)
        ax.set_xlabel("Density")
        ax.set_ylabel("Density Estimate")
        ax.legend(frameon=True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(outdir / f"{canon_site(site).replace(' ', '_').replace('-', '_')}_density_distribution.png", dpi=300)
        plt.close(fig)

    desc_all = pd.concat(desc_list, ignore_index=True) if desc_list else pd.DataFrame()
    kw_all = pd.concat(kw_list, ignore_index=True) if kw_list else pd.DataFrame()
    dunn_all = pd.concat(dunn_list, ignore_index=True) if dunn_list else pd.DataFrame()

    if not desc_all.empty:
        desc_all.loc[:, ["Mean", "SD", "Min", "Max"]] = desc_all[["Mean", "SD", "Min", "Max"]].round(3)
    if not kw_all.empty:
        kw_all["H"] = kw_all["H"].round(3)
        kw_all["p_numeric"] = kw_all["p"]
        kw_all["p"] = kw_all["p"].apply(format_p)
    if not dunn_all.empty:
        dunn_all["Z"] = dunn_all["Z"].round(3)
        dunn_all["p_raw_numeric"] = dunn_all["p_raw"]
        dunn_all["p_holm_numeric"] = dunn_all["p_holm"]
        dunn_all["p_raw"] = dunn_all["p_raw"].apply(format_p)
        dunn_all["p_holm"] = dunn_all["p_holm"].apply(format_p)

    save_excel({
        "Descriptive_stats": desc_all,
        "Kruskal_Wallis": kw_all,
        "Dunn_test": dunn_all,
    }, outdir / "three_sites_density_kruskal_dunn.xlsx")


# ---------------------------------------------------------------------
# 06 Density among sites
# ---------------------------------------------------------------------
def run_density_among_sites(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["dens_among"]
    data = df[[colmap["site"], colmap["category"], colmap["density"]]].copy()
    data.columns = ["site", "category", "density"]
    data["site"] = data["site"].astype(str).str.strip()
    data["category"] = data["category"].apply(clean_category)
    data["density"] = pd.to_numeric(data["density"], errors="coerce")
    data = data[data["category"].isin(CATEGORIES)].dropna(subset=["density"]).copy()

    # Overall among sites
    overall_desc = []
    for site in SITES:
        vals = data.loc[data["site"].astype(str).map(canon_site).str.match(site_pattern(site)), "density"]
        overall_desc.append({"site": site, "N": vals.count(), "Mean": vals.mean(), "SD": vals.std(), "Min": vals.min(), "Max": vals.max()})
    desc_overall = pd.DataFrame(overall_desc)
    groups_overall = [data.loc[data["site"].astype(str).map(canon_site).str.match(site_pattern(s)), "density"].to_numpy() for s in SITES]
    H_overall, p_overall = kruskal(*groups_overall)
    kw_overall = pd.DataFrame({"Test": ["Kruskal-Wallis"], "Sites compared": ["HWK EE vs. MNK Skull vs. EF-HR"], "H": [H_overall], "df": [2], "p": [p_overall]})

    tagged = data.copy()

    def standardize_site(value: str):
        v = canon_site(value)
        for s in SITES:
            if site_pattern(s).match(v):
                return s
        return pd.NA

    tagged["site_std"] = tagged["site"].map(standardize_site).astype("object")
    dunn_overall = dunn_test_holm(tagged.dropna(subset=["site_std"]), "site_std", "density", SITES)

    # By category
    desc_by_cat, kw_by_cat, dunn_by_cat = [], [], []
    for cat in CATEGORIES:
        sub = tagged[tagged["category"] == cat].dropna(subset=["site_std"]).copy()
        desc_rows = []
        for site in SITES:
            vals = sub.loc[sub["site_std"] == site, "density"]
            desc_rows.append({"Category": cat, "site": site, "N": vals.count(), "Mean": vals.mean(), "SD": vals.std(), "Min": vals.min(), "Max": vals.max()})
        desc_by_cat.append(pd.DataFrame(desc_rows))
        groups = [sub.loc[sub["site_std"] == s, "density"].to_numpy() for s in SITES]
        H, p = kruskal(*groups)
        kw_by_cat.append(pd.DataFrame({"Category": [cat], "Sites compared": ["HWK EE vs. MNK Skull vs. EF-HR"], "H": [H], "df": [2], "p": [p]}))
        dunn = dunn_test_holm(sub, "site_std", "density", SITES)
        dunn.insert(0, "Category", cat)
        dunn_by_cat.append(dunn)

    desc_by_cat_df = pd.concat(desc_by_cat, ignore_index=True)
    kw_by_cat_df = pd.concat(kw_by_cat, ignore_index=True)
    dunn_by_cat_df = pd.concat(dunn_by_cat, ignore_index=True)

    for tbl in [desc_overall, desc_by_cat_df]:
        tbl.loc[:, ["Mean", "SD", "Min", "Max"]] = tbl[["Mean", "SD", "Min", "Max"]].round(3)
    for tbl in [kw_overall, kw_by_cat_df]:
        tbl["H"] = tbl["H"].round(3)
        tbl["p_numeric"] = tbl["p"]
        tbl["p"] = tbl["p"].apply(format_p)
    for tbl in [dunn_overall, dunn_by_cat_df]:
        tbl["Z"] = tbl["Z"].round(3)
        tbl["p_raw_numeric"] = tbl["p_raw"]
        tbl["p_holm_numeric"] = tbl["p_holm"]
        tbl["p_raw"] = tbl["p_raw"].apply(format_p)
        tbl["p_holm"] = tbl["p_holm"].apply(format_p)

    # Plot overall boxplot
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.boxplot([tagged.loc[tagged["site_std"] == s, "density"].to_numpy() for s in SITES], labels=SITES, widths=0.35)
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "three_sites_density_boxplot.png", dpi=300)
    plt.close(fig)

    # Plot grouped boxplot
    fig, ax = plt.subplots(figsize=(10, 5.8))
    base_positions, offsets, width = [1, 3, 5], [-0.28, 0.0, 0.28], 0.22
    for j, site in enumerate(SITES):
        site_data = [tagged.loc[(tagged["category"] == cat) & (tagged["site_std"] == site), "density"].to_numpy() for cat in CATEGORIES]
        positions = [base_positions[i] + offsets[j] for i in range(len(CATEGORIES))]
        ax.boxplot(site_data, positions=positions, widths=width, patch_artist=False, manage_ticks=False)
    for site in SITES:
        ax.plot([], [], label=site)
    ax.set_xticks(base_positions)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Density")
    ax.legend(frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "three_sites_by_category_density_boxplot.png", dpi=300)
    plt.close(fig)

    save_excel({
        "Descriptive_stats": desc_overall,
        "Kruskal_Wallis": kw_overall,
        "Dunn_test": dunn_overall,
    }, outdir / "three_sites_density_stats_kruskal_dunn.xlsx")
    save_excel({
        "Descriptive_stats": desc_by_cat_df,
        "Kruskal_Wallis": kw_by_cat_df,
        "Dunn_test": dunn_by_cat_df,
    }, outdir / "three_sites_by_category_density_stats_kruskal_dunn.xlsx")


# ---------------------------------------------------------------------
# 07 Shape analysis: angularity / sphericity chi-square
# ---------------------------------------------------------------------
def count_and_chisq(data: pd.DataFrame, feature_col: str, site_col: str, category_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_tables, contingency_tables, chisq_results = [], [], []
    for site, dsite in data.groupby(site_col):
        d = dsite[[category_col, feature_col]].dropna().copy()
        if d.empty:
            continue
        counts = d.groupby([category_col, feature_col]).size().reset_index(name="count")
        counts.insert(0, "site", site)
        count_tables.append(counts)
        cont = pd.crosstab(d[category_col], d[feature_col])
        for cat in ["hammerstone", "unmodified"]:
            if cat not in cont.index:
                cont.loc[cat] = 0
        cont = cont.loc[["hammerstone", "unmodified"]]
        cont_out = cont.copy()
        cont_out.insert(0, "site", site)
        cont_out.insert(1, "category", cont_out.index)
        contingency_tables.append(cont_out.reset_index(drop=True))
        if cont.shape[1] >= 2 and cont.values.sum() > 0:
            chi2_stat, p, dof, _ = chi2_contingency(cont)
            chisq_results.append({"site": site, "variable": feature_col, "chi2": chi2_stat, "df": dof, "p_value": p, "n": int(cont.values.sum())})
    return (
        pd.concat(count_tables, ignore_index=True) if count_tables else pd.DataFrame(),
        pd.concat(contingency_tables, ignore_index=True) if contingency_tables else pd.DataFrame(),
        pd.DataFrame(chisq_results),
    )


def run_shape_analysis(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["shape"]
    sub = df.copy()
    site_col, category_col = colmap["site"], colmap["category"]
    tech_col = colmap["technological class"]
    ang_col, sph_col = colmap["angularity"], colmap["sphericity"]

    sub[category_col] = sub[category_col].astype(str).str.strip().str.lower()
    sub[tech_col] = sub[tech_col].astype(str).str.strip()
    sub = sub[sub[category_col].isin(["hammerstone", "unmodified"])].copy()
    sub = sub[~sub[tech_col].str.contains("FragHamSt", case=False, na=False)].copy()

    ang_counts, ang_cont, ang_chisq = count_and_chisq(sub, ang_col, site_col, category_col)
    sph_counts, sph_cont, sph_chisq = count_and_chisq(sub, sph_col, site_col, category_col)
    chisq_df = pd.concat([ang_chisq, sph_chisq], ignore_index=True)

    save_excel({
        "Chi_square_results": chisq_df,
        "angularity_counts": ang_counts,
        "sphericity_counts": sph_counts,
        "angularity_contingency": ang_cont,
        "sphericity_contingency": sph_cont,
    }, outdir / "hammerstone_unmodified_angularity_sphericity_chisquare.xlsx")


# ---------------------------------------------------------------------
# 08 Dimension/category statistics
# ---------------------------------------------------------------------
def descriptive_stats_site(df_site: pd.DataFrame) -> pd.DataFrame:
    rows = []
    site_name = df_site["site"].iloc[0]
    for cat in CATEGORIES:
        sub = df_site[df_site["category"] == cat]
        for var in DIMENSION_VARS:
            x = pd.to_numeric(sub[var], errors="coerce").dropna()
            mean, ci_low, ci_high = mean_ci95(x)
            rows.append({
                "Site": site_name,
                "Category": cat,
                "Variable": var,
                "N": len(x),
                "Min": x.min() if len(x) else np.nan,
                "Max": x.max() if len(x) else np.nan,
                "Mean": mean,
                "SD": x.std(ddof=1) if len(x) > 1 else np.nan,
                "CI95_low": ci_low,
                "CI95_high": ci_high,
            })
    return pd.DataFrame(rows)


def mann_whitney_tests_site(df_site: pd.DataFrame) -> pd.DataFrame:
    rows = []
    site_name = df_site["site"].iloc[0]
    for g1, g2 in [("Hammerstone", "Core"), ("Unmodified", "Core")]:
        d1 = df_site[df_site["category"] == g1]
        d2 = df_site[df_site["category"] == g2]
        for var in ["length", "weight"]:
            x1 = pd.to_numeric(d1[var], errors="coerce").dropna()
            x2 = pd.to_numeric(d2[var], errors="coerce").dropna()
            if len(x1) == 0 or len(x2) == 0:
                u_stat, p_val = np.nan, np.nan
            else:
                u_stat, p_val = mannwhitneyu(x1, x2, alternative="two-sided")
            rows.append({
                "Site": site_name,
                "Comparison": f"{g1} vs {g2}",
                "Variable": var,
                "N_group1": len(x1),
                "N_group2": len(x2),
                "U": u_stat,
                "p_value": p_val,
            })
    return pd.DataFrame(rows)


def make_boxplot(df_site: pd.DataFrame, out_png: Path | None = None):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    fig.patch.set_facecolor("#eeeeee")
    colors = ["#a9c4d6", "#b7d989", "#e6a197"]
    units = {"length": "(mm)", "width": "(mm)", "thickness": "(mm)", "weight": "(g)"}
    titles = {"length": "Length", "width": "Width", "thickness": "Thickness", "weight": "Weight"}

    for ax, var in zip(axes, DIMENSION_VARS):
        box_data = [pd.to_numeric(df_site.loc[df_site["category"] == cat, var], errors="coerce").dropna() for cat in CATEGORIES]
        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            widths=0.55,
            labels=CATEGORIES,
            medianprops=dict(linewidth=1.2, color="black"),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker="o", markersize=3.5, markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.7),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.95)
        ax.set_title(titles[var], fontsize=11)
        ax.set_ylabel(units[var], fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_facecolor("#eeeeee")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(df_site["site"].iloc[0], fontsize=12, y=1.02)
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    return fig


def run_dimension_stats(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["dimension"]
    data = df[[colmap[k] for k in ["site", "category", "length", "width", "thickness", "weight"]]].copy()
    data.columns = ["site", "category", "length", "width", "thickness", "weight"]
    data["site"] = data["site"].astype(str).str.strip()
    data["category"] = data["category"].apply(clean_category)
    for v in DIMENSION_VARS:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data[data["category"].isin(CATEGORIES)].copy()

    all_stats, all_tests = [], []
    pdf_path = outdir / "olduvai_category_boxplots.pdf"
    with PdfPages(pdf_path) as pdf:
        for site in SITES:
            df_site = data[data["site"].astype(str).map(canon_site).str.match(site_pattern(site))].copy()
            if df_site.empty:
                continue
            all_stats.append(descriptive_stats_site(df_site))
            all_tests.append(mann_whitney_tests_site(df_site))
            tag = canon_site(site).replace(" ", "_").replace("-", "_")
            fig = make_boxplot(df_site, outdir / f"{tag}_boxplot.png")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    stats_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    tests_df = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()
    if not stats_df.empty:
        stats_df.loc[:, ["Min", "Max", "Mean", "SD", "CI95_low", "CI95_high"]] = stats_df[["Min", "Max", "Mean", "SD", "CI95_low", "CI95_high"]].round(3)
    if not tests_df.empty:
        tests_df["U"] = tests_df["U"].round(4)
        tests_df["p_numeric"] = tests_df["p_value"]
        tests_df["p_value"] = tests_df["p_value"].apply(format_p)

    save_excel({"Descriptive_stats": stats_df, "Mann_Whitney_tests": tests_df}, outdir / "olduvai_category_statistics_and_tests.xlsx")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_XLSX}\n"
            "Please keep olduvai_RMS.py and olduvai_data.xlsx in the same folder."
        )
    make_dirs()
    df, colmap = read_data(INPUT_XLSX, SHEET_NAME)
    run_pca(df, colmap)
    run_mca(df, colmap)
    run_logistic(df, colmap)
    run_density_raw_material(df, colmap)
    run_density_within_site(df, colmap)
    run_density_among_sites(df, colmap)
    run_shape_analysis(df, colmap)
    run_dimension_stats(df, colmap)
    print("All analyses completed.")
    print(f"Input : {INPUT_XLSX}")
    print(f"Output: {OUTPUT_ROOT}")
    print("Subfolders:")
    for k, p in OUTDIRS.items():
        print(f"  - {p}")


if __name__ == "__main__":
    main()
