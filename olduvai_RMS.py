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

from datetime import datetime
import math
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
import statsmodels
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse, Patch
from matplotlib.transforms import Bbox
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

# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_XLSX = BASE_DIR / "olduvai_data.xlsx"
OUTPUT_ROOT = BASE_DIR / "analysis"
SHEET_NAME = 0  # use 0 to read the first sheet; change if needed
JITTER_SEED = 20260210

SITES = ["HWK EE", "MNK Skull", "EF-HR"]
CATEGORIES = ["Core", "Hammerstone", "Unmodified"]
NUMERIC_VARS_PCA = ["density", "length", "width", "thickness", "weight"]
DIMENSION_VARS = ["length", "width", "thickness", "weight"]
ACTIVE_MCA_VARS = ["sphericity", "angularity", "raw material"]
RAW_MATERIAL_ORDER = ["Phonolite", "Trachyte", "Basalt", "Lava Indet"]
SITE_DENSITY_COLORS = {
    "HWK EE": "#a6cee3",
    "MNK Skull": "#b2df8a",
    "EF-HR": "#fb9a99",
}
SITE_DENSITY_LEGEND_LABELS = {
    "HWK EE": "HWKEE",
    "MNK Skull": "MNK Skull",
    "EF-HR": "EFHR",
}
DENSITY_AXIS_FONTSIZE = 16.5
DENSITY_TICK_FONTSIZE = 16.5
MCA_FONT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}
MCA_MODALITY_STYLE = {
    "color": "#f2a23a",
    "marker": "^",
    "s": 28,
    "edgecolors": "none",
}
MCA_CATEGORY_STYLE = {
    "color": "#00a7d8",
    "marker": "D",
    "s": 28,
    "edgecolors": "none",
}
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
    "runinfo": OUTPUT_ROOT / "00_RunInfo",
    "pca": OUTPUT_ROOT / "01_PCA",
    "mca": OUTPUT_ROOT / "02_MCA",
    "logit": OUTPUT_ROOT / "03_Logistic_AME",
    "dens_rm": OUTPUT_ROOT / "04_Density_RawMaterial",
    "dens_within": OUTPUT_ROOT / "05_Density_WithinSite_ByCategory",
    "dens_among": OUTPUT_ROOT / "06_Density_AmongSites",
    "shape": OUTPUT_ROOT / "07_Shape_ChiSquare",
    "dimension": OUTPUT_ROOT / "08_Dimension_CategoryStats",
    "supplementary": OUTPUT_ROOT / "09_Supplementary",
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


def holm_adjust_pvalues(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted

    valid_values = values[valid]
    m = len(valid_values)
    order = np.argsort(valid_values)
    sorted_values = valid_values[order]
    adjusted_sorted = np.maximum.accumulate(
        (m - np.arange(m)) * sorted_values
    )
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted_valid = np.empty(m, dtype=float)
    adjusted_valid[order] = adjusted_sorted
    adjusted[valid] = adjusted_valid
    return adjusted


def add_holm_columns(
    results: pd.DataFrame,
    family_label: str,
) -> pd.DataFrame:
    if results.empty:
        return results

    out = results.copy()
    out["p_raw_numeric"] = pd.to_numeric(out["p_value"], errors="coerce")
    out["p_holm_numeric"] = holm_adjust_pvalues(out["p_raw_numeric"])
    out["p_raw"] = out["p_raw_numeric"].apply(format_p)
    out["p_holm"] = out["p_holm_numeric"].apply(format_p)
    out["significant_holm_0_05"] = out["p_holm_numeric"].apply(
        lambda value: "" if pd.isna(value) else bool(value < 0.05)
    )
    out["p_adjustment"] = "Holm"
    out["adjustment_family"] = family_label
    return out.drop(columns=["p_value"])


def save_excel(sheets: Dict[str, pd.DataFrame], path: Path) -> None:
    def write(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            for name, df in sheets.items():
                sheet_name = name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes = "A2"
                for column_index, column_name in enumerate(df.columns, start=1):
                    values = [str(column_name)]
                    values.extend(
                        "" if pd.isna(value) else str(value)
                        for value in df.iloc[:, column_index - 1]
                    )
                    width = min(max(len(value) for value in values) + 2, 40)
                    column_letter = worksheet.cell(row=1, column=column_index).column_letter
                    worksheet.column_dimensions[column_letter].width = width

    try:
        write(path)
    except PermissionError as exc:
        fallback = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        write(fallback)
        print(f"Warning: could not overwrite {path} ({exc}); saved {fallback}")


def unique_excel_sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\[\]\*:/\\?]", "_", str(name)).strip() or "Sheet"
    base = cleaned[:31]
    candidate = base
    i = 1
    while candidate in used:
        suffix = f"_{i}"
        candidate = f"{base[:31 - len(suffix)]}{suffix}"
        i += 1
    used.add(candidate)
    return candidate


def add_workbook_to_sheets(
    sheets: Dict[str, pd.DataFrame],
    used: set[str],
    source: Path,
    prefix: str,
) -> None:
    if not source.exists():
        return
    try:
        workbook_sheets = pd.read_excel(source, sheet_name=None)
    except Exception as exc:
        sheet_name = unique_excel_sheet_name(f"{prefix}_read_error", used)
        sheets[sheet_name] = pd.DataFrame({"source": [str(source)], "error": [repr(exc)]})
        return
    for sheet_name, table in workbook_sheets.items():
        out_name = unique_excel_sheet_name(f"{prefix}_{sheet_name}", used)
        sheets[out_name] = table


def add_csv_to_sheets(
    sheets: Dict[str, pd.DataFrame],
    used: set[str],
    source: Path,
    sheet_name: str,
) -> None:
    if not source.exists():
        return
    try:
        table = pd.read_csv(source)
    except Exception as exc:
        table = pd.DataFrame({"source": [str(source)], "error": [repr(exc)]})
    sheets[unique_excel_sheet_name(sheet_name, used)] = table


def copy_file_for_supplement(
    source: Path,
    destination: Path,
    manifest_rows: List[Dict[str, str]],
    category: str,
) -> None:
    if not source.exists():
        manifest_rows.append({
            "category": category,
            "source": str(source),
            "destination": "",
            "status": "missing",
        })
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, destination)
        copied_to = destination
    except PermissionError as exc:
        copied_to = destination.with_name(
            f"{destination.stem}_{datetime.now():%Y%m%d_%H%M%S}{destination.suffix}"
        )
        shutil.copy2(source, copied_to)
        print(f"Warning: could not overwrite {destination} ({exc}); saved {copied_to}")
    manifest_rows.append({
        "category": category,
        "source": str(source),
        "destination": str(copied_to),
        "status": "copied",
    })


def copy_directory_files_for_supplement(
    source_dir: Path,
    destination_dir: Path,
    manifest_rows: List[Dict[str, str]],
    category: str,
) -> None:
    if not source_dir.exists():
        manifest_rows.append({
            "category": category,
            "source": str(source_dir),
            "destination": "",
            "status": "missing folder",
        })
        return
    for source in sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file() and not p.name.startswith("~$")
    ):
        rel = source.relative_to(source_dir)
        copy_file_for_supplement(source, destination_dir / rel, manifest_rows, category)


def latest_file(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def write_run_info(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["runinfo"]

    versions = pd.DataFrame([
        {"package": "Python", "version": sys.version.split()[0]},
        {"package": "pandas", "version": pd.__version__},
        {"package": "numpy", "version": np.__version__},
        {"package": "scipy", "version": scipy.__version__},
        {"package": "statsmodels", "version": statsmodels.__version__},
        {"package": "scikit-learn", "version": sklearn.__version__},
        {"package": "matplotlib", "version": matplotlib.__version__},
    ])

    data = df.copy()
    data["_site_std"] = data[colmap["site"]].map(canon_site)
    data["_category_std"] = data[colmap["category"]].apply(clean_category)

    sample_counts = (
        data.groupby(["_site_std", "_category_std"], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={"_site_std": "site", "_category_std": "category"})
    )

    raw_material_counts = (
        data.groupby(["_site_std", "_category_std", colmap["raw material"]], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={"_site_std": "site", "_category_std": "category", colmap["raw material"]: "raw material"})
    )

    missing_rows = []
    for (site, category), group in data.groupby(["_site_std", "_category_std"], dropna=False):
        n = len(group)
        missing_rows.append({
            "site": site,
            "category": category,
            "n": n,
            "missing_sphericity": group[colmap["sphericity"]].isna().sum(),
            "missing_sphericity_%": group[colmap["sphericity"]].isna().mean() * 100 if n else np.nan,
            "missing_angularity": group[colmap["angularity"]].isna().sum(),
            "missing_angularity_%": group[colmap["angularity"]].isna().mean() * 100 if n else np.nan,
        })
    missing_morphology = pd.DataFrame(missing_rows)

    method_notes = pd.DataFrame([
        {"item": "Density and group comparisons", "note": "Kruskal-Wallis tests were used for three-group density comparisons; pairwise rank comparisons use Dunn-style z tests with Holm-adjusted p-values where applicable."},
        {"item": "PCA", "note": "Continuous variables were z-standardized before PCA. The script exports eigenvalues, loadings, scores, and within-site correlation matrices for the variables used in PCA."},
        {"item": "MCA", "note": "The main MCA uses complete cases for the active categorical variables and does not impute missing morphology. A sensitivity workflow treats missing or indeterminate observations as an active category."},
        {"item": "Logistic regression", "note": "Binary logit models were fitted with statsmodels Logit. The output workbook includes convergence fields, warnings, and class-specific predictor ranges to flag possible complete or quasi-complete separation."},
        {"item": "Figures", "note": "Boxplots include jittered raw data points so dispersion, clustering, and outliers remain visible."},
    ])

    save_excel({
        "Software_versions": versions,
        "Sample_counts": sample_counts,
        "Raw_material_counts": raw_material_counts,
        "Missing_morphology": missing_morphology,
        "Method_notes": method_notes,
    }, outdir / "analysis_run_info.xlsx")


def overlay_jittered_points(
    ax,
    grouped_values,
    positions,
    jitter_width: float = 0.045,
    facecolors: str = "white",
    edgecolors: str = "0.25",
    alpha: float = 0.45,
    linewidths: float = 0.45,
) -> None:
    rng = np.random.default_rng(JITTER_SEED)
    for pos, vals in zip(positions, grouped_values):
        vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        x = rng.normal(loc=pos, scale=jitter_width, size=len(vals))
        ax.scatter(
            x,
            vals,
            s=13,
            alpha=alpha,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=linewidths,
            zorder=3,
        )


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
        corr_df = (
            sub[NUMERIC_VARS_PCA]
            .corr()
            .reset_index()
            .rename(columns={"index": "variable"})
        )

        tag = canon_site(site).replace(" ", "_").replace("-", "_")
        sheets[f"{tag}_explained"] = explained
        sheets[f"{tag}_loadings"] = loading_df
        sheets[f"{tag}_scores"] = score_df
        sheets[f"{tag}_corr"] = corr_df

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
def _clean_mca_values(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Standardize MCA categorical variables and code indeterminate values as missing."""
    d = df.copy()
    missing_like = {
        "": np.nan,
        "nan": np.nan,
        "None": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "S.Def": np.nan,
        "SDef": np.nan,
        "Indeterminate": np.nan,
        "indeterminate": np.nan,
        "Unclear": np.nan,
        "unclear": np.nan,
    }
    for v in ACTIVE_MCA_VARS + [category_col]:
        d[v] = d[v].astype("object")
        d[v] = d[v].where(~d[v].isna(), np.nan)
        d[v] = d[v].astype(str).str.strip().replace(missing_like)
    return d


def prep_mca_main(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """
    Main MCA: complete-case analysis.

    Missing or indeterminate observations in active categorical variables
    are treated as unobserved and excluded. No modal imputation is used.
    """
    d = _clean_mca_values(df, category_col)
    return d.dropna(subset=ACTIVE_MCA_VARS).copy()


def prep_mca_sensitivity(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """
    Sensitivity MCA: missing/indeterminate observations are retained as an
    active category to evaluate preservation or observability effects.
    """
    d = _clean_mca_values(df, category_col)
    for v in ACTIVE_MCA_VARS:
        d[v] = d[v].fillna("Indeterminate")
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


def _split_mca_modality(modality: str) -> Tuple[str, str]:
    text = str(modality)
    for variable in ACTIVE_MCA_VARS:
        prefix = f"{variable}_"
        if text.startswith(prefix):
            return variable, text[len(prefix):]
    return "", text


def mca_display_label(modality: str) -> str:
    variable, value = _split_mca_modality(modality)
    label = str(value).strip()
    normalized = label.lower().replace("_", "-").replace(" ", "-")

    if normalized == "indeterminate":
        if variable == "raw material":
            return "Raw material indet."
        if variable:
            return f"{variable.capitalize()} indet."
        return "Indeterminate"

    if variable == "sphericity":
        if normalized in {"high", "high-sphericity"}:
            return "High-sphericity"
        if normalized in {"low", "low-sphericity"}:
            return "Low-sphericity"
    if variable == "angularity":
        replacements = {
            "Subangular": "Sub-angular",
            "Sub-Angular": "Sub-angular",
        }
        return replacements.get(label, label)
    if variable == "raw material":
        if label == "Trachyte":
            return "Trachyte-andesite"
        return label
    return label


def style_mca_axes(ax, x_values: Iterable[float], y_values: Iterable[float]) -> None:
    x = pd.to_numeric(pd.Series(list(x_values)), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(list(y_values)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        x = np.array([0.0])
    if len(y) == 0:
        y = np.array([0.0])

    xmin, xmax = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    ymin, ymax = min(float(y.min()), 0.0), max(float(y.max()), 0.0)
    xspan = max(xmax - xmin, 1.0)
    yspan = max(ymax - ymin, 1.0)
    ax.set_xlim(xmin - xspan * 0.12, xmax + xspan * 0.12)
    ax.set_ylim(ymin - yspan * 0.12, ymax + yspan * 0.12)

    ax.axhline(0, color="#bfbfbf", linewidth=0.75, zorder=0)
    ax.axvline(0, color="#bfbfbf", linewidth=0.75, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#555555")
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(axis="both", colors="#444444", direction="out", length=3.5, width=0.8)


def _offset_data(ax, x: float, y: float, x_points: float = 4.0, y_points: float = 0.0) -> Tuple[float, float]:
    pixels_per_point = ax.figure.dpi / 72.0
    px, py = ax.transData.transform((x, y))
    return ax.transData.inverted().transform((px + x_points * pixels_per_point, py + y_points * pixels_per_point))


def mca_label_position(ax, x: float, y: float) -> Tuple[float, float, str]:
    tx, ty = _offset_data(ax, x, y, x_points=4.0)
    return tx, ty, "left"


def repel_text(ax, texts, iterations: int = 300, anchor_points: Iterable[Tuple[float, float]] | None = None) -> None:
    if not texts:
        return

    anchors = np.asarray(list(anchor_points or []), dtype=float)
    lock_to_right = anchors.shape == (len(texts), 2)
    inv = ax.transData.inverted()

    for _ in range(iterations):
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer=renderer).expanded(1.04, 1.16) for t in texts]
        moves = [np.array([0.0, 0.0]) for _ in texts]
        min_label_x = []
        max_label_x = []
        min_label_y = []
        max_label_y = []
        if lock_to_right:
            for anchor_x, anchor_y in anchors:
                min_x, _ = _offset_data(ax, float(anchor_x), float(anchor_y), x_points=3.2)
                max_x, _ = _offset_data(ax, float(anchor_x), float(anchor_y), x_points=6.0)
                _, min_y = _offset_data(ax, float(anchor_x), float(anchor_y), y_points=-8.0)
                _, max_y = _offset_data(ax, float(anchor_x), float(anchor_y), y_points=8.0)
                min_label_x.append(min_x)
                max_label_x.append(max_x)
                min_label_y.append(min_y)
                max_label_y.append(max_y)

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                ci = np.array([(boxes[i].x0 + boxes[i].x1) / 2, (boxes[i].y0 + boxes[i].y1) / 2])
                cj = np.array([(boxes[j].x0 + boxes[j].x1) / 2, (boxes[j].y0 + boxes[j].y1) / 2])
                direction = ci - cj
                if np.allclose(direction, 0):
                    direction = np.array([1.0, 0.6])
                direction = direction / np.linalg.norm(direction)
                overlap_x = min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0)
                overlap_y = min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0)
                push = max(1.4, min(overlap_x, overlap_y) * 0.62)
                if lock_to_right:
                    direction[0] *= 0.18
                moves[i] += direction * push
                moves[j] -= direction * push

        if anchors.size:
            anchor_pixels = ax.transData.transform(anchors)
            for i, box in enumerate(boxes):
                center = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
                for px, py in anchor_pixels:
                    point_box = Bbox.from_extents(px - 4.0, py - 4.0, px + 4.0, py + 4.0)
                    if not box.overlaps(point_box):
                        continue
                    direction = center - np.array([px, py])
                    if np.allclose(direction, 0):
                        direction = np.array([1.0, 0.4])
                    direction = direction / np.linalg.norm(direction)
                    moves[i] += direction * 1.2

        axes_box = ax.get_window_extent(renderer=renderer)
        for i, box in enumerate(boxes):
            if box.x0 < axes_box.x0 + 2:
                moves[i][0] += axes_box.x0 + 2 - box.x0
            if box.x1 > axes_box.x1 - 2:
                moves[i][0] -= box.x1 - axes_box.x1 + 2
            if box.y0 < axes_box.y0 + 2:
                moves[i][1] += axes_box.y0 + 2 - box.y0
            if box.y1 > axes_box.y1 - 2:
                moves[i][1] -= box.y1 - axes_box.y1 + 2

        max_move = 0.0
        for text, move in zip(texts, moves):
            dist = float(np.linalg.norm(move))
            if dist == 0:
                continue
            x, y = text.get_position()
            x2, y2 = inv.transform(ax.transData.transform((x, y)) + move)
            text.set_position((x2, y2))
            max_move = max(max_move, dist)
        if lock_to_right:
            for text, min_x, max_x, min_y, max_y in zip(texts, min_label_x, max_label_x, min_label_y, max_label_y):
                x, y = text.get_position()
                text.set_position((min(max(x, min_x), max_x), min(max(y, min_y), max_y)))
                text.set_ha("left")
        if max_move < 0.25:
            break


def run_mca(df: pd.DataFrame, colmap: Dict[str, str]) -> None:
    outdir = OUTDIRS["mca"]
    summary_rows = []
    missing_rows = []

    for site in SITES:
        sub = site_subset(df, colmap, site)
        use = sub[[colmap["category"], colmap["sphericity"], colmap["angularity"], colmap["raw material"]]].copy()
        use.columns = ["category", "sphericity", "angularity", "raw material"]
        site_dir = outdir / canon_site(site).replace(" ", "_").replace("-", "_")
        site_dir.mkdir(exist_ok=True)
        use.to_csv(site_dir / "subset_used.csv", index=False)
        clean_for_missing = _clean_mca_values(use, "category")
        complete_for_cramers = clean_for_missing.dropna(subset=ACTIVE_MCA_VARS).copy()
        cramers_v_table(complete_for_cramers, "category").to_csv(site_dir / "cramersV_category_vs_vars_complete_cases.csv", index=False)

        clean_for_missing = _clean_mca_values(use, "category")
        for category, group in clean_for_missing.groupby("category", dropna=False):
            n = len(group)
            complete_active = group.dropna(subset=ACTIVE_MCA_VARS)
            missing_rows.append({
                "site": site,
                "category": category,
                "n": n,
                "missing_sphericity": int(group["sphericity"].isna().sum()),
                "missing_sphericity_%": group["sphericity"].isna().mean() * 100 if n else np.nan,
                "missing_angularity": int(group["angularity"].isna().sum()),
                "missing_angularity_%": group["angularity"].isna().mean() * 100 if n else np.nan,
                "missing_raw_material": int(group["raw material"].isna().sum()),
                "missing_raw_material_%": group["raw material"].isna().mean() * 100 if n else np.nan,
                "complete_active_n": len(complete_active),
                "complete_active_%": len(complete_active) / n * 100 if n else np.nan,
            })

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

            coords["label"] = coords["modality"].apply(mca_display_label)
            plot_x = pd.concat([coords["Dim1"], centroids["Dim1"]], ignore_index=True)
            plot_y = pd.concat([coords["Dim2"], centroids["Dim2"]], ignore_index=True)

            with plt.rc_context(MCA_FONT_RC):
                fig, ax = plt.subplots(figsize=(5.2, 4.4))
                style_mca_axes(ax, plot_x, plot_y)

                ax.scatter(coords["Dim1"], coords["Dim2"], zorder=3, **MCA_MODALITY_STYLE)
                ax.scatter(centroids["Dim1"], centroids["Dim2"], zorder=4, **MCA_CATEGORY_STYLE)

                texts = []
                anchors = []
                for _, r in coords.iterrows():
                    anchors.append((r.Dim1, r.Dim2))
                    tx, ty, ha = mca_label_position(ax, float(r.Dim1), float(r.Dim2))
                    texts.append(
                        ax.text(
                            tx,
                            ty,
                            r.label,
                            fontsize=9,
                            color="#222222",
                            ha=ha,
                            va="center",
                            zorder=5,
                        )
                    )
                for _, r in centroids.iterrows():
                    anchors.append((r.Dim1, r.Dim2))
                    tx, ty, ha = mca_label_position(ax, float(r.Dim1), float(r.Dim2))
                    texts.append(
                        ax.text(
                            tx,
                            ty,
                            r["category"],
                            fontsize=9,
                            color="#222222",
                            ha=ha,
                            va="center",
                            zorder=5,
                        )
                    )

                repel_text(ax, texts, anchor_points=anchors)
                ax.set_xlabel(f"Dim 1 ({et.loc[0, 'var_%']:.1f}%)")
                ax.set_ylabel(f"Dim 2 ({et.loc[1, 'var_%']:.1f}%)")
            fig.tight_layout(pad=0.45)
            fig.savefig(wdir / "biplot.png", dpi=300, bbox_inches="tight")
            fig.savefig(wdir / "biplot.pdf", bbox_inches="tight")
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
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(outdir / "MCA_missingness_by_category.csv", index=False)


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


def predictor_range_checks(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        if col == "const":
            continue
        vals = pd.to_numeric(X[col], errors="coerce")
        vals0 = vals[y == 0].dropna()
        vals1 = vals[y == 1].dropna()
        if vals0.empty or vals1.empty:
            continue
        low0, high0 = vals0.min(), vals0.max()
        low1, high1 = vals1.min(), vals1.max()
        overlap = max(low0, low1) <= min(high0, high1)
        rows.append({
            "predictor": col,
            "unmodified_min": low0,
            "unmodified_max": high0,
            "selected_min": low1,
            "selected_max": high1,
            "non_overlapping_ranges": not overlap,
        })
    return pd.DataFrame(rows)


def fit_logit(
    data: pd.DataFrame,
    target: str,
    predictors: List[str],
    categorical: Dict[str, List[str]] | None = None,
):
    d = data[data["category"].isin([target, "Unmodified"])].copy()
    d["y"] = (d["category"] == target).astype(int)
    d = d[["y"] + predictors].dropna().copy()
    info = {
        "target": target,
        "n": len(d),
        "selected_n": int(d["y"].sum()) if not d.empty else 0,
        "unmodified_n": int((d["y"] == 0).sum()) if not d.empty else 0,
        "fit_status": "not_fit",
        "converged": np.nan,
        "llf": np.nan,
        "llnull": np.nan,
        "pseudo_r2": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "warnings": "",
        "ame_warnings": "",
        "ame_error": "",
        "error": "",
    }
    if d.empty or d["y"].nunique() < 2:
        info["error"] = "Insufficient complete observations or only one outcome class."
        return None, None, None, None, info, pd.DataFrame()
    X = d[predictors].copy()
    if categorical:
        for col, levels in categorical.items():
            X[col] = pd.Categorical(X[col], categories=levels)
        X = pd.get_dummies(X, columns=list(categorical.keys()), drop_first=True)
    X = sm.add_constant(X, has_constant="add").astype(float)
    y = d["y"].astype(int)
    range_checks = predictor_range_checks(y, X)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.Logit(y, X).fit(disp=False)
        info["warnings"] = " | ".join(sorted({str(w.message) for w in caught}))
    except Exception as exc:
        info["fit_status"] = "failed"
        info["error"] = repr(exc)
        return d, X, None, None, info, range_checks

    info["fit_status"] = "fit"
    info["converged"] = bool(model.mle_retvals.get("converged", np.nan))
    info["llf"] = model.llf
    info["llnull"] = model.llnull
    info["pseudo_r2"] = model.prsquared
    info["aic"] = model.aic
    info["bic"] = model.bic

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
        with warnings.catch_warnings(record=True) as ame_caught:
            warnings.simplefilter("always")
            ame = (
                model.get_margeff(at="overall")
                .summary_frame()
                .reset_index()
                .rename(
                    columns={
                        "index": "predictor",
                        "dy/dx": "AME",
                        "Pr(>|z|)": "AME_p_value",
                    }
                )
            )
        info["ame_warnings"] = " | ".join(sorted({str(w.message) for w in ame_caught}))
        res = res.merge(ame[["predictor", "AME", "AME_p_value"]], on="predictor", how="left")
    except Exception as exc:
        info["ame_error"] = repr(exc)
        res["AME"] = np.nan
        res["AME_p_value"] = np.nan
    return d, X, model, res, info, range_checks


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
    all_results, all_pca, all_diagnostics, all_range_checks = [], [], [], []

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

        d_h, X_h, m_h, res_h, info_h, checks_h = fit_logit(site_df, "Hammerstone", ["density", "pc1", "sphericity", "angularity_ord"], categorical={"sphericity": ["Low", "High"]})
        info_h.update({"site": site, "model": "Hammerstone vs Unmodified"})
        all_diagnostics.append(info_h)
        if not checks_h.empty:
            checks_h.insert(0, "site", site)
            checks_h.insert(1, "model", "Hammerstone vs Unmodified")
            all_range_checks.append(checks_h)
        res_h = clean_predictor_names(res_h)
        if res_h is not None:
            res_h.insert(0, "site", site)
            res_h.insert(1, "model", "Hammerstone vs Unmodified")
            all_results.append(res_h)

        d_c, X_c, m_c, res_c, info_c, checks_c = fit_logit(site_df, "Core", ["density", "pc1"])
        info_c.update({"site": site, "model": "Core vs Unmodified"})
        all_diagnostics.append(info_c)
        if not checks_c.empty:
            checks_c.insert(0, "site", site)
            checks_c.insert(1, "model", "Core vs Unmodified")
            all_range_checks.append(checks_c)
        res_c = clean_predictor_names(res_c)
        if res_c is not None:
            res_c.insert(0, "site", site)
            res_c.insert(1, "model", "Core vs Unmodified")
            all_results.append(res_c)

        tag = canon_site(site).replace(" ", "_").replace("-", "_")
        plot_predictions(site, d_h, m_h, d_c, m_c, outdir / f"{tag}_probability_plot.png")

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    pca_df = pd.concat(all_pca, ignore_index=True) if all_pca else pd.DataFrame()
    diagnostics_df = pd.DataFrame(all_diagnostics)
    range_checks_df = pd.concat(all_range_checks, ignore_index=True) if all_range_checks else pd.DataFrame()

    if not results_df.empty:
        results_df["OR (95% CI)"] = (
            results_df["OR"].round(3).astype(str) + " (" +
            results_df["OR_95CI_low"].round(3).astype(str) + "–" +
            results_df["OR_95CI_high"].round(3).astype(str) + ")"
        )

    save_excel({
        "Logistic_results": results_df,
        "PCA_loadings": pca_df,
        "Model_diagnostics": diagnostics_df,
        "Predictor_range_checks": range_checks_df,
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
    overall_groups = [tagged.loc[tagged["site_std"] == s, "density"].to_numpy() for s in SITES]
    ax.boxplot(overall_groups, tick_labels=SITES, widths=0.35)
    overlay_jittered_points(ax, overall_groups, range(1, len(SITES) + 1))
    ax.set_ylabel("Density", fontsize=DENSITY_AXIS_FONTSIZE)
    ax.tick_params(axis="both", labelsize=DENSITY_TICK_FONTSIZE)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "three_sites_density_boxplot.png", dpi=300)
    plt.close(fig)

    # Plot grouped boxplot
    fig, ax = plt.subplots(figsize=(10, 5.8))
    base_positions, offsets, width = [1, 2.45, 3.9], [-0.28, 0.0, 0.28], 0.22
    for j, site in enumerate(SITES):
        color = SITE_DENSITY_COLORS[site]
        site_data = [tagged.loc[(tagged["category"] == cat) & (tagged["site_std"] == site), "density"].to_numpy() for cat in CATEGORIES]
        positions = [base_positions[i] + offsets[j] for i in range(len(CATEGORIES))]
        bp = ax.boxplot(
            site_data,
            positions=positions,
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            medianprops=dict(color="black", linewidth=1.1),
            boxprops=dict(linewidth=1.0, color="black"),
            whiskerprops=dict(linewidth=1.0, color="black"),
            capprops=dict(linewidth=1.0, color="black"),
            flierprops=dict(
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                linestyle="none",
            ),
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(1.0)
            box.set_edgecolor("black")
        overlay_jittered_points(
            ax,
            site_data,
            positions,
            jitter_width=0.026,
            facecolors=color,
            edgecolors="0.30",
            alpha=0.50,
            linewidths=0.35,
        )
    legend_handles = [
        Patch(facecolor=SITE_DENSITY_COLORS[site], edgecolor=SITE_DENSITY_COLORS[site], label=SITE_DENSITY_LEGEND_LABELS[site])
        for site in SITES
    ]
    ax.set_xticks(base_positions)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Density", fontsize=DENSITY_AXIS_FONTSIZE)
    ax.tick_params(axis="both", labelsize=DENSITY_TICK_FONTSIZE)
    ax.legend(handles=legend_handles, frameon=True)
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
    for g1, g2 in [
        ("Core", "Hammerstone"),
        ("Core", "Unmodified"),
        ("Hammerstone", "Unmodified"),
    ]:
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


def hammerstone_stats_among_sites(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    site_samples: Dict[str, pd.DataFrame] = {}
    descriptive_rows = []
    test_rows = []

    for site in SITES:
        mask = data["site"].astype(str).map(canon_site).str.match(site_pattern(site))
        site_samples[site] = data.loc[
            mask & (data["category"] == "Hammerstone"),
            ["length", "weight"],
        ].copy()

        for var in ["length", "weight"]:
            values = pd.to_numeric(site_samples[site][var], errors="coerce").dropna()
            mean, ci_low, ci_high = mean_ci95(values)
            descriptive_rows.append({
                "Site": site,
                "Category": "Hammerstone",
                "Variable": var,
                "N": len(values),
                "Min": values.min() if len(values) else np.nan,
                "Max": values.max() if len(values) else np.nan,
                "Mean": mean,
                "SD": values.std(ddof=1) if len(values) > 1 else np.nan,
                "CI95_low": ci_low,
                "CI95_high": ci_high,
            })

    site_comparisons = [
        ("HWK EE", "MNK Skull"),
        ("HWK EE", "EF-HR"),
        ("MNK Skull", "EF-HR"),
    ]
    for site1, site2 in site_comparisons:
        for var in ["length", "weight"]:
            x1 = pd.to_numeric(site_samples[site1][var], errors="coerce").dropna()
            x2 = pd.to_numeric(site_samples[site2][var], errors="coerce").dropna()
            if len(x1) == 0 or len(x2) == 0:
                u_stat, p_val = np.nan, np.nan
            else:
                u_stat, p_val = mannwhitneyu(x1, x2, alternative="two-sided")
            test_rows.append({
                "Category": "Hammerstone",
                "Comparison": f"{site1} vs {site2}",
                "Variable": var,
                "N_group1": len(x1),
                "N_group2": len(x2),
                "U": u_stat,
                "p_value": p_val,
            })

    return pd.DataFrame(descriptive_rows), pd.DataFrame(test_rows)


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
            tick_labels=CATEGORIES,
            medianprops=dict(linewidth=1.2, color="black"),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker="o", markersize=3.5, markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.7),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.95)
        overlay_jittered_points(ax, box_data, range(1, len(CATEGORIES) + 1))
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
        tests_df = add_holm_columns(
            tests_df,
            f"Table 7: all {len(tests_df)} Mann-Whitney tests",
        )

    table7_notes = pd.DataFrame([
        {
            "Item": "P-value adjustment",
            "Details": "Holm step-down family-wise error rate correction",
        },
        {
            "Item": "Adjustment family",
            "Details": f"All {len(tests_df)} Mann-Whitney tests reported in Table 7",
        },
        {
            "Item": "Significance threshold",
            "Details": "Adjusted p (p_holm) < 0.05",
        },
        {
            "Item": "U statistic convention",
            "Details": "U for the first-listed group (SciPy mannwhitneyu statistic)",
        },
    ])
    save_excel(
        {
            "Descriptive_stats": stats_df,
            "Mann_Whitney_tests": tests_df,
            "P_adjustment_notes": table7_notes,
        },
        outdir / "olduvai_category_statistics_and_tests.xlsx",
    )

    hammerstone_stats_df, hammerstone_tests_df = hammerstone_stats_among_sites(data)
    if not hammerstone_stats_df.empty:
        hammerstone_stats_df.loc[
            :,
            ["Min", "Max", "Mean", "SD", "CI95_low", "CI95_high"],
        ] = hammerstone_stats_df[
            ["Min", "Max", "Mean", "SD", "CI95_low", "CI95_high"]
        ].round(3)
    if not hammerstone_tests_df.empty:
        hammerstone_tests_df["U"] = hammerstone_tests_df["U"].round(4)
        hammerstone_tests_df = add_holm_columns(
            hammerstone_tests_df,
            f"Table 8: all {len(hammerstone_tests_df)} Mann-Whitney tests",
        )

    table8_notes = pd.DataFrame([
        {
            "Item": "P-value adjustment",
            "Details": "Holm step-down family-wise error rate correction",
        },
        {
            "Item": "Adjustment family",
            "Details": f"All {len(hammerstone_tests_df)} Mann-Whitney tests reported in Table 8",
        },
        {
            "Item": "Significance threshold",
            "Details": "Adjusted p (p_holm) < 0.05",
        },
        {
            "Item": "U statistic convention",
            "Details": "U for the first-listed assemblage (SciPy mannwhitneyu statistic)",
        },
    ])
    save_excel(
        {
            "Descriptive_stats": hammerstone_stats_df,
            "Mann_Whitney_tests": hammerstone_tests_df,
            "P_adjustment_notes": table8_notes,
        },
        outdir / "olduvai_hammerstone_among_sites_statistics_and_tests.xlsx",
    )


# ---------------------------------------------------------------------
# 09 Supplementary package
# ---------------------------------------------------------------------
def reset_supplementary_root() -> Path:
    supp_root = OUTDIRS["supplementary"]
    output_root = OUTPUT_ROOT.resolve()
    resolved = supp_root.resolve()
    if output_root not in resolved.parents:
        raise RuntimeError(f"Refusing to reset unexpected supplementary path: {supp_root}")
    if supp_root.exists():
        try:
            shutil.rmtree(supp_root)
        except PermissionError as exc:
            supp_root = OUTPUT_ROOT / f"09_Supplementary_{datetime.now():%Y%m%d_%H%M%S}"
            print(
                f"Warning: could not reset {OUTDIRS['supplementary']} ({exc}); "
                f"saved package to {supp_root}"
            )
    supp_root.mkdir(parents=True, exist_ok=True)
    return supp_root


def write_supplementary_master(supp_root: Path) -> Path:
    sheets: Dict[str, pd.DataFrame] = {}
    used: set[str] = set()

    logistic_file = latest_file(
        OUTDIRS["logit"],
        "Olduvai_logistic_results_all_sites*.xlsx",
    )
    dimension_file = latest_file(
        OUTDIRS["dimension"],
        "olduvai_category_statistics_and_tests*.xlsx",
    )
    hammerstone_site_file = latest_file(
        OUTDIRS["dimension"],
        "olduvai_hammerstone_among_sites_statistics_and_tests*.xlsx",
    )

    workbook_sources = [
        ("RunInfo", OUTDIRS["runinfo"] / "analysis_run_info.xlsx"),
        ("PCA", OUTDIRS["pca"] / "PCA_results_3sites.xlsx"),
        ("Logit", logistic_file),
        ("DenRaw", OUTDIRS["dens_rm"] / "density_raw_material_statistics.xlsx"),
        ("DenWithin", OUTDIRS["dens_within"] / "three_sites_density_kruskal_dunn.xlsx"),
        ("DenSites", OUTDIRS["dens_among"] / "three_sites_density_stats_kruskal_dunn.xlsx"),
        ("DenSiteCat", OUTDIRS["dens_among"] / "three_sites_by_category_density_stats_kruskal_dunn.xlsx"),
        ("Shape", OUTDIRS["shape"] / "hammerstone_unmodified_angularity_sphericity_chisquare.xlsx"),
        ("Dimension", dimension_file),
        ("HammerSite", hammerstone_site_file),
    ]

    for prefix, source in workbook_sources:
        if source is not None:
            add_workbook_to_sheets(sheets, used, source, prefix)

    add_csv_to_sheets(sheets, used, OUTDIRS["mca"] / "MCA_summary.csv", "MCA_summary")
    add_csv_to_sheets(
        sheets,
        used,
        OUTDIRS["mca"] / "MCA_missingness_by_category.csv",
        "MCA_missingness",
    )

    for csv_path in sorted(OUTDIRS["mca"].rglob("*.csv")):
        if csv_path.name in {"MCA_summary.csv", "MCA_missingness_by_category.csv"}:
            continue
        rel = csv_path.relative_to(OUTDIRS["mca"]).with_suffix("")
        sheet_name = "MCA_" + "_".join(rel.parts)
        add_csv_to_sheets(sheets, used, csv_path, sheet_name)

    master_path = supp_root / "Supplementary_master.xlsx"
    save_excel(sheets, master_path)
    return master_path


def run_supplementary_bundle() -> None:
    supp_root = reset_supplementary_root()
    manifest_rows: List[Dict[str, str]] = []

    category_dirs = {
        "00_RunInfo_Methods": OUTDIRS["runinfo"],
        "01_PCA": OUTDIRS["pca"],
        "02_MCA": OUTDIRS["mca"],
        "04_Density_RawMaterial": OUTDIRS["dens_rm"],
        "05_Density_WithinSite_ByCategory": OUTDIRS["dens_within"],
        "06_Density_AmongSites": OUTDIRS["dens_among"],
        "07_Shape_ChiSquare": OUTDIRS["shape"],
        "08_Dimension_CategoryStats": OUTDIRS["dimension"],
    }

    for category, source_dir in category_dirs.items():
        copy_directory_files_for_supplement(
            source_dir,
            supp_root / category,
            manifest_rows,
            category,
        )

    logit_dir = supp_root / "03_Logistic_AME"
    latest_logit = latest_file(OUTDIRS["logit"], "Olduvai_logistic_results_all_sites*.xlsx")
    if latest_logit is not None:
        copy_file_for_supplement(
            latest_logit,
            logit_dir / "Olduvai_logistic_results_all_sites.xlsx",
            manifest_rows,
            "03_Logistic_AME",
        )
    for plot in sorted(OUTDIRS["logit"].glob("*.png")):
        copy_file_for_supplement(plot, logit_dir / plot.name, manifest_rows, "03_Logistic_AME")

    code_dir = supp_root / "09_Reproducibility_Code_Data"
    copy_file_for_supplement(Path(__file__).resolve(), code_dir / "olduvai_RMS.py", manifest_rows, "09_Reproducibility_Code_Data")
    copy_file_for_supplement(INPUT_XLSX, code_dir / INPUT_XLSX.name, manifest_rows, "09_Reproducibility_Code_Data")
    requirements = BASE_DIR / "requirements.txt"
    if requirements.exists():
        copy_file_for_supplement(requirements, code_dir / requirements.name, manifest_rows, "09_Reproducibility_Code_Data")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = supp_root / "Supplementary_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    master_path = write_supplementary_master(supp_root)

    readme_lines = [
        "Supplementary package generated by olduvai_RMS.py",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Contents:",
        "00_RunInfo_Methods: software versions, sample counts, missingness, method notes.",
        "01_PCA: PCA eigenvalues, loadings, scores, correlations, and figures.",
        "02_MCA: main and sensitivity MCA outputs, missingness summaries, and biplots.",
        "03_Logistic_AME: logistic regression results, diagnostics, predictor-range checks, and probability plots.",
        "04-06_Density: raw-material, within-site, and among-site density statistics and figures.",
        "07_Shape_ChiSquare: angularity/sphericity chi-square summaries.",
        "08_Dimension_CategoryStats: size and weight summaries, within-site category tests, among-site hammerstone tests, and boxplots.",
        "09_Reproducibility_Code_Data: analysis script, input workbook, and requirements file when available.",
        "",
        f"Master workbook: {master_path.name}",
        f"Manifest: {manifest_path.name}",
    ]
    (supp_root / "README_supplementary_files.txt").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )


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
    write_run_info(df, colmap)
    run_pca(df, colmap)
    run_mca(df, colmap)
    run_logistic(df, colmap)
    run_density_raw_material(df, colmap)
    run_density_within_site(df, colmap)
    run_density_among_sites(df, colmap)
    run_shape_analysis(df, colmap)
    run_dimension_stats(df, colmap)
    run_supplementary_bundle()
    print("All analyses completed.")
    print(f"Input : {INPUT_XLSX}")
    print(f"Output: {OUTPUT_ROOT}")
    print("Subfolders:")
    for k, p in OUTDIRS.items():
        print(f"  - {p}")


if __name__ == "__main__":
    main()
