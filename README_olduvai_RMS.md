# Olduvai RMS reproducible analysis

This folder contains the data, Python code, dependency list, and installation helpers required to reproduce the analyses for the Olduvai stone-selection study.

The workflow includes PCA, MCA, binary logistic regression with average marginal effects, density analyses, shape chi-square tests, dimensional statistics, Mann-Whitney tests, Holm-adjusted p-values, figures, and supplementary output files.

---

## 1. Required files

Keep these files together in the same folder:

```text
olduvai_RMS/
├── olduvai_data.xlsx
├── olduvai_RMS.py
├── requirements.txt
├── install_modules_windows.bat
├── install_modules_macos.sh
└── README_olduvai_RMS.md
```

The script uses paths relative to its own location, so the project folder can be moved without editing any Windows or macOS path.

---

## 2. Reference Python environment

The complete workflow was successfully tested with:

- CPython 3.12.13, 64-bit
- the exact package versions pinned in `requirements.txt`
- Windows 11

For the closest numerical reproducibility, use Python 3.12.x in a clean virtual environment and install the pinned package versions. Python 3.12.13 is the reference version actually used for the verification run. Other Python versions may work, but they were not used for this verification.

Microsoft Excel is not required. The script reads and writes `.xlsx` files through Python packages.

---

## 3. Windows installation

### Recommended: use the installation helper

Double-click:

```text
install_modules_windows.bat
```

The helper searches for a 64-bit Python 3.12 interpreter, creates or reuses a project-local `.venv`, upgrades pip, installs the pinned packages, and verifies the required imports.

After installation, run the analysis from PowerShell or Command Prompt:

```bat
.venv\Scripts\python.exe olduvai_RMS.py
```

### Manual installation

If `python` refers to Python 3.12 on your computer:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe olduvai_RMS.py
```

If the `python` command is unavailable, use the full path to a Python 3.12 executable when creating `.venv`.

---

## 4. macOS installation

Install 64-bit Python 3.12, open Terminal, navigate to the project folder, and run:

```bash
chmod +x install_modules_macos.sh
./install_modules_macos.sh
```

The helper looks for `python3.12` (or a `python3` command that is Python 3.12), creates or reuses `.venv`, installs the pinned packages, and verifies the required imports.

Run the analysis with:

```bash
./.venv/bin/python olduvai_RMS.py
```

---

## 5. Input and output paths

The script uses these relative paths:

```python
BASE_DIR = Path(__file__).resolve().parent
INPUT_XLSX = BASE_DIR / "olduvai_data.xlsx"
OUTPUT_ROOT = BASE_DIR / "analysis"
```

It reads `olduvai_data.xlsx` from the project folder and creates `analysis/` in that same folder.

The main output directories are:

```text
analysis/
├── 00_RunInfo
├── 01_PCA
├── 02_MCA
├── 03_Logistic_AME
├── 04_Density_RawMaterial
├── 05_Density_WithinSite_ByCategory
├── 06_Density_AmongSites
├── 07_Shape_ChiSquare
├── 08_Dimension_CategoryStats
└── 09_Supplementary
```

If the run is successful, the final console message is:

```text
All analyses completed.
```

---

## 6. Reproducibility checklist

Before running the analysis, confirm that:

- `olduvai_RMS.py` and `olduvai_data.xlsx` are in the same folder;
- Python is 64-bit version 3.12.x;
- the exact versions in `requirements.txt` are installed in `.venv`;
- the project folder is writable;
- no existing output workbook is open in Excel;
- OneDrive files, if used, are fully downloaded and available locally.

With the same code, data, Python version, and package versions, the reported statistical tables should reproduce. Binary-identical files are not guaranteed across operating systems because fonts, PDF metadata, timestamps, and low-level numerical libraries can differ slightly.
