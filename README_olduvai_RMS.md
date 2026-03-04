# Olduvai RMS Compact Analysis Script

This project provides a compact and reproducible Python workflow for the main analyses used in the Olduvai stone selection study, including:

- PCA
- MCA
- Binary logistic regression with average marginal effects (AME)
- Density analysis by raw material
- Density analysis within sites by category
- Density analysis among sites
- Shape analysis (chi-square tests for angularity and sphericity)
- Dimensional statistics and boxplots

The script uses a single Excel file as input and automatically saves outputs into separate subfolders for each analysis stage.

---

## 1. Required files

Place the following files in the same project folder, for example:

```text
olduvai_RMS/
├── olduvai_data.xlsx
├── olduvai_RMS.py
├── install_modules_py314.bat
├── install_modules_macos.sh
├── requirements.txt
└── README_olduvai_RMS.md
```

To run the project without changing paths, keep **`olduvai_RMS.py`** and **`olduvai_data.xlsx`** in the **same folder**.

---

## 2. Recommended Python environment

This script is intended to run with **Python 3.14 (64-bit)** and depends on several common scientific Python packages.

Recommended setup:

- Python 3.14
- A clean virtual environment

---

## 3. Installing the required packages on Windows

### Option 1. Double-click the installation script

Double-click `install_modules_py314.bat`.

### Option 2. Install manually from the command line

Open **PowerShell** or **Command Prompt** and run:

```bash
pip install -r requirements.txt
```

### Optional: use a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Installing the required packages on macOS

### Option 1. Run the installation script

Open **Terminal**, navigate to the project folder, and run:

```bash
chmod +x install_modules_macos.sh
./install_modules_macos.sh
```

This script will:

- check whether `python3` is installed;
- make sure `pip` is available;
- optionally create a virtual environment (`.venv`);
- install all required modules from `requirements.txt`.

### Option 2. Install manually from the command line

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

---

## 5. Input and output paths

The `olduvai_RMS.py` uses **relative paths** by default:

```python
BASE_DIR = Path(__file__).resolve().parent
INPUT_XLSX = BASE_DIR / "olduvai_data.xlsx"
OUTPUT_ROOT = BASE_DIR / "analysis"
```

This means:

- the script automatically reads `olduvai_data.xlsx` from the same folder as `olduvai_RMS.py`;
- the output folder `analysis` will be created automatically in the same folder.

You do **not** need to edit local Windows or macOS paths as long as the file structure is kept the same.

---

## 6. Running the script

### Windows

```bash
python olduvai_RMS.py
```

or

```bash
py olduvai_RMS.py
```

### macOS

```bash
python3 olduvai_RMS.py
```

If the run is successful, the script will print:

```text
All analyses completed.
```

---

## 7. Reproducibility checklist

Before running the code, make sure that:

- `olduvai_RMS.py` and `olduvai_data.xlsx` are in the same folder;
- `requirements.txt` is in the project folder;
- all required packages are installed;
- the recommended Python version is being used;
- no output files are open during execution.

---

## 8. Output structure

The script automatically creates the following subfolders under `analysis`:

```text
01_PCA
02_MCA
03_Logistic_AME
04_Density_RawMaterial
05_Density_WithinSite_ByCategory
06_Density_AmongSites
07_Shape_ChiSquare
08_Dimension_CategoryStats
```

### Contents of each folder

**01_PCA**

- PCA tables
- loading plots
- score plots with 95% confidence ellipses
- biplots

**02_MCA**

- MCA subsets
- eigenvalue tables
- contribution tables
- cos2 tables
- centroids
- biplots (PNG and PDF)

**03_Logistic_AME**

- logistic regression tables
- odds ratios
- average marginal effects
- probability plots

**04_Density_RawMaterial**

- descriptive statistics by raw material
- Mann-Whitney pairwise comparisons

**05_Density_WithinSite_ByCategory**

- descriptive statistics within each site
- Kruskal-Wallis tests
- Dunn’s post hoc tests
- density distribution plots

**06_Density_AmongSites**

- overall among-site tests
- by-category among-site tests
- Dunn’s post hoc tests
- density boxplots

**07_Shape_ChiSquare**

- contingency tables
- chi-square results for angularity and sphericity

**08_Dimension_CategoryStats**

- descriptive statistics for dimensions
- Mann-Whitney tests
- PNG boxplots
- combined PDF of boxplots
