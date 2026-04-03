# X-RHO-SCATTER

**Coherent elastic x-ray scattering from non-aufbau electronic configurations**

A Python code that computes coherent elastic x-ray scattering differential cross-sections from molecular electron densities obtained via Hartree-Fock or DFT. The code:

1. **Wavefunction generation** (`wavefunction/`): Uses Psi4/Psi4NumPy to compute ground-state, localized, and non-aufbau (e.g. core-hole) wavefunctions via the Maximum Overlap Method (MOM), outputting `.molden` files.

2. **Scattering calculation** (`scattering/`): Reads `.molden` files with ORBKIT, builds the electron density on a real-space grid, computes the molecular form factor via numerical Fourier-transform integration (cubature), and multiplies by the Thomson differential cross-section to obtain dsigma/dOmega.

Developed for coherent x-ray scattering on 1,3-cyclohexadiene core-hole cation states, the results of which can be found in the Faraday Discussion paper below.

---

## Citation

If you use or are inspired by this code please cite:

**Ultraintense, ultrashort pulse X-ray scattering in small molecules**
- P. J. Ho, A. E. A. Fouda, K. Li, G. Doumy and L. Young
- *Faraday Discuss.*, 2021, **228**, 139-160
- [https://doi.org/10.1039/D0FD00106F](https://doi.org/10.1039/D0FD00106F)

This code uses Psi4NumPy, ORBKIT and cubature. Please cite the associated papers:
- [Psi4NumPy](https://github.com/psi4/psi4numpy)
- [ORBKIT](https://github.com/orbkit/orbkit)
- [cubature](https://github.com/saullocastro/cubature)

`wf.py` is an adaptation of PSIXAS, and `kshelper.py` is taken from PSIXAS:
- [PSIXAS](https://github.com/Masterluke87/psixas)

## Table of Contents

1. [Code Architecture](#code-architecture)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Running Simulations](#running-simulations)
5. [Input File Reference](#input-file-reference)
6. [Output Files](#output-files)
7. [Example Workflow](#example-workflow)

---

## Code Architecture

```
X-RHO-SCATTER/
├── wavefunction/
│   ├── run_wf.py              # Driver script: defines geometry, options, runs wavefunction pipeline
│   ├── wf.py                  # Core wavefunction module: ground_state(), localize(), non_aufbau_state()
│   ├── kshelper.py            # SCF helper utilities from PSIXAS: diag_H, Timer, ACDIIS, printHeader
│   └── wf_outfiles/           # Output directory for .molden and .npz orbital files
│       ├── Neutral_HF.molden
│       ├── Neutral_HF_loc.molden
│       ├── Core_Hole_HF.molden
│       ├── hf_gsorbs.npz
│       └── hf_loc_gsorbs.npz
├── scattering/
│   ├── run_dsigma.py          # Driver script: defines energies, states, methods, runs scattering calc
│   ├── dsigma.py              # Scattering module: crosssection class, Mol_Form_Factor, Thomson, Qvector
│   ├── run_plotting.py        # Driver script: reads output data and produces detector plots
│   ├── plotting.py            # Plotting module: plot_detector class with log_data() and method_diff()
│   ├── molden_files/          # Input .molden files (user-provided or copied from wavefunction/)
│   ├── data_files/            # Output scattering data files
│   └── plots/                 # Output plot images
└── README.md
```

### Module Summary

| Module | File | Role |
|--------|------|------|
| **Wavefunction** | `wf.py` | Modified From PSIXAS: Three-step wavefunction pipeline: ground-state SCF -> Pipek-Mezey orbital localization -> non-aufbau SCF with MOM |
| **SCF Helpers** | `kshelper.py` | From PSIXAS: Provides `diag_H` (Fock diagonalization), `Timer` (timing), `ACDIIS` (ADIIS+CDIIS extrapolation), `printHeader` (formatted output) |
| **Scattering** | `dsigma.py` | `crosssection` class for computing differential cross section: reads molden files via ORBKIT, computes electron density, numerically integrates the molecular form factor with cubature, applies Thomson cross-section |
| **Plotting** | `plotting.py` | `plot_detector` class: `log_data()` for log-scale detector images, `method_diff()` for method-comparison plots |

### Wavefunction Pipeline

The wavefunction module (`wf.py`) performs three sequential steps:

1. **`ground_state(dft, func)`** — Runs a standard Psi4 SCF (HF or DFT) calculation on the neutral molecule. Saves MO coefficients to `.npz` and writes a `.molden` file.

2. **`localize(wfn, loc_sub, dft, func)`** — Applies Pipek-Mezey localization to a user-specified subset of orbitals (e.g. core 1s orbitals), producing atom-centered orbitals necessary for site-specific core-hole creation.

3. **`non_aufbau_state(dft, func, mol, scf_wfn, **options)`** — Runs a custom UHF/UKS SCF with modified orbital occupations (e.g. removing an electron from a specific core orbital). Uses the Maximum Overlap Method (MOM) to track orbitals across iterations and prevent variational collapse to the ground state. Supports optional orbital freezing and DIIS/damping convergence acceleration.

## Dependencies

| Dependency | Purpose |
|------------|---------|
| **Python** ≥ 3.6 | Runtime |
| **Psi4** | Quantum chemistry engine (SCF, integrals, basis sets) |
| **Psi4NumPy** | NumPy interface to Psi4 |
| **NumPy** | Array operations |
| **ORBKIT** | Reads `.molden` files, computes electron density on grids |
| **cubature** | Adaptive numerical integration (molecular form factor) |
| **Matplotlib** | Plotting detector images |
| **SciPy** | ADIIS optimization (used internally by `kshelper.py`) |

### Installing Dependencies

**Psi4** (via conda):
```bash
conda install psi4 -c psi4
```

**ORBKIT**:
```bash
pip install orbkit
```
or from source: [https://github.com/orbkit/orbkit](https://github.com/orbkit/orbkit)

**cubature**:
```bash
pip install cubature
```

**Other Python packages**:
```bash
pip install numpy matplotlib scipy
```

---

## Installation

No installation step is required. Clone the repository and run the scripts directly from within their respective directories:

```bash
git clone https://github.com/afouda11/X-RHO-SCATTER.git
cd X-RHO-SCATTER
```

> **Note**: Scripts must be run from within their directory (`wavefunction/` or `scattering/`) because they use relative paths for file I/O.

---

## Running Simulations

### Step 1: Generate Wavefunctions

```bash
cd wavefunction
python run_wf.py
```

This runs the three-step pipeline (ground state → localization → core-hole SCF) and writes `.molden` and `.npz` files to `wf_outfiles/`.

### Step 2: Compute Scattering Cross-Sections

Copy or symlink the relevant `.molden` files into `scattering/molden_files/` with the naming convention `<state>_<method>.molden`:

```bash
cd ../scattering
mkdir -p molden_files
cp ../wavefunction/wf_outfiles/Neutral_HF.molden molden_files/Neutral_HF.molden
cp ../wavefunction/wf_outfiles/Core_Hole_HF.molden molden_files/C3_HF.molden
```

Then run the scattering calculation:

```bash
python run_dsigma.py
```

Output data files are written to `data_files/`.

### Step 3: Plot Results

```bash
mkdir -p plots
python run_plotting.py
```

Detector images are saved to `plots/`.

---

## Input File Reference

### `run_wf.py` — Wavefunction Driver

The driver script serves as the input file. Key user-configurable sections:

#### Molecular Geometry

Defined using `psi4.geometry()`. The molecule must be:
- Centered at the origin (`nocom`)
- Fixed in space (`noreorient`)
- In C1 symmetry (`symmetry C1`)

```python
mol = psi4.geometry("""
    C   1.4225   0.0668  -0.1118
    H   2.5020   0.1902  -0.1154
    ...
    symmetry C1
    noreorient
    nocom
""")
```

#### Psi4 SCF Options

```python
psi4.set_options({'basis'         : '6-311+G*',
                  'reference'     : 'uhf',
                  'e_convergence' : 1e-8,
                  'd_convergence' : 1e-8})
psi4.set_num_threads(4)
```

#### Method Selection

```python
dft  = False      # True for DFT, False for HF
func = 'b3lyp'    # DFT functional (ignored when dft=False)
```

#### Non-Aufbau SCF Options

| Key | Type | Description |
|-----|------|-------------|
| `E_CONV` | float | Energy convergence threshold (Ha) |
| `D_CONV` | float | Density convergence threshold |
| `GAMMA` | float | Damping factor (0–1): fraction of previous Fock matrix mixed in |
| `DIIS_EPS` | float | DIIS error threshold to switch from damping to DIIS |
| `VSHIFT` | float | Virtual orbital level shift (Ha); aids convergence, removed once converged |
| `MAXITER` | int | Maximum SCF iterations |
| `DIIS_LEN` | int | Maximum number of DIIS vectors |
| `DIIS_MODE` | string | `"ADIIS+CDIIS"` or `"CDIIS"` |
| `MIXMODE` | string | Initial mixing mode: `"DAMP"` (switches to DIIS automatically) |
| `LOC_SUB` | list[int] | Orbital indices to localize (e.g. core orbitals) |
| `ORBS` | list[int] | Orbital indices to modify occupation |
| `OCCS` | list[float] | New occupation for each orbital in `ORBS` |
| `FREEZE` | list[str] | `"T"` or `"F"` — freeze orbital from rotating during SCF |
| `SPIN` | list[str] | `"a"` or `"b"` — spin channel for each orbital |
| `OVL` | list[str] | `"T"` or `"F"` — use maximum overlap tracking (MOM) |

> **Note**: `ORBS`, `OCCS`, `FREEZE`, `SPIN`, and `OVL` must all have the same length. They define a 1-to-1 mapping for each orbital to be modified.

### `run_dsigma.py` — Scattering Driver

| Variable | Type | Description |
|----------|------|-------------|
| `e_dict` | dict | Photon energies: `{"label": energy_eV, ...}` |
| `state_list` | list[str] | State labels (must match molden filenames) |
| `method_list` | list[str] | Method labels (must match molden filenames) |
| `Theta` | array | Scattering polar angles (radians) |
| `Phi` | array | Scattering azimuthal angles (radians) |
| `nprocs` | int | Number of parallel processes for the form factor calculation |
| `precision` | float | Absolute and relative error tolerance for cubature integration |
| `extent` | float | Half-width of the real-space integration box (Bohr) |

#### Molden File Naming Convention

The scattering code expects molden files in `molden_files/` named:

```
<state>_<method>.molden
```

For example, with `state_list = ["Neutral", "C3"]` and `method_list = ["HF", "DFT"]`:

```
molden_files/Neutral_HF.molden
molden_files/Neutral_DFT.molden
molden_files/C3_HF.molden
molden_files/C3_DFT.molden
```

### `run_plotting.py` — Plotting Driver

| Variable | Type | Description |
|----------|------|-------------|
| `energy` | list[str] | Energy labels (must match keys in `e_dict` from `run_dsigma.py`) |
| `state` | dict | `{"state_label": n_electrons, ...}` — used for normalization |
| `method` | list[str] | Method labels |
| `normalise` | bool | Normalize scattering data to N² (number of electrons squared) |
| `method_diff` | bool | Plot difference between two methods (method[1] − method[0]) |

---

## Output Files

### Wavefunction Output (`wf_outfiles/`)

| File | Description |
|------|-------------|
| `Neutral_HF.molden` | Ground-state HF orbitals in Molden format |
| `Neutral_HF_loc.molden` | Localized ground-state HF orbitals |
| `Core_Hole_HF.molden` | Core-hole state HF orbitals |
| `hf_gsorbs.npz` | Ground-state MO coefficients, occupations, orbital energies (NumPy archive) |
| `hf_loc_gsorbs.npz` | Localized MO coefficients and occupations |
| `Neutral_DFT.molden` | Ground-state DFT orbitals (when `dft=True`) |
| `Neutral_DFT_loc.molden` | Localized ground-state DFT orbitals |
| `Core_Hole_DFT.molden` | Core-hole state DFT orbitals |
| `dft_gsorbs.npz` | DFT ground-state orbital data |
| `dft_loc_gsorbs.npz` | DFT localized orbital data |

### Scattering Output (`data_files/`)

| File Pattern | Content |
|--------------|---------|
| `<state>_<method>_<energy>Kev.txt` | Three columns: `Theta  Phi  dsigma/dOmega` for each angle pair |
| `xy.txt` | Angular grid: `Theta  Phi` pairs (used by plotting code) |

### Plots (`plots/`)

| File Pattern | Content |
|--------------|---------|
| `log_data_<method>_detector.png` | Log-scale 2D detector image for each method |
| `method_diff_<label>_detector.png` | Method-difference detector image |

---

## Example Workflow

This example computes the coherent x-ray scattering pattern from a C 1s core-hole state of 1,3-cyclohexadiene at the Hartree-Fock level.

### 1. Generate the core-hole wavefunction

Edit `wavefunction/run_wf.py` to set your molecular geometry, basis set, and core-hole orbital:

```python
# Create a core hole on the 4th localized orbital (carbon 1s), beta spin
options = {
    ...
    "LOC_SUB"  : [0, 1, 2, 3, 4, 5],   # Localize the 6 carbon 1s orbitals
    "ORBS"     : [3],                    # Modify orbital index 3
    "OCCS"     : [0.0],                  # Set occupation to 0 (remove electron)
    "FREEZE"   : ["T"],                  # Freeze orbital from rotating
    "SPIN"     : ["b"],                  # Beta spin channel
    "OVL"      : ["T"]                   # Track via maximum overlap
}

dft  = False       # Use Hartree-Fock
```

Run:

```bash
cd wavefunction
python run_wf.py
```

### 2. Compute scattering cross-sections

Copy molden files and edit `scattering/run_dsigma.py`:

```python
e_dict      = {"9": 9000}                    # 9 keV photon energy
state_list  = ["Neutral", "CoreHole"]
method_list = ["HF"]

Theta     = np.linspace(0, (80 * np.pi/180), 100)   # Polar angles
Phi       = np.linspace(0, (2 * np.pi), 100)         # Azimuthal angles
nprocs    = 8
precision = 1e-3
extent    = 9.0       # Integration box half-width (Bohr)
```

Run:

```bash
cd scattering
python run_dsigma.py
```

### 3. Plot detector images

```bash
python run_plotting.py
```

Output images are saved to `plots/`.

---
