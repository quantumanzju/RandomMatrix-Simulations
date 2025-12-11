# RandomMatrix-Simulations
Reed–Muller based phase-encoded exact and approximate computation of matrix permanents
# Reed–Muller Based Phase Encoding for Permanent Reconstruction

This repository contains code and numerical experiments for reconstructing
matrix permanents using Reed–Muller (RM) based phase encoding and
Laplace-based approximations. It implements the experiments described in
Sections 3–4 of:

> Jian Fu, “Phase-Encoded Exact and Approximate Computation of Matrix
> Permanents: A Sequence-Algebraic Perspective”, arXiv:submit/7070346.

In brief, the code supports:

- exact permanent computation via **Ryser’s formula**,
- RM-based **encoded permanent** reconstruction,
- a **fast Laplace / polynomial approximation**, and
- numerical comparisons in several dimensions.

> **Practical note**  
> Several scripts rely on randomized initializations and may need to be
> run multiple times to obtain a good reconstruction (acceptable error).
> This is expected: in our experiments we typically repeat the script
> several times and keep the best (or first successful) run.

---

## Repository Structure

The repository is organized as follows (folder names are indicative; adapt
if your layout is slightly different):

```text
.
├─ src/                  # Python source code
│  ├─ Exact_RM_FixedSequence.py
│  ├─ Exact_RM_AnySequence.py
│  ├─ ApproxPerm_LaplacePoly.py
│  └─ (other Python scripts)
├─ matlab/               # MATLAB scripts (if applicable)
│  ├─ Exact_RM_FixedSequence.m
│  ├─ Exact_RM_AnySequence.m
│  └─ ApproxPerm_LaplacePoly.m
├─ data/                 # Text data (intermediate or auxiliary results)
│  ├─ dim11.txt
│  ├─ dim15.txt
│  └─ dim20.txt
├─ results/              # Final numerical results and figures (outputs)
│  ├─ dim11_dis.bmp
│  ├─ dim11_cor.bmp
│  ├─ dim11_err.bmp
│  ├─ dim15_dis.bmp
│  ├─ dim15_cor.bmp
│  ├─ dim15_err.bmp
│  ├─ dim20_dis.bmp
│  ├─ dim20_cor.bmp
│  └─ dim20_err.bmp
├─ requirements.txt      # Python dependencies
├─ LICENSE               # Open-source license (MIT)
└─ README.md
```

---

## Main Scripts

### `Exact_RM_AnySequence.py`

Demonstration script: Reed–Muller based phase encoding and
single-parameter reconstruction of a matrix permanent.

This script:

1. Generates a q-ary order-1 Reed–Muller code RM_q(1, m);
2. Uses RM codewords as phase-encoding patterns;
3. Recovers modular phase exponents by solving a constrained linear
   system over Z_q;
4. Applies column-wise phase perturbations to the matrix;
5. Reconstructs the permanent from a 1-parameter linear model and
   compares it with the exact Ryser permanent.

**Randomness**

This script uses random choices (e.g. codewords, initializations) and the
reconstruction problem is non-convex. A single run may not always achieve
good accuracy. In practice:

- run the script multiple times;
- record the reconstruction error in each run;
- keep the run with the smallest error.

### `Exact_RM_FixedSequence.py`

Same reconstruction idea as above, but with a **fixed sequence** of phase
patterns. This serves primarily as a controlled baseline for comparison
with the “any sequence” case.

It also uses randomized components and may need multiple runs to obtain a
good reconstruction.

### `ApproxPerm_LaplacePoly.py`

Implements a **fast approximation** of the permanent using encoded
statistics combined with Laplace expansion and polynomial calibration.

This script:

- computes the exact permanent using Ryser’s formula for a set of
  sample matrices;
- computes the encoded statistic and Laplace-based approximation for
  the same samples;
- fits a low-degree polynomial (typically quadratic) that maps the
  encoded statistic r to an estimate rho(r) of the permanent;
- evaluates accuracy via error statistics and correlation measures.

It is used to compare:

- exact Ryser permanents;
- RM-based encoded reconstruction;
- fast Laplace / polynomial approximations.

---

## Data and Outputs

### Input

The primary logical input is the **matrix** whose permanent is to be
approximated. Depending on the script:

- matrices may be **generated internally** (e.g. random ensembles), or  
- loaded from text files such as `dim11.txt`, `dim15.txt`, `dim20.txt`.

If you use the `.txt` files as inputs, please check the script headers
for the precise format (dimension, number of samples, etc.). Typically,
each file corresponds to a fixed dimension (11, 15, or 20).

### Output

The main outputs are numerical results and figures.

- **Text outputs**  
  Intermediate numerical results may be written to `.txt` files in
  `data/` or elsewhere (depending on how the scripts are configured).

- **Figures** (stored in `results/`)  

  For each dimension N {11, 15, 20} we typically produce:

  - `*_dis.bmp` – distribution plots  
    (e.g. histograms of reconstructed vs exact permanents, or error
    distributions);
  - `*_cor.bmp` – correlation plots  
    (approximate vs exact permanents, often with a fitted line);
  - `*_err.bmp` – error plots  
    (reconstruction error vs sample index or parameter).

Concretely:

- `dim11_dis.bmp`, `dim11_cor.bmp`, `dim11_err.bmp` – results for (N=11);  
- `dim15_dis.bmp`, `dim15_cor.bmp`, `dim15_err.bmp` – results for (N=15);  
- `dim20_dis.bmp`, `dim20_cor.bmp`, `dim20_err.bmp` – results for (N=20).

These figures should be reproducible by running the corresponding
scripts.

---

## Example Console Output (N = 20)

A typical run of the encoded permanent + Laplace approximation for
dimension N = 20 with 100 samples produces output similar to:

```text
Computing true permanent (Ryser): sample 10 / 100
...
Ryser exact computation time: 850.9701 seconds

Encoded permanent + Laplace: sample 10 / 100
...
Fast encoded approximation time: 0.8098 seconds

Dimension N = 20, calibrated coefficients (quadratic in r):
  For A   :  rho ~ 222.455356 * r^2 + -464.280948 * r + 242.826342
  For A^T :  rho ~ 222.455356 * r^2 + -464.280948 * r + 242.826342
Combine A and A^T (simple average)

=== Error statistics for combined calibrated estimator (A & A^T) ===
RMSE      = 0.1391 %
MAE       = 0.1059 %
MaxAE     = 0.5296 %
Std(error)= 0.1146 %

--- Correlation and R^2 between r and rho (N = 20) ---
corr(r(A),  rho(A))      = -0.995322
corr(r(A^T),rho(A^T))    = -0.998376
R^2 for quadratic fit (A)   = 0.998691
R^2 for quadratic fit (A^T) = 0.999064
```

This illustrates that, in this configuration:

- exact Ryser computation for 100 samples at N = 20 takes about
  **851 seconds**;
- the encoded + Laplace approximation for the same samples takes about
  **0.81 seconds**;
- the calibrated estimator achieves **sub-percent relative error**
  with R^2 approx 0.999 for the quadratic fit.

---

## Requirements

### Python

- Python 3.8 or newer.
- Recommended: use a virtual environment (`venv` or `conda`).

Install the required packages with:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` will typically include:

```text
numpy
scipy
matplotlib
```

Please extend this file according to the actual imports used in your code.

### MATLAB (optional)

To run the MATLAB versions of the scripts, you need a working MATLAB
installation. The code should only require core linear-algebra
functionality; no special toolboxes are expected.

---

## How to Run

Below are example commands. Adjust arguments and paths to match your
actual script interface.

### 1. Clone the repository

```bash
git clone https://github.com/quantumanzju/RandomMatrix-Simulations.git
cd RandomMatrix-Simulations
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run exact RM-based reconstruction (any sequence)

```bash
python src/Exact_RM_AnySequence.py \
    --dim 20 \
    --input data/dim20.txt \
    --output results/
```

### 4. Run exact RM-based reconstruction (fixed sequence)

```bash
python src/Exact_RM_FixedSequence.py \
    --dim 20 \
    --input data/dim20.txt \
    --output results/
```

### 5. Run encoded permanent + Laplace approximation

```bash
python src/ApproxPerm_LaplacePoly.py \
    --dim 20 \
    --input data/dim20.txt \
    --output results/
```

> **If your scripts do not yet support command-line arguments**  
> (for example, all parameters are hard-coded at the top of each file),
> replace the above with instructions such as:
>
> 1. Open `src/Exact_RM_AnySequence.py` in a text editor.  
> 2. Set the dimension \(N\), number of samples, and paths (e.g. to
>    `data/` and `results/`) in the configuration section at the top of
>    the file.  
> 3. Run:
>    ```bash
>    python src/Exact_RM_AnySequence.py
>    ```
> 4. Repeat analogously for the other scripts.

---

## Reproducibility and Randomness

- The reconstruction algorithms involve random codewords and/or random
  initializations.
- Different runs with the same configuration may produce **slightly
  different** outputs and errors.
- To reproduce a specific run, you should:
  - fix the random seeds (e.g. `numpy.random.seed` in Python, `rng`
    in MATLAB);
  - record all relevant parameters (dimension, number of samples,
    code parameters, etc.).

In our experiments we typically:

1. run each configuration multiple times;  
2. compute error metrics (RMSE, MAE, MaxAE) for each run;  
3. keep either the best run (smallest error) or report aggregated
   statistics over several runs.

---

## License

This project is released under the [MIT License](./LICENSE).  
You are free to use, modify, and redistribute the code, provided that the
original copyright notice is retained.

---

## Citation

If you use this code or the accompanying results in your research, please
cite the paper and optionally this repository, for example:


J. Fu, "Phase-Encoded Exact and Approximate Computation of Matrix Permanents:
A Sequence-Algebraic Perspective", arXiv:submit/7070346.
Code available at: https://github.com/quantumanzju/RandomMatrix-Simulations

