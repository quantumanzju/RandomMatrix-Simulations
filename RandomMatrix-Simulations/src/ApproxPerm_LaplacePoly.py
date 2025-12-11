
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt

"""
%% ApproxPerm_LaplacePoly.py
% Polynomial-time approximation of matrix permanents with Laplace calibration.
%
% This script implements the dimension-aware polynomial approximation
% described in Section~\ref{sec:experiments} (Laplace-calibrated estimator).
% For a fixed matrix dimension N, we:
%   1) draw a number of random test matrices A;
%   2) compute the ground-truth permanents perm(A) by an exact method
%      (Ryser or the exact phase-encoding solver);
%   3) evaluate a low-cost encoded statistic r(A) (and optionally r(A^T));
%   4) approximate perm(A) by a Laplace-type ansatz whose coefficients
%      depend polynomially on N and are pre-trained offline.
%
% In this script:
%   - numdim    : matrix dimension N;
%   - numsample : number of random samples used for testing;
%   - A{i}      : i-th N×N test matrix;
%   - true_perm(i) : ground-truth permanent of A{i};
%   - Penc0(i), Penc1(i) : encoded estimates for A{i} and A{i}^T
%                          (see the main text for the exact form);
%   - r0(i), r1(i)       : characteristic ratios extracted from Penc0, Penc1;
%   - coff               : pre-trained global coefficients for the
%                          dimension-aware Laplace polynomial model.
%
% The 3×2 matrix `coff` stores the N-dependent coefficients [a_N, b_N, c_N]
% of the calibrated approximation model. Each row corresponds to one
% coefficient, parameterized as a polynomial in N:
%
%   a_N = coff(1,1)*N + coff(1,2),   etc.
%
% These coefficients are obtained once in an offline training phase
% (by fitting to exact permanents over a range of dimensions and random
% matrices) and then reused here to provide a fast, purely polynomial-time
% estimator for new test instances.
"""


# ======================= Main script =======================

def main():
    # ----------------- Parameters -----------------
    numdim = 15      # matrix dimension N
    numsample = 100  # number of random samples

    A_list = [None] * numsample
    true_perm = np.zeros(numsample, dtype=float)  # ground-truth permanents
    Penc0 = np.zeros(numsample, dtype=float)      # encoded estimate for A
    Penc1 = np.zeros(numsample, dtype=float)      # encoded estimate for A^T
    r0 = np.zeros(numsample, dtype=float)         # characteristic ratio r(A)
    r1 = np.zeros(numsample, dtype=float)         # characteristic ratio r(A^T)

    # Dimension-aware global coefficients (trained offline)
    # Each row is a polynomial in N; three rows correspond to [a_N, b_N, c_N]
    coff = np.array([
        [15.291014416532047, -83.364932630744100],
        [-31.598207421729818, 167.6832006771538],
        [16.307121673277322, -83.316091487811010]
    ], dtype=float)

    # ----------------- Exact permanents (Ryser) -----------------
    t0 = time.time()
    for m in range(numsample):
        if (m + 1) % 10 == 0:
            print(f'Computing true permanent (Ryser): sample {m+1} / {numsample}')
        # In MATLAB: rng('shuffle'); here we simply rely on NumPy's RNG
        A_list[m] = np.random.rand(numdim, numdim)
        true_perm[m] = matrix_permanent(A_list[m])
    elapsedTime_exact = time.time() - t0
    print(f'Ryser exact computation time: {elapsedTime_exact:.4f} seconds')

    # ----------------- Encoded permanent + Laplace features -----------------
    t1 = time.time()
    for m in range(numsample):
        if (m + 1) % 10 == 0:
            print(f'Encoded permanent + Laplace: sample {m+1} / {numsample}')

        # Original matrix A
        tmp_Penc0, tmp_Perm_LP0 = matrix_perm(A_list[m])
        Penc0[m] = tmp_Penc0.real
        r0[m] = np.mean(tmp_Perm_LP0 / tmp_Penc0)

        # Transposed matrix A^T
        tmp_Penc1, tmp_Perm_LP1 = matrix_perm(A_list[m].T)
        Penc1[m] = tmp_Penc1.real
        r1[m] = np.mean(tmp_Perm_LP1 / tmp_Penc1)
    elapsedTime_approx = time.time() - t1
    print(f'Fast encoded approximation time: {elapsedTime_approx:.4f} seconds')

    # ----------------- Ratios rho = P_enc / perm -----------------
    rho0 = Penc0 / true_perm  # for A
    rho1 = Penc1 / true_perm  # for A^T (perm(A^T) = perm(A))

    # ----------------- Dimension-aware quadratic calibration g_N(r) -----------------
    # Corresponds to MATLAB polyval(coff(k,:), numdim)
    p0 = np.zeros(3)
    p1 = np.zeros(3)
    for k in range(3):
        p0[k] = np.polyval(coff[k, :], numdim)
        p1[k] = np.polyval(coff[k, :], numdim)

    a0, b0, c0 = p0
    a1, b1, c1 = p1

    print(f'Dimension N = {numdim}, calibrated coefficients (quadratic in r):')
    print(f'  For A   :  rho ~ {a0:.6f} * r^2 + {b0:.6f} * r + {c0:.6f}')
    print(f'  For A^T :  rho ~ {a1:.6f} * r^2 + {b1:.6f} * r + {c1:.6f}')

    # Calibration functions g_N(r) for each sample
    g0 = a0 * r0**2 + b0 * r0 + c0   # g_N(r(A))
    g1 = a1 * r1**2 + b1 * r1 + c1   # g_N(r(A^T))

    # Avoid extremely small denominators
    eps_floor = 1e-8
    mask0 = np.abs(g0) < eps_floor
    g0[mask0] = eps_floor * np.sign(g0[mask0] + eps_floor)

    mask1 = np.abs(g1) < eps_floor
    g1[mask1] = eps_floor * np.sign(g1[mask1] + eps_floor)

    # One-sided calibrated estimators
    Perm0_pred = Penc0 / g0  # calibrated estimate using A
    Perm1_pred = Penc1 / g1  # calibrated estimate using A^T

    # Combined estimator (A and A^T)
    print('Combine A and A^T (simple average)')
    Perm_pred = (Perm0_pred + Perm1_pred) / 2

    # ----------------- Error statistics -----------------
    # Relative error in percentage
    err_raw0 = 100 * (Penc0 / true_perm - 1)       # uncalibrated (A)
    err_raw1 = 100 * (Penc1 / true_perm - 1)       # uncalibrated (A^T)
    err_cal0 = 100 * (Perm0_pred / true_perm - 1)  # calibrated (A)
    err_cal1 = 100 * (Perm1_pred / true_perm - 1)  # calibrated (A^T)
    err_comb = 100 * (Perm_pred / true_perm - 1)   # calibrated combined

    # Focus on combined calibrated estimator
    error = err_comb

    RMSE = np.sqrt(np.mean(error**2))
    MAE = np.mean(np.abs(error))
    MaxAE = np.max(np.abs(error))
    StdErr = np.std(error)

    print('=== Error statistics for combined calibrated estimator (A & A^T) ===')
    print(f'RMSE      = {RMSE:.4f} %')
    print(f'MAE       = {MAE:.4f} %')
    print(f'MaxAE     = {MaxAE:.4f} %')
    print(f'Std(error)= {StdErr:.4f} %')

    # ----------------- Correlation and R^2 for rho vs r -----------------
    # Pearson correlation between r and rho
    corr_r_rho_A = np.corrcoef(r0, rho0)[0, 1]
    corr_r_rho_AT = np.corrcoef(r1, rho1)[0, 1]

    # R^2 for quadratic fit rho ~ g_N(r) on A
    rho_fit0 = a0 * r0**2 + b0 * r0 + c0
    SS_res0 = np.sum((rho0 - rho_fit0) ** 2)
    SS_tot0 = np.sum((rho0 - np.mean(rho0)) ** 2)
    R2_A = 1 - SS_res0 / SS_tot0

    # R^2 for quadratic fit on A^T
    rho_fit1 = a1 * r1**2 + b1 * r1 + c1
    SS_res1 = np.sum((rho1 - rho_fit1) ** 2)
    SS_tot1 = np.sum((rho1 - np.mean(rho1)) ** 2)
    R2_AT = 1 - SS_res1 / SS_tot1

    print('--- Correlation and R^2 between r and rho (N = {0}) ---'.format(numdim))
    print(f'corr(r(A),  rho(A))      = {corr_r_rho_A:.6f}')
    print(f'corr(r(A^T),rho(A^T))    = {corr_r_rho_AT:.6f}')
    print(f'R^2 for quadratic fit (A)   = {R2_A:.6f}')
    print(f'R^2 for quadratic fit (A^T) = {R2_AT:.6f}')

    # ----------------- Figures for the paper (English labels) -----------------

    # 1) rho vs r scatter + fitted curve for A
    plt.figure()
    plt.scatter(r0, rho0, s=40)
    r_line = np.linspace(np.min(r0), np.max(r0), 200)
    rho_fit = a0 * r_line**2 + b0 * r_line + c0
    plt.plot(r_line, rho_fit, 'r', linewidth=1.5)
    plt.xlabel('Laplace characteristic ratio r(A)', fontsize=12)
    plt.ylabel(r'Encoding-truth ratio $\rho(A) = P_{enc}(A) / \mathrm{perm}(A)$',
               fontsize=12)
    plt.title(
        f'N = {numdim}: r(A) vs rho(A)\n'
        f'corr = {corr_r_rho_A:.4f},  R^2 (quadratic) = {R2_A:.4f}',
        fontsize=12
    )
    plt.legend([f'Samples (N = {numdim})', 'Quadratic fit $g_N(r)$'],
               loc='best')
    plt.grid(True)
    plt.gca().tick_params(labelsize=12)

    # 2) Relative error vs sample index (combined calibrated)
    plt.figure()
    plt.plot(np.arange(1, numsample + 1), error, 'o-', linewidth=1.2, markersize=5)
    plt.xlabel('Sample index', fontsize=12)
    plt.ylabel('Relative error (%)', fontsize=12)
    plt.title(
        f'Combined calibrated estimator (A and A^T), N = {numdim}\n'
        f'RMSE = {RMSE:.3f}%, MaxAE = {MaxAE:.3f}%',
        fontsize=12
    )
    plt.grid(True)
    plt.gca().tick_params(labelsize=12)

    # 3) Histogram of relative errors (combined calibrated)
    plt.figure()
    plt.hist(error, bins=20, facecolor=(0.2, 0.6, 0.8), edgecolor='k')
    plt.xlabel('Relative error (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(
        f'Error distribution of combined calibrated estimator, N = {numdim}\n'
        f'RMSE = {RMSE:.3f}%, MAE = {MAE:.3f}%',
        fontsize=12
    )
    plt.grid(True)
    plt.gca().tick_params(labelsize=12)

    # 4) Compare uncalibrated vs calibrated errors (boxplot)
    plt.figure()
    data_for_box = np.column_stack((err_raw0, err_cal0, err_comb))
    plt.boxplot(data_for_box, labels=['Raw(A)', 'Calibrated(A)', 'Calibrated combined'])
    plt.ylabel('Relative error (%)', fontsize=12)
    plt.title(f'Comparison of different estimators, N = {numdim}', fontsize=12)
    plt.grid(True)
    plt.gca().tick_params(labelsize=12)

    plt.show()


# ======================= Supporting functions =======================

def matrix_perm(matrix: np.ndarray):
    """
    Python equivalent of the MATLAB function matrix_perm(matrix).

    Parameters
    ----------
    matrix : np.ndarray
        Input square matrix.

    Returns
    -------
    Perm : float
        Encoded permanent estimate.
    Perm_Laplace : np.ndarray
        Laplace-type row-wise permanent estimates (length = matrix dimension).
    """
    numdim = matrix.shape[0]
    q = numdim + 1
    m = 1
    seq_size = q ** m

    x1 = qary_rm_linear(q, m)

    # MATLAB: rng(133);
    rng = np.random.default_rng(133)
    sel_seq = random_select(seq_size, numdim, rng)
    signMatrix_sub = x1[sel_seq, :]

    omega = np.exp(2j * np.pi / q)  # q-ary complex root of unity
    signMatrix_sub = omega ** signMatrix_sub.astype(complex)

    Perm = matrix_permS(matrix, signMatrix_sub)
    Perm_Laplace = matrix_perPL(matrix, signMatrix_sub)

    return Perm, Perm_Laplace


def matrix_perPL(matrix: np.ndarray, signMatrix_sub: np.ndarray):
    """
    Python equivalent of the MATLAB function matrix_perPL(matrix, signMatrix_sub).

    This computes a Laplace expansion using the encoded permanent subroutine
    for each minor, returning a vector of size N containing row-wise
    contributions of the Laplace-type permanent decomposition.
    """
    k = matrix.shape[0]
    Perm_Laplace = np.zeros(k, dtype=complex)

    # Precompute all column index combinations (to avoid repeated setdiff calls)
    cols_index = [np.concatenate((np.arange(0, col), np.arange(col + 1, k)))
                  for col in range(k)]

    # Precompute, for each column, the corresponding sub sign matrix
    precomputed_sign_sub = []
    all_rows = np.arange(k)
    for col in range(k):
        mask_rows = all_rows != col
        precomputed_sign_sub.append(signMatrix_sub[mask_rows, :])

    all_rows = np.arange(k)
    for row in range(k):
        row_sub_index = all_rows != row
        row_sub_matrix = matrix[row_sub_index, :]  # (k-1) × k

        for col in range(k):
            col_sub_index = cols_index[col]
            sub_matrix = row_sub_matrix[:, col_sub_index]
            sub_signMatrix_sub = precomputed_sign_sub[col]
            sub_Perm = matrix_permS(sub_matrix, sub_signMatrix_sub)

            Perm_Laplace[row] += matrix[row, col] * sub_Perm

    return Perm_Laplace


def matrix_permS(matrix: np.ndarray, signMatrix_sub: np.ndarray):
    """
    Python equivalent of the MATLAB function matrix_permS(matrix, signMatrix_sub).

    Encoded permanent estimator via correlation with q-ary phase patterns.
    """
    dim = matrix.shape[0]
    numbit = signMatrix_sub.shape[1]

    # Parallel computations
    column_sums = np.sum(signMatrix_sub, axis=0)   # 1 × M
    verify_JC = column_sums ** dim

    S = matrix @ signMatrix_sub                    # N × M
    lams = np.prod(signMatrix_sub, axis=0)         # 1 × M
    JC = np.prod(S, axis=0)                        # 1 × M

    correlation = JC @ np.conjugate(lams)
    verify_correlation = verify_JC @ np.conjugate(lams)

    verify = verify_correlation / (numbit * np.math.factorial(dim))
    Perm = correlation / numbit
    Perm = np.real(Perm / verify)  # keep the real part, as in MATLAB
    return Perm


def matrix_permanent(A: np.ndarray) -> float:
    """
    Exact matrix permanent using Ryser's formula.

    Parameters
    ----------
    A : np.ndarray
        Square matrix (n x n).

    Returns
    -------
    float
        Permanent of A.

    Notes
    -----
    Complexity is exponential in n; suitable only for small dimensions.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")

    if n == 0:
        return 1.0  # By convention, the permanent of an empty matrix is 1

    # If A has a zero row or zero column, the permanent is 0
    if np.any(np.all(A == 0, axis=0)) or np.any(np.all(A == 0, axis=1)):
        return 0.0

    total = 0.0
    idx_list = list(range(n))

    # Loop over all non-empty subsets S ⊆ {1, ..., n}
    for k in range(1, n + 1):
        for subset in itertools.combinations(idx_list, k):
            S = list(subset)
            row_sums = np.sum(A[:, S], axis=1)
            prod_val = np.prod(row_sums)
            sign_val = (-1) ** k
            total += sign_val * prod_val

    perm = ((-1) ** n) * total
    return float(perm)


def qary_rm_linear(q: int, m: int):
    """
    Generate all q-ary linear Reed–Muller codewords of order m over Z_q.

    Parameters
    ----------
    q : int
        Alphabet size (q-ary).
    m : int
        Dimension parameter.

    Returns
    -------
    codewords : np.ndarray
        Array of shape (q^m, q^m) containing all linear functions evaluated
        on all q^m input vectors over Z_q^m.
    """
    qm = q ** m

    # Generate all input vectors x ∈ Z_q^m
    inputs = np.zeros((qm, m), dtype=int)
    for i in range(qm):
        idx = i
        for j in range(m - 1, -1, -1):
            inputs[i, j] = (idx // (q ** j)) % q

    # Generate all coefficient vectors a ∈ Z_q^m
    coeffs = np.zeros((qm, m), dtype=int)
    for i in range(qm):
        idx = i
        for j in range(m - 1, -1, -1):
            coeffs[i, j] = (idx // (q ** j)) % q

    # Evaluate all linear functions a · x over Z_q
    codewords = np.mod(coeffs @ inputs.T, q)
    return codewords


def random_select(N: int, k: int, rng=None):
    """
    Randomly select k distinct indices from {0, ..., N-1}.

    Parameters
    ----------
    N : int
        Upper bound (number of available indices).
    k : int
        Number of indices to select.
    rng : np.random.Generator, optional
        NumPy random number generator. If None, a new generator is created.

    Returns
    -------
    selected : np.ndarray
        1D array of length k with distinct indices in [0, N-1].
    """
    if k > N:
        raise ValueError("k cannot be larger than N.")
    if rng is None:
        rng = np.random.default_rng()
    perm = rng.permutation(N)
    selected = perm[:k]
    return selected


if __name__ == "__main__":
    main()