import numpy as np
import itertools

"""
%% Exact_RM_FixedSequence.py
% Fixed-sequence Reed–Muller based permanent reconstruction (dimension-specific).
%
% This script implements a dimension-specific version of the phase-encoding
% scheme. For a fixed matrix size N = numPorts, we:
%   1. Generate a q-ary order-1 Reed–Muller code RM_q(1,m);
%   2. Use a *fixed* set of RM codewords (sel_seq) as phase patterns;
%   3. Use a *pre-computed* coefficient matrix coeff1, obtained once
%      by solving the modular phase-matching problem;
%   4. For any new N×N matrix A, we:
%        - encode A using these fixed phase patterns;
%        - apply column-wise π phase shifts to obtain perm_c;
%        - solve a small linear system to reconstruct the permanent.
%
% Compared to the general version (which solves for phase_order via
% ultra_fast_match_mod4 at each run), this fixed-sequence version
% dispenses with computing phase_order once a good sequence and coefficients
% (sel_seq, coeff1) have been found for the given dimension."""

# ======================= Main script =======================

def main():
    # ----- Parameters -----
    q = 2
    m = 15
    numPorts = 19

    # ----- Generate q-ary RM linear code -----
    x1 = qary_rm_linear(q, m)
    lengths = x1.shape[0]

    # ----- Test matrix and exact permanent (Ryser) -----
    # 为了更贴近 MATLAB rng('shuffle')，这里直接用默认 RNG，
    # 如果你在 MATLAB 中固定了 rng(seed)，可以在这里也用同一个 seed。
    A = np.random.rand(numPorts, numPorts)
    perm_correct = matrix_permanent(A)

    # ----- Fixed RM codeword indices (MATLAB 1-based) -----
    sel_seq_matlab = np.array([
        6755, 19312, 15193, 15759, 16959, 17056, 17890, 31004,
        1498,  6923, 18096, 28749,  2542,   261, 20519, 21946,
        6126, 28166, 21342
    ], dtype=int)

    if sel_seq_matlab.size != numPorts:
        raise ValueError(
            f"Length of sel_seq ({sel_seq_matlab.size}) must equal numPorts ({numPorts})."
        )

    # 转成 0-based 索引以匹配 Python 数组
    sel_seq = sel_seq_matlab - 1

    if np.any(sel_seq < 0) or np.any(sel_seq >= lengths):
        raise ValueError(
            f"Converted indices out of range: min={sel_seq.min()}, max={sel_seq.max()}, "
            f"but x1 has {lengths} rows."
        )

    # ----- Construct phase-encoding matrix from fixed codewords -----
    sign_index = x1[sel_seq, :]           # integer codewords in Z_q
    omega = np.exp(2j * np.pi / q)
    signMatrix_sub = omega ** sign_index  # numPorts × L phase matrix

    # ----- Phase-encoding estimate of perm(A) -----
    perm_est = matrix_perm_encoded(A, signMatrix_sub)

    # ----- Column-wise π phase shifts and encoded permanents perm_c -----
    perm_c = np.zeros(numPorts, dtype=complex)
    for col in range(numPorts):
        As = A.copy()
        th = np.pi
        As[:, col] = np.exp(1j * th) * As[:, col]
        perm_c[col] = matrix_perm_encoded(As, signMatrix_sub)

    # ----- Pre-computed coefficient matrix for this dimension -----
    coeff1 = np.array([
        [ 1,  1,  1,  1, -1, -1, -1, -1],
        [ 1,  1, -1, -1,  1,  1, -1, -1],
        [ 1, -1,  1, -1,  1, -1,  1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1,  1,  1,  1,  1, -1, -1],
        [ 1, -1,  1, -1,  1, -1,  1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1, -1,  1,  1, -1,  1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  1,  1, -1, -1,  1,  1, -1],
        [-1, -1,  1,  1,  1,  1, -1, -1],
        [-1,  1,  1, -1, -1,  1,  1, -1],
        [-1, -1,  1,  1,  1,  1, -1, -1],
        [ 1,  1, -1, -1,  1,  1, -1, -1],
        [ 1, -1,  1, -1,  1, -1,  1, -1],
        [-1,  1,  1, -1, -1,  1,  1, -1],
        [ 1,  1, -1, -1,  1,  1, -1, -1],
        [ 1, -1, -1,  1, -1,  1,  1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1]
    ], dtype=float)

    C = coeff1
    perm_c_mr = perm_c

    # ----- Solve linear system and reconstruct permanent -----
    X_sol = solve_system(C, perm_c_mr.conj().T, perm_est)

    # 对比实部（MATLAB 输出本身也接近实数）
    X_sol_real = X_sol.real

    print(f'\n===== Fixed-sequence reconstruction (N = {numPorts}) =====')
    print(f'Ryser exact permanent                                  = {perm_correct: .16e}')
    print(f'Phase-encoded estimate (no reconstruction)             = {perm_est: .16e}')
    print(f'Reconstructed X_sol (real part)                        = {X_sol_real: .16e}')
    print(f'X_sol / perm_correct                                   = {X_sol_real / perm_correct: .16e}')
    rel_err = abs(X_sol_real - perm_correct) / max(1.0, abs(perm_correct))
    print(f'Relative error |X_sol - perm_correct| / |perm_correct| = {rel_err: .3e}')
    print('=================================================')


def matrix_perm_encoded(matrix: np.ndarray, signMatrix_sub: np.ndarray) -> complex:
    L = signMatrix_sub.shape[1]
    S = matrix @ signMatrix_sub
    lams = np.prod(signMatrix_sub, axis=0)
    JC = np.prod(S, axis=0)
    corr = JC @ np.conjugate(lams)
    return corr / L


def matrix_permanent(A: np.ndarray) -> float:
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    if n == 0:
        return 1.0
    if np.any(np.all(A == 0, axis=0)) or np.any(np.all(A == 0, axis=1)):
        return 0.0
    total = 0.0
    idx_list = list(range(n))
    for k in range(1, n + 1):
        for subset in itertools.combinations(idx_list, k):
            S = list(subset)
            row_sums = np.sum(A[:, S], axis=1)
            prod_val = np.prod(row_sums)
            total += ((-1) ** k) * prod_val
    return float(((-1) ** n) * total)


def qary_rm_linear(q: int, m: int) -> np.ndarray:
    qm = q ** m
    inputs = np.zeros((qm, m), dtype=int)
    for i in range(qm):
        idx = i
        for j in range(m - 1, -1, -1):
            inputs[i, j] = (idx // (q ** j)) % q
    coeffs = np.zeros((qm, m), dtype=int)
    for i in range(qm):
        idx = i
        for j in range(m - 1, -1, -1):
            coeffs[i, j] = (idx // (q ** j)) % q
    codewords = np.mod(coeffs @ inputs.T, q)
    return codewords


def solve_system(C: np.ndarray, b: np.ndarray, S=None) -> complex:
    m, n = C.shape
    b = b.reshape(-1, 1)
    if S is not None:
        A = np.vstack([C, np.ones((1, n))])
        b_aug = np.vstack([b, np.array([[S]])])
        X, *_ = np.linalg.lstsq(A, b_aug, rcond=None)
    else:
        X, *_ = np.linalg.lstsq(C, b, rcond=None)
    x_n = X[-1, 0]
    print(f'Recovered last component X{n} = {x_n:.6f}')
    return x_n


if __name__ == "__main__":
    main()