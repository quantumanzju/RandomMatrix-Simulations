
import numpy as np
import itertools
import time

"""
%% Exact_RM_AnySequence.py
% Demonstration script: Reed–Muller based phase encoding and
% single-parameter reconstruction of a matrix permanent.
%
% This script implements the numerical experiment described in
% Sections 3–4. It:
%   1. Generates a q-ary order-1 Reed–Muller code RM_q(1,m);
%   2. Uses RM codewords as phase-encoding patterns;
%   3. Recovers the modular phase exponents by solving a constrained
%      linear system over Z_q;
%   4. Applies column-wise phase perturbations to the matrix;
%   5. Reconstructs the permanent from a 1-parameter linear model,
%      and compares it with the exact Ryser permanent.
%
% The structure and variable names are kept close to the original
% implementation used in our experiments, with only cosmetic changes
% for readability.
%
% NOTE:
%   Due to random initialization and the non-convex nature of the
%   reconstruction, the script may fail to find a good solution in a
%   single run. It is expected that you may need to execute the script
%   multiple times and keep the run with the smallest reconstruction
%   error.
"""

# ======================= Main script =======================

def main():
    # ----- Basic parameters -----
    q = 2          # Alphabet size (q-ary RM code, here binary)
    m = 15         # RM code parameter, code length = q^m
    numPorts = 19  # Matrix dimension / number of optical modes (N in the text)

    # ----- Generate q-ary order-1 Reed–Muller linear code RM_q(1,m) -----
    # x1: (q^m) × (q^m) matrix, each row is a codeword in Z_q^{q^m}
    x1 = qary_rm_linear(q, m)
    lengths = x1.shape[0]

    # ----- Test matrix and ground-truth permanent -----
    np.random.seed()  # similar to rng('shuffle')
    A = np.random.rand(numPorts, numPorts)      # real i.i.d. test matrix
    perm_correct = matrix_permanent(A)          # exact permanent via Ryser

    # ----- Select RM codewords as phase-encoding patterns -----
    # Randomly select numPorts distinct RM codewords as phase patterns.
    np.random.seed()
    sel_seq = random_select(lengths, numPorts)      # MATLAB-style indices in {2,...,lengths}
    sel_seq = sel_seq[np.random.permutation(len(sel_seq))]
    sign_index = x1[sel_seq, :]                    # integer codewords in Z_q

    # Map Z_q symbols to complex q-th roots of unity
    omega = np.exp(2j * np.pi / q)
    signMatrix_sub = omega ** sign_index           # numPorts × L phase matrix

    # Column-wise product λ_j of all phases in pattern j
    lams = np.prod(signMatrix_sub, axis=0)

    # ----- Recover modular phase exponents (Section 4) -----
    # ultra_fast_match_mod implements a fast constrained solver over Z_q
    phase_order = ultra_fast_match_mod(signMatrix_sub, numPorts, lams, q)
    sum_order = np.sum(phase_order, axis=1)  # not explicitly used here
    print('Recovered modular phase exponents (phase_order):')
    print(phase_order)

    # ----- Phase-encoding estimator for the permanent (Section 3) -----
    # Implements:
    #   Perm(A) = (1/L) ∑_j [ ∏_i (∑_k A_{ik} P_{kj}) ] · conj(∏_k P_{kj})
    perm_est = matrix_perm_encoded(A, signMatrix_sub)

    # ----- Column-wise phase perturbations and encoded permanents -----
    # Apply a fixed phase shift e^{iπ} to one column at a time and
    # recompute the encoded permanent.
    perm_c = np.zeros(numPorts, dtype=complex)
    for col in range(numPorts):
        As = A.copy()
        th = np.pi  # fixed phase shift
        As[:, col] = np.exp(1j * th) * As[:, col]
        perm_c[col] = matrix_perm_encoded(As, signMatrix_sub)

    # ----- 1-parameter linear reconstruction model -----
    # Build a simple 1-parameter linear model based on the recovered
    # modular exponents:
    #
    #   coeff1(j) = 1 - (k+1) * phase_order(j)
    #
    # and solve for a single scalar X_sol from:
    #
    #   coeff1 * X ≈ perm_c^T
    #
    # using a linear system with a sum constraint S = perm_est.
    k = 1
    # phase_order: shape (numPorts, r) in MATLAB; we use the same convention
    coeff1 = 1 - (k + 1) * phase_order.T        # shape: (numPorts, r)
    coeff_sum_c = np.sum(coeff1, axis=0)        # not explicitly used, printed for diagnostics

    print('Coefficient matrix coeff1:')
    print(coeff1)
    print('Column sums coeff_sum_c:')
    print(coeff_sum_c)

    perm_c_mr1 = perm_c          # shorthand (row vector)
    coeff = coeff1               # coefficient matrix
    perm_c_mr = perm_c_mr1       # right-hand side

    # Solve for the last variable via the original linear solver.
    # The solver supports an optional "sum" constraint S; here we pass
    # perm_est as S, consistent with the original implementation.
    X_sol = solve_system(coeff, perm_c_mr.conj().T, perm_est)

    # ----- Numerical comparison -----
    print(f'\n===== Fixed-sequence reconstruction (N = {numPorts}) =====')
    print(f'Ryser exact permanent                                  = {perm_correct: .16e}')
    print(f'Phase-encoding estimator (no reconstruction)           = {perm_est: .16e}')
    print(f'Reconstructed X_sol                                    = {X_sol: .16e}')
    print(f'X_sol / perm_correct                                   = {X_sol / perm_correct: .16e}')
    rel_err = abs(X_sol - perm_correct) / max(1.0, abs(perm_correct))
    print(f'Relative error |X_sol - perm_correct| / |perm_correct| = {rel_err: .3e}')
    print('=================================================')


# ======================= Function definitions =======================

def matrix_perm_encoded(matrix: np.ndarray, signMatrix_sub: np.ndarray) -> complex:
    """
    Phase-encoding based estimator of the matrix permanent.

    Implements
        Perm(A) = (1/L) sum_{j=1}^L [ prod_i (sum_k A_{ik} P_{kj}) ] * conj(prod_k P_{kj}),

    where each column j of signMatrix_sub stores a phase pattern P_{·j}.
    """
    L = signMatrix_sub.shape[1]
    S = matrix @ signMatrix_sub          # row-wise sums with phases
    lams = np.prod(signMatrix_sub, axis=0)   # column-wise products
    JC = np.prod(S, axis=0)              # product over rows
    corr = JC @ np.conjugate(lams)
    Perm = corr / L
    return Perm


def matrix_permanent(A: np.ndarray) -> float:
    """
    Exact permanent via Ryser's formula.

    perm(A) = (-1)^n sum_{∅≠S⊆{1..n}} (-1)^{|S|} prod_i (sum_{j∈S} A_{ij})

    Parameters
    ----------
    A : np.ndarray
        Square matrix (n x n).

    Returns
    -------
    float
        Permanent of A.
    """
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

    perm = ((-1) ** n) * total
    return float(perm)


def qary_rm_linear(q: int, m: int) -> np.ndarray:
    """
    Generate the q-ary order-1 Reed–Muller linear code RM_q(1,m).

    Returns a qm × qm matrix (qm = q^m), whose rows enumerate all affine
    linear functions over Z_q^m, evaluated on all qm input points.
    """
    qm = q ** m

    # Enumerate all inputs x ∈ Z_q^m
    inputs = np.zeros((qm, m), dtype=int)
    for i in range(qm):
        idx = i
        for j in range(m - 1, -1, -1):
            inputs[i, j] = (idx // (q ** j)) % q

    # Enumerate all coefficient vectors a ∈ Z_q^m
    coeffs = np.zeros((qm, m), dtype=int)
    for i in range(qm):
        idx = i
        for j in range(m - 1, -1, -1):
            coeffs[i, j] = (idx // (q ** j)) % q

    # Evaluate all linear functions a·x over all inputs
    codewords = np.mod(coeffs @ inputs.T, q)
    return codewords


def random_select(N: int, k: int) -> np.ndarray:
    """
    Randomly select k distinct integers from {2, ..., N} (MATLAB-style 1-based).

    This follows the original convention of skipping index 1.
    Returned indices are 0-based for Python, but we mimic the MATLAB
    behavior of drawing from {2,...,N} by shifting.

    In MATLAB:
        full_set = 2:N;
        perm = full_set(randperm(N-1));
        selected = perm(1:k);

    Here we return indices in [1, N-1] (0-based) corresponding to [2, ..., N] (1-based).
    """
    if k > (N - 1):
        raise ValueError("k cannot exceed N-1.")
    full_set = np.arange(1, N)          # 0-based, corresponds to MATLAB 2..N
    perm = np.random.permutation(full_set)
    selected = perm[:k]
    return selected


def solve_system(C: np.ndarray, b: np.ndarray, S=None) -> complex:
    """
    Solve a small linear system and return the last variable.

    Two modes:
      - with sum constraint S:  [C; 1^T] X = [b; S]
      - without S (S is None):  C X ≈ b

    Here C is an m×n matrix of coefficients, b is an m×1 vector, and the
    solution X is obtained via a least-squares solver. The function
    returns only the last component X_n, consistent with the original code.
    """
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


# ===== Ultra-fast modular exponent matching and helpers =====

def ultra_fast_match_mod(seqs: np.ndarray, n: int, lams: np.ndarray, q: int,
                         *flags) -> np.ndarray:
    """
    Fast solver for modular linear constraints induced by phase encodings.

    Given
        seqs : N × M complex phase matrix (each column a phase pattern),
        n    : target sum parameter (problem-dependent),
        lams : 1 × M column products λ_j,
        q    : modulus (e.g., q = 2),

    we convert them to modular constraints on integer exponents and
    solve via DFS with aggressive pruning and constraint propagation.

    Parameters
    ----------
    flags : str, optional
        'remainder_only' to return only remainder vectors;
        'no_prune'       to disable aggressive pruning.
    """
    N, M = seqs.shape

    output_remainder_only = False
    enable_aggressive_prune = True
    if len(flags) > 0:
        if any(str(f).lower() == 'remainder_only' for f in flags):
            output_remainder_only = True
        if any(str(f).lower() == 'no_prune' for f in flags):
            enable_aggressive_prune = False

    # Convert complex phases to integer exponents modulo q
    c = np.mod(np.round(np.angle(seqs) * q / (2 * np.pi)), q).astype(int)  # N × M
    d = np.mod(np.round(np.angle(lams) * q / (2 * np.pi)), q).astype(int)  # 1 × M

    # ----- Constraint propagation -----
    if enable_aggressive_prune:
        A = c.T  # M × N
        b = d.reshape(-1, 1)  # M × 1

        free_vars, fixed_values = robust_constraint_propagation(A, b, q)
        fixed_sum = np.sum(fixed_values[~free_vars])

        if np.any(~free_vars):
            A_reduced = A[:, free_vars]
            b_reduced = np.zeros_like(b)
            fixed_indices = np.where(~free_vars)[0]

            for j in range(A.shape[0]):
                if fixed_indices.size > 0:
                    fixed_contribution = np.dot(A[j, fixed_indices],
                                                fixed_values[fixed_indices])
                else:
                    fixed_contribution = 0
                b_reduced[j, 0] = int(np.mod(b[j, 0] - fixed_contribution, q))
        else:
            A_reduced = A
            b_reduced = b
            fixed_sum = 0
    else:
        free_vars = np.ones(N, dtype=bool)
        fixed_values = np.zeros(N, dtype=int)
        A_reduced = c.T
        b_reduced = d.reshape(-1, 1)
        fixed_sum = 0

    # ----- DFS search over reduced system -----
    num_free = np.sum(free_vars)
    if num_free > 0:
        free_solutions = dfs_mod_solver(A_reduced, b_reduced, q, n, num_free, fixed_sum)
        remainder_list = reconstruct_solutions(free_solutions, free_vars, fixed_values)
    else:
        # Nothing to solve; everything fixed
        remainder_list = fixed_values.reshape(1, -1)

    if output_remainder_only:
        matching_exponents = remainder_list
    else:
        matching_exponents = generate_minimal_solutions(remainder_list, n, q)

    return matching_exponents


def robust_constraint_propagation(A: np.ndarray, b: np.ndarray, q: int):
    """
    Iterative elimination of variables that are uniquely determined
    by single-variable constraints in the modular system A x ≡ b (mod q).

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix (M × N).
    b : np.ndarray
        Right-hand side vector (M × 1).
    q : int
        Modulus.

    Returns
    -------
    free_vars : np.ndarray (bool)
        Boolean mask of free variables.
    fixed_values : np.ndarray (int)
        Values of variables already fixed by propagation.
    """
    M, N = A.shape
    free_vars = np.ones(N, dtype=bool)
    fixed_values = np.zeros(N, dtype=int)

    changed = True
    iteration_count = 0
    max_iterations = N * 10

    while changed and iteration_count < max_iterations:
        iteration_count += 1
        changed = False

        for j in range(M):
            non_zero_idx = np.where(A[j, :] != 0)[0]
            free_non_zero = non_zero_idx[free_vars[non_zero_idx]]

            if len(free_non_zero) == 1:
                idx = free_non_zero[0]

                fixed_indices = np.where(~free_vars)[0]
                if fixed_indices.size > 0:
                    fixed_contrib = np.dot(A[j, fixed_indices], fixed_values[fixed_indices])
                else:
                    fixed_contrib = 0

                coeff = int(A[j, idx])
                g = np.gcd(coeff, q)
                if g == 0:
                    raise ValueError(f"Coefficient {coeff} and modulus {q} share gcd = 0.")

                target_val = int(np.mod(b[j, 0] - fixed_contrib, q))
                if target_val % g != 0:
                    # Inconsistent constraint; skip it.
                    continue

                effective_q = q // g
                reduced_target = target_val // g
                reduced_coeff = coeff // g
                _, inv_red, _ = extended_gcd(reduced_coeff, effective_q)
                fixed_val = (reduced_target * inv_red) % effective_q

                fixed_values[idx] = fixed_val
                free_vars[idx] = False
                changed = True
                break

    return free_vars, fixed_values


def dfs_mod_solver(A: np.ndarray, b: np.ndarray, q: int, n: int,
                   num_vars: int, fixed_sum: int) -> np.ndarray:
    """
    Depth-first search solver for the reduced modular system,
    with pruning by partial sums and constraint consistency.

    Parameters
    ----------
    A : np.ndarray
        Reduced constraint matrix (M × num_vars).
    b : np.ndarray
        Reduced right-hand side (M × 1).
    q : int
        Modulus.
    n : int
        Target sum parameter.
    num_vars : int
        Number of free variables.
    fixed_sum : int
        Sum contributed by already fixed variables.

    Returns
    -------
    solutions : np.ndarray
        All satisfying assignments over free variables.
    """
    M, N = A.shape
    if N != num_vars:
        raise ValueError(f"Number of variables ({N}) inconsistent with expected ({num_vars}).")

    # Remove all-zero rows and check consistency
    valid_rows = np.ones(M, dtype=bool)
    for i in range(M):
        if np.all(A[i, :] == 0):
            if int(b[i, 0]) % q != 0:
                return np.zeros((0, num_vars), dtype=int)
            else:
                valid_rows[i] = False

    A = A[valid_rows, :]
    b = b[valid_rows, :]
    M = A.shape[0]

    if M > 0:
        last_var_idx = np.zeros(M, dtype=int)
        for i in range(M):
            nz = np.where(A[i, :] != 0)[0]
            last_var_idx[i] = np.max(nz)

        affected_constraints = [np.where(A[:, j] != 0)[0] for j in range(num_vars)]
    else:
        last_var_idx = np.array([], dtype=int)
        affected_constraints = [np.array([], dtype=int) for _ in range(num_vars)]

    max_solutions = min(1000000, q ** num_vars)
    solutions = np.zeros((max_solutions, num_vars), dtype=int)
    count = 0

    if fixed_sum > n:
        return np.zeros((0, num_vars), dtype=int)

    def dfs_recursive(depth, current_x, current_sum, constraint_vals):
        nonlocal count, solutions

        for val in range(q):
            new_x = current_x.copy()
            new_x[depth] = val
            new_sum = current_sum + val

            if fixed_sum + new_sum > n:
                continue

            new_constraint_vals = constraint_vals.copy()
            valid = True

            if M > 0:
                aff_cons = affected_constraints[depth]
                for con_idx in aff_cons:
                    new_constraint_vals[con_idx] = \
                        (new_constraint_vals[con_idx] + A[con_idx, depth] * val) % q
                    if depth >= last_var_idx[con_idx]:
                        if new_constraint_vals[con_idx] != b[con_idx, 0]:
                            valid = False
                            break

            if not valid:
                continue

            if depth < num_vars - 1:
                dfs_recursive(depth + 1, new_x, new_sum, new_constraint_vals)
            else:
                if (fixed_sum + new_sum) % q != n % q:
                    continue

                all_ok = True
                if M > 0:
                    for i in range(M):
                        if new_constraint_vals[i] != b[i, 0]:
                            all_ok = False
                            break

                if all_ok:
                    count += 1
                    if count > solutions.shape[0]:
                        solutions = np.vstack([solutions, np.zeros((10000, num_vars), dtype=int)])
                    solutions[count - 1, :] = new_x

    init_constraint_vals = np.zeros(M, dtype=int)
    if num_vars > 0:
        dfs_recursive(0, np.zeros(num_vars, dtype=int), 0, init_constraint_vals)

    return solutions[:count, :]


def reconstruct_solutions(free_solutions: np.ndarray,
                          free_vars: np.ndarray,
                          fixed_values: np.ndarray) -> np.ndarray:
    """
    Expand solutions on free variables to full vectors.

    Parameters
    ----------
    free_solutions : np.ndarray
        Solutions over free variables (num_sol × num_free).
    free_vars : np.ndarray (bool)
        Mask of free variables.
    fixed_values : np.ndarray
        Fixed values for non-free variables.

    Returns
    -------
    full_solutions : np.ndarray
        Solutions in the full variable space (num_sol × N).
    """
    num_sol = free_solutions.shape[0]
    N = free_vars.size

    if num_sol > 0:
        full_solutions = np.zeros((num_sol, N), dtype=int)
        if np.any(~free_vars):
            fixed_columns = np.tile(fixed_values, (num_sol, 1))
            full_solutions[:, free_vars] = free_solutions
            full_solutions[:, ~free_vars] = fixed_columns[:, ~free_vars]
        else:
            full_solutions = free_solutions
    else:
        full_solutions = fixed_values.reshape(1, -1)

    return full_solutions


def generate_minimal_solutions(remainder_list: np.ndarray, n: int, q: int) -> np.ndarray:
    """
    In this implementation we simply take the modular remainders as minimal solutions.

    Parameters
    ----------
    remainder_list : np.ndarray
        List of remainder solutions modulo q.
    n : int
        Target sum parameter (unused here).
    q : int
        Modulus (unused here).

    Returns
    -------
    minimal_solutions : np.ndarray
        The same as remainder_list.
    """
    return remainder_list


def extended_gcd(a: int, b: int):
    """Extended Euclidean algorithm: returns (g, x, y) such that g = ax + by."""
    if b == 0:
        return a, 1, 0
    else:
        g, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return g, x, y


if __name__ == "__main__":
    main()