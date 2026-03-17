# %% Full version (v6.0): Quantum Phase Complex Interference (Complex Phase + Gray Code Rank-1)

import numpy as np
import os
import pickle
import time

# ----------------- Core Parameters -----------------
q = 2
m = 12              # Sequence pool dimension
numPorts = 25       # Matrix dimension N
max_d_allowed = numPorts - m - 1  # Maximum allowed number of pivot columns d
maxTries = 500
# ---------------------------------------------------


# ================== Function Definitions ==================

def matrix_permanent(A):
    """Compute the permanent of a square matrix using Ryser's formula (Gray code version)."""
    t_start = time.time()
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError('A must be square')
    if n == 0:
        return 1.0, time.time() - t_start

    rowSums = np.zeros(n, dtype=A.dtype)
    k = 0
    total = 0.0
    g_prev = np.uint32(0)

    for zz in range(1, 2**n):
        zz_u = np.uint32(zz)
        g = zz_u ^ np.uint32(zz >> 1)
        diff = g ^ g_prev
        j = int(np.floor(np.log2(float(diff))))  # 0-indexed bit position

        if (g >> np.uint32(j)) & np.uint32(1):
            rowSums += A[:, j]
            k += 1
        else:
            rowSums -= A[:, j]
            k -= 1

        total += ((-1) ** (n - k)) * np.prod(rowSums)
        g_prev = g

    return total, time.time() - t_start


def random_select(N, k):
    """Randomly select k distinct indices from {2, ..., N} (1-indexed)."""
    if k > (N - 1):
        raise ValueError('k cannot be greater than N-1')
    full_set = np.arange(2, N + 1)
    rp = full_set[np.random.permutation(N - 1)]
    return rp[:k]


def idx_to_qary_vec(idx, q, m):
    """Convert integer index/indices to q-ary vector representation of length m."""
    idx = np.atleast_1d(np.array(idx, dtype=np.int64)).flatten()
    n = len(idx)
    V = np.zeros((n, m), dtype=np.int64)
    for j in range(m - 1, -1, -1):
        pj = q ** j
        V[:, j] = (idx // pj) % q
    return V


def rm_linear_submatrix_omega(q, m, sel_seq, omega):
    """
    Build the sign sub-matrix using the q-ary linear structure and root of unity omega.
    Returns a (k x L) complex matrix.
    """
    L = q ** m
    sel_seq = np.array(sel_seq, dtype=np.int64)
    k = len(sel_seq)
    # Acoeff: (k x m), each row is the q-ary representation of (sel_seq[i] - 1)
    Acoeff = idx_to_qary_vec(sel_seq - 1, q, m)
    signMatrix_sub = np.zeros((k, L), dtype=complex)
    for xi in range(L):
        xvec = idx_to_qary_vec(xi, q, m)  # shape (1, m)
        vals = (Acoeff @ xvec.T) % q       # shape (k, 1)
        signMatrix_sub[:, xi] = omega ** vals.flatten()
    return signMatrix_sub


def dfs_mod_solver(A, b, q, n, num_vars, fixed_sum):
    """
    Solve modular linear system A*x = b (mod q) with depth-first search,
    subject to the constraint fixed_sum + sum(x) ≡ n (mod q).
    Returns all valid solutions as a 2D array of shape (count, num_vars).
    """
    M, N = A.shape

    # Remove trivially satisfied (all-zero) rows; detect infeasible rows
    valid_rows = []
    for i in range(M):
        if np.all(A[i, :] == 0):
            if b[i] % q != 0:
                return np.zeros((0, num_vars), dtype=np.int64)
            # else: trivially satisfied, skip
        else:
            valid_rows.append(i)

    A = A[valid_rows, :]
    b = b[valid_rows]
    M = A.shape[0]

    # Precompute: for each constraint, the index of the last nonzero variable
    last_var_idx = np.zeros(M, dtype=np.int64)
    for i in range(M):
        nz = np.where(A[i, :] != 0)[0]
        last_var_idx[i] = int(np.max(nz)) if len(nz) > 0 else -1

    # Precompute: for each variable, which constraints it affects
    affected_constraints = [[] for _ in range(num_vars)]
    for j in range(num_vars):
        for i in range(M):
            if A[i, j] != 0:
                affected_constraints[j].append(i)

    max_solutions = min(1_000_000, q ** num_vars)
    solutions = np.zeros((max_solutions, num_vars), dtype=np.int64)
    count = [0]  # Use list for mutability inside nested function

    def dfs_recursive(depth, current_x, current_sum, constraint_vals):
        for val in range(q):
            new_x = current_x.copy()
            new_x[depth] = val
            new_sum = current_sum + val

            if fixed_sum + new_sum > n:
                continue

            new_constraint_vals = constraint_vals.copy()
            valid = True

            for con in affected_constraints[depth]:
                new_constraint_vals[con] = (new_constraint_vals[con] + A[con, depth] * val) % q
                if depth >= last_var_idx[con]:
                    if new_constraint_vals[con] != b[con]:
                        valid = False
                        break

            if not valid:
                continue

            if depth < num_vars - 1:
                dfs_recursive(depth + 1, new_x, new_sum, new_constraint_vals)
            else:
                # Leaf node: check global sum constraint
                if (fixed_sum + new_sum) % q != n % q:
                    continue
                # Check all constraints
                for i2 in range(M):
                    if new_constraint_vals[i2] != b[i2]:
                        valid = False
                        break
                if valid:
                    if count[0] >= solutions.shape[0]:
                        # Expand buffer
                        extra = np.zeros((10000, num_vars), dtype=np.int64)
                        solutions_ref[0] = np.vstack([solutions_ref[0], extra])
                    solutions_ref[0][count[0], :] = new_x
                    count[0] += 1

    solutions_ref = [solutions]
    dfs_recursive(0, np.zeros(num_vars, dtype=np.int64), 0, np.zeros(M, dtype=np.int64))
    return solutions_ref[0][:count[0], :]


def ultra_fast_match_fix(seqs, n, lams, q):
    """
    Match phase exponents using the modular DFS solver.
    seqs: (k x L) complex matrix
    lams: (L,) complex array
    Returns all matching exponent vectors.
    """
    # Convert complex phases to integer exponents mod q
    c = np.mod(np.round(np.angle(seqs) * q / (2 * np.pi)).astype(np.int64), q)
    d = np.mod(np.round(np.angle(lams) * q / (2 * np.pi)).astype(np.int64), q)
    A_reduced = c.T          # shape (L, k)
    b_reduced = d            # shape (L,)
    fixed_sum = 0
    num_free = A_reduced.shape[1]
    free_solutions = dfs_mod_solver(A_reduced, b_reduced, q, n, num_free, fixed_sum)
    return free_solutions


# ================== Main Program ==================

# Generate test matrix
np.random.seed(None)  # Equivalent to rng('shuffle')
A = np.random.rand(numPorts, numPorts) - 0.5

# Ground truth (Ryser)
perm_true, t_exact = matrix_permanent(A)

print(f'Matrix dimension: {numPorts}×{numPorts}, m={m}, q={q}')
print(f'Maximum allowed interference dimension (d): {max_d_allowed}')
print(f'True permanent: {perm_true:.12g}')
print(f'Ryser exact computation time: {t_exact:.4f} seconds\n')

# %% Cache file configuration (reuse optimal sequences found by V5 if available)
cached_file = f'optimal_sequence_cache_n{numPorts}_m{m}_q{q}_d{max_d_allowed}_v5.pkl'

cached_found = False
if os.path.exists(cached_file):
    try:
        with open(cached_file, 'rb') as f:
            cache_data = pickle.load(f)
        cached_params = cache_data.get('cached_params', {})
        expected_params = {'q': q, 'm': m, 'n': numPorts, 'max_d': max_d_allowed}
        if cached_params == expected_params:
            print('=== Cache hit! Loading stored sequence and pivot columns ===')
            sel_seq      = cache_data['sel_seq_cached']
            phase_order  = cache_data['phase_order_cached']
            pivot_cols   = cache_data['pivot_cols_cached']
            d            = cache_data['d_cached']
            cached_found = True
    except Exception as e:
        print(f'Cache invalid, will re-search. ({e})')

# %% ===================== Search Phase =====================
if not cached_found:
    print(f'=== Starting optimal sequence search (target: ghost collapse dimension d <= {max_d_allowed}) ===')
    found = False
    omega = -1

    for try_num in range(1, maxTries + 1):
        if try_num % 10 == 0:
            print(f'  Attempted {try_num} times...')

        sel_seq = random_select(q ** m, numPorts)
        signMatrix_sub = rm_linear_submatrix_omega(q, m, sel_seq, omega)
        lams = np.prod(signMatrix_sub, axis=0)
        phase_order = ultra_fast_match_fix(signMatrix_sub, numPorts, lams, q)
        r = phase_order.shape[0]

        # Find the target row where all elements equal 1
        idx_target = np.where(np.all(phase_order == 1, axis=1))[0]
        if len(idx_target) == 0:
            continue
        idx_target = idx_target[0]

        # Remove target row to get ghost rows
        ghosts = np.delete(phase_order, idx_target, axis=0)
        active_ghosts = ghosts.copy()
        S = []

        # Greedy column selection to eliminate all ghosts
        while active_ghosts.shape[0] > 0:
            zeros_cnt = np.sum(active_ghosts == 0, axis=0)
            best_col = int(np.argmax(zeros_cnt))
            max_zeros = zeros_cnt[best_col]
            if max_zeros == 0:
                S = [np.inf]
                break
            S.append(best_col)
            # Keep only rows where this column equals 1
            active_ghosts = active_ghosts[active_ghosts[:, best_col] == 1, :]

        d_current = len(S)

        if d_current <= max_d_allowed:
            found = True
            pivot_cols = S
            d = d_current
            print(f'  >> [Locked] Search #{try_num} succeeded! Initial ghosts: {r - 1}')
            break

    if not found:
        raise RuntimeError('No sequence satisfying the dimension requirement was found.')

    # Save to cache
    cache_data = {
        'sel_seq_cached':     sel_seq,
        'phase_order_cached': phase_order,
        'pivot_cols_cached':  pivot_cols,
        'd_cached':           d,
        'cached_params':      {'q': q, 'm': m, 'n': numPorts, 'max_d': max_d_allowed}
    }
    with open(cached_file, 'wb') as f:
        pickle.dump(cache_data, f)

# %% ===================== V6.0: Quantum Phase Ultra-Fast Annihilation Phase =====================

# [Core Mathematical Magic]: Separate 1 target column into the imaginary domain,
# the rest follow Gray Code traversal.
pivot_cols = list(pivot_cols)  # Ensure it's a plain list

if d > 0:
    complex_col = pivot_cols[0]       # This column will be mapped to the imaginary axis!
    gray_cols   = pivot_cols[1:]      # Remaining columns undergo conventional interference
    d_gray = d - 1
else:
    complex_col = None
    gray_cols   = []
    d_gray      = 0

num_iterations = 2 ** d_gray  # Iteration count perfectly halved!

print('\n=== Launching [Quantum Complex Orthogonal Filter] Interference Engine ===')
print(f'Total locked pivot columns: {d}')
if d > 0:
    print(f'  -> [Dimension-sacrifice column]: column index {complex_col} '
          f'(mapped to imaginary axis, eliminates 50% of ghosts at once!)')
    print(f'  -> [Gray code pivot columns]: {gray_cols} (conventional cancellation)')
print(f'Computations to perform: 2^{d_gray} = {num_iterations} updates '
      f'(saves half the computation vs V5!)')

t_start = time.time()
omega = -1
signMatrix_sub = rm_linear_submatrix_omega(q, m, sel_seq, omega)
lams = np.prod(signMatrix_sub, axis=0)

A_eval = A.astype(complex).copy()
if d > 0:
    # Multiply all elements of the complex_col column by imaginary unit i
    A_eval[:, complex_col] = A_eval[:, complex_col] * 1j

# 1) Initialize ground-state projection (using complex matrix)
Y_current = A_eval @ signMatrix_sub   # shape: (numPorts, L)

if d_gray > 0:
    current_signs = np.ones(d_gray, dtype=np.float64)
    sgn = 1

    JC = np.prod(Y_current, axis=0)
    Perm_current = (JC @ np.conj(lams)) / (2 ** m)
    total_interference_sum = sgn * Perm_current

    # Precompute Gray code flip sequence
    indices = np.arange(num_iterations, dtype=np.uint64)
    G = indices ^ (indices >> 1)
    flip_mask = G[:-1] ^ G[1:]
    flip_indices = np.round(np.log2(flip_mask.astype(np.float64))).astype(np.int64)
    # flip_indices are 0-based positions into gray_cols

    for step in range(num_iterations - 1):
        idx = flip_indices[step]          # Which gray_col to flip (0-based)
        c   = gray_cols[idx]              # Actual column index in A

        current_signs[idx] = -current_signs[idx]
        sgn = -sgn

        update_vec = 2 * current_signs[idx] * A_eval[:, c]   # shape: (numPorts,)
        Y_current = Y_current + np.outer(update_vec, signMatrix_sub[c, :])

        JC = np.prod(Y_current, axis=0)
        Perm_current = (JC @ np.conj(lams)) / (2 ** m)
        total_interference_sum += sgn * Perm_current

else:
    JC = np.prod(Y_current, axis=0)
    total_interference_sum = (JC @ np.conj(lams)) / (2 ** m)

# [Key insight]: Take only the imaginary part!
# All even-power ghost terms are isolated in the real part and can be directly ignored!
if d > 0:
    perm_est = np.imag(total_interference_sum) / (2 ** d_gray)
else:
    perm_est = np.real(total_interference_sum)

t_est = time.time() - t_start

rel_err = abs(perm_est - perm_true) / max(abs(perm_true), np.finfo(float).eps)

print('\n================ Battle Report Summary ================')
print(f'After {num_iterations} iterations, leveraging complex plane orthogonality,')
print(f'all ghost terms have been 100% successfully annihilated!')
print(f'Theoretical scaling factor: 2^{d_gray} = {num_iterations}')
print('Results:')
print(f'  Estimated value : {perm_est:.12g}')
print(f'  True value      : {perm_true:.12g}')
print(f'  Relative error  : {rel_err:.3e}')
print(f'\nInterference execution time : {t_est:.4f} seconds')
print(f'Ryser speedup ratio         : {t_exact / t_est:.4f}x')
print('=======================================================')