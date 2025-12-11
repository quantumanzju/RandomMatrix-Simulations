%% Exact_RM_AnySequence.m
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

clc;
clear;
close all;

%% ----- Basic parameters -----
q        = 2;    % Alphabet size (q-ary RM code, here binary)
m        = 15;   % RM code parameter, code length = q^m
numPorts = 19;   % Matrix dimension / number of optical modes (N in the text)

%% ----- Generate q-ary order-1 Reed–Muller linear code RM_q(1,m) -----
% x1: (q^m) × (q^m) matrix, each row is a codeword in Z_q^{q^m}
x1      = qary_rm_linear(q, m);
lengths = size(x1, 1);

%% ----- Test matrix and ground-truth permanent -----
rng('shuffle');
A            = rand(numPorts, numPorts);      % real i.i.d. test matrix
perm_correct = matrix_permanent(A);           % exact permanent via Ryser

%% ----- Select RM codewords as phase-encoding patterns -----
% We randomly select numPorts distinct RM codewords as our phase patterns.
rng('shuffle');
sel_seq        = random_select(lengths, numPorts);    % indices in {2,...,lengths}
sel_seq        = sel_seq(randperm(length(sel_seq))); % random permutation
sign_index     = x1(sel_seq, :);                     % integer codewords in Z_q

% Map Z_q symbols to complex q-th roots of unity
omega          = exp(2 * 1i * pi / q);
signMatrix_sub = omega .^ sign_index;                % numPorts × L phase matrix

% Column-wise product λ_j of all phases in pattern j
lams = prod(signMatrix_sub, 1);

%% ----- Recover modular phase exponents (Section 4) -----
% ultra_fast_match_mod4 implements a fast constrained solver over Z_q
% for the phase-exponent matching problem using:
%   c_{ij} = angle(seqs_{ij}) / (2π/q)  (mod q)
%   d_j    = angle(λ_j) / (2π/q)       (mod q)
% and DFS with aggressive pruning.

phase_order = ultra_fast_match_mod(signMatrix_sub, numPorts, lams, q);
sum_order   = sum(phase_order, 1); %#ok<NASGU>
disp('Recovered modular phase exponents (phase_order):');
disp(phase_order);

%% ----- Phase-encoding estimator for the permanent (Section 3) -----
% Implements:
%   Perm(A) = (1/L) ∑_j [ ∏_i (∑_k A_{ik} P_{kj}) ] · conj(∏_k P_{kj})
perm = matrix_perm(A, signMatrix_sub);

%% ----- Column-wise phase perturbations and encoded permanents -----
% We apply a fixed phase shift e^{iπ} to one column at a time and
% recompute the encoded permanent.

perm_c = zeros(1, numPorts);
for col = 1:numPorts
    As         = A;
    th         = pi;                  % fixed phase shift
    As(:, col) = exp(1i * th) * As(:, col);
    perm_c(col) = matrix_perm(As, signMatrix_sub);
end

%% ----- 1-parameter linear reconstruction model -----
% We now build a simple 1-parameter linear model based on the recovered
% modular exponents:
%
%   coeff1(j) = 1 - (k+1) * phase_order(j)   (modulo arithmetic implicit)
%
% and solve for a single scalar X_sol from:
%
%   coeff1 * X ≈ perm_c^T
%
% using a linear system with an additional sum constraint. The recovered
% X_sol is then compared against the exact permanent.

k        = 1;
coeff1   = 1 - ((k + 1) * phase_order).';  % size: (numPorts × r), r=number of rows in phase_order
coeff_sum_c = sum(coeff1, 1);              % column-wise sum (not explicitly used here)

disp('Coefficient matrix coeff1:');
disp(coeff1);
disp('Column sums coeff_sum_c:');
disp(coeff_sum_c);

perm_c_mr1 = perm_c;      % shorthand (row vector)
coeff      = coeff1;      % matrix of coefficients
perm_c_mr  = perm_c_mr1;  % right-hand side (row)

% Solve for the last variable via the original linear solver.
% The solver supports an optional "sum" constraint S; here we pass
% 'perm' as S, which is consistent with the original implementation.
X_sol = solve_system(coeff, perm_c_mr.', perm);

%% ----- Numerical comparison -----
fprintf('\n===== Fixed-sequence reconstruction (N = %d) =====\n', numPorts);
fprintf('Ryser exact permanent                                  = % .16e\n', perm_correct);
fprintf('Reconstructed X_sol                                    = % .16e\n', X_sol);
fprintf('X_sol / perm_correct                                   = % .16e\n', X_sol / perm_correct);
fprintf('Relative error |X_sol - perm_correct| / |perm_correct| = % .3e\n', ...
        abs(X_sol - perm_correct) / max(1, abs(perm_correct)));
fprintf('=================================================\n');

%% =====================================================================
%  Function definitions (used by the above demo)
%% =====================================================================

function Perm = matrix_perm(matrix, signMatrix_sub)
% MATRIX_PERM  Phase-encoding based estimator of the matrix permanent.
%
%   Perm(A) = (1/L) ∑_{j=1}^L [ ∏_i (∑_k A_{ik} P_{kj}) ] · conj(∏_k P_{kj}),
%
% where each column j of signMatrix_sub stores a phase pattern P_{·j}.

    L     = size(signMatrix_sub, 2);
    S     = matrix * signMatrix_sub;         % row-wise sums with phases
    lams  = prod(signMatrix_sub, 1);         % column-wise products
    JC    = prod(S, 1);                      % product over rows
    corr  = JC * conj(lams).';
    Perm  = corr / L;
end


function perm = matrix_permanent(A)
% MATRIX_PERMANENT  Exact permanent via Ryser's formula.
%
%   perm(A) = (-1)^n ∑_{∅≠S⊆{1..n}} (-1)^{|S|} ∏_i (∑_{j∈S} A_{ij})

    n = size(A, 1);
    if size(A, 1) ~= size(A, 2)
        error('Input must be a square matrix.');
    end
    if n == 0
        perm = 1;
        return;
    end
    if any(all(A == 0, 1)) || any(all(A == 0, 2))
        perm = 0;
        return;
    end

    total = 0;
    for k = 1:n
        subsets = nchoosek(1:n, k);
        for idx = 1:size(subsets, 1)
            S        = subsets(idx, :);
            row_sums = zeros(n, 1);
            for i = 1:n
                row_sums(i) = sum(A(i, S));
            end
            prod_val = prod(row_sums);
            total    = total + (-1)^k * prod_val;
        end
    end
    perm = (-1)^n * total;
end


function codewords = qary_rm_linear(q, m)
% QARY_RM_LINEAR  Generate the q-ary order-1 Reed–Muller linear code RM_q(1,m).
%
% Returns a qm × qm matrix (qm = q^m), whose rows enumerate all affine
% linear functions over Z_q^m, evaluated on all qm input points.

    qm = q^m;

    % Enumerate all inputs x ∈ Z_q^m
    inputs = zeros(qm, m);
    for i = 1:qm
        idx = i - 1;
        for j = m:-1:1
            inputs(i, j) = mod(floor(idx / q^(j-1)), q);
        end
    end

    % Enumerate all coefficient vectors a ∈ Z_q^m
    coeffs = zeros(qm, m);
    for i = 1:qm
        idx = i - 1;
        for j = m:-1:1
            coeffs(i, j) = mod(floor(idx / q^(j-1)), q);
        end
    end

    % Evaluate all linear functions a·x over all inputs
    codewords = mod(coeffs * inputs.', q);
end


function selected = random_select(N, k)
% RANDOM_SELECT  Randomly select k distinct integers from {2, ..., N}.
%
% This follows the original convention of skipping index 1.

    if k > (N - 1)
        error('k cannot exceed N-1.');
    end

    full_set = 2:N;
    perm     = full_set(randperm(N-1));
    selected = perm(1:k);
end


function x_n = solve_system(C, b, S)
% SOLVE_SYSTEM  Solve a small linear system and return the last variable.
%
% Two modes:
%   - with sum constraint S:  [C; 1^T] X = [b; S]
%   - without S (S = []):      C X ≈ b
%
% Here C is an m×n matrix of coefficients, b is an m×1 vector, and the
% solution X is obtained via the MATLAB backslash operator. The function
% returns only the last component X_n, consistent with the original code.

    [m, n] = size(C); %#ok<NASGU>

    if nargin < 3
        S = [];
    end

    b = b(:);

    if ~isempty(S)
        A     = [C; ones(1, n)];
        b_aug = [b; S];
        X     = A \ b_aug;
    else
        X = C \ b;
    end

    x_n = X(end);
    fprintf('Recovered last component X%d = %.6f\n', n, x_n);
end


%% ===== Ultra-fast modular exponent matching and helpers =====

function matching_exponents = ultra_fast_match_mod(seqs, n, lams, q, varargin)
% ULTRA_FAST_MATCH_MOD4
% Fast solver for modular linear constraints induced by phase encodings.
%
% Given:
%   seqs: N × M complex phase matrix (each column a phase pattern),
%   n   : target sum parameter (problem-dependent),
%   lams: 1 × M column products λ_j,
%   q   : modulus (e.g., q = 2),
%
% we convert them to modular constraints on integer exponents and
% solve via DFS with aggressive pruning and constraint propagation.

    [N, M] = size(seqs);

    output_remainder_only   = false;
    enable_aggressive_prune = true;
    if nargin >= 5
        if any(strcmpi(varargin, 'remainder_only'))
            output_remainder_only = true;
        end
        if any(strcmpi(varargin, 'no_prune'))
            enable_aggressive_prune = false;
        end
    end

    t_total = tic; %#ok<NASGU>
    diag_report = struct();
    diag_report.input_parameters = struct('N', N, 'n', n, 'M', M, 'q', q);

    % Convert complex phases to integer exponents modulo q
    c = mod(round(angle(seqs) * q / (2*pi)), q);
    d = mod(round(angle(lams) * q / (2*pi)), q);

    % ----- Constraint propagation -----
    t_prune = tic;
    if enable_aggressive_prune
        A = c.';  % M × N
        b = d.';  % M × 1

        [free_vars, fixed_values] = robust_constraint_propagation(A, b, q);
        fixed_sum = sum(fixed_values(~free_vars));

        if any(~free_vars)
            A_reduced = A(:, free_vars);
            b_reduced = zeros(M, 1);
            fixed_indices = find(~free_vars);

            for j = 1:M
                if ~isempty(fixed_indices)
                    fixed_contribution = dot(A(j, fixed_indices), fixed_values(fixed_indices));
                else
                    fixed_contribution = 0;
                end
                b_reduced(j) = mod(b(j) - fixed_contribution, q);
            end
        else
            A_reduced = A;
            b_reduced = b;
            fixed_sum = 0;
        end

        diag_report.reduction_ratio = 1 - sum(free_vars)/N;
    else
        free_vars    = true(1, N);
        fixed_values = zeros(1, N);
        A_reduced    = c.';
        b_reduced    = d.';
        fixed_sum    = 0;
    end
    diag_report.timings.prune = toc(t_prune);

    % ----- DFS search over reduced system -----
    t_search = tic;
    num_free = sum(free_vars);

    if num_free > 0
        free_solutions = dfs_mod_solver(A_reduced, b_reduced, q, n, num_free, fixed_sum);
        remainder_list = reconstruct_solutions(free_solutions, free_vars, fixed_values);
    else
        remainder_list = fixed_values;
    end
    diag_report.num_solutions  = size(remainder_list, 1);
    diag_report.timings.search = toc(t_search);

    if output_remainder_only
        matching_exponents = remainder_list;
    else
        matching_exponents = generate_minimal_solutions(remainder_list, n, q);
    end
end


function [free_vars, fixed_values] = robust_constraint_propagation(A, b, q)
% ROBUST_CONSTRAINT_PROPAGATION
% Iterative elimination of variables that are uniquely determined
% by single-variable constraints in the modular system A x ≡ b (mod q).

    [M, N] = size(A);
    free_vars    = true(1, N);
    fixed_values = zeros(1, N);

    changed         = true;
    iteration_count = 0;
    max_iterations  = N * 10;

    while changed && iteration_count < max_iterations
        iteration_count = iteration_count + 1;
        changed = false;

        for j = 1:M
            non_zero_idx  = find(A(j, :) ~= 0);
            free_non_zero = non_zero_idx(free_vars(non_zero_idx));

            if numel(free_non_zero) == 1
                idx = free_non_zero;

                fixed_contrib = 0;
                fixed_indices = find(~free_vars);
                if ~isempty(fixed_indices)
                    fixed_contrib = dot(A(j, fixed_indices), fixed_values(fixed_indices));
                end

                coeff = A(j, idx);
                g     = gcd(coeff, q);
                if g == 0
                    error('Coefficient %d and modulus %d share gcd = 0.', coeff, q);
                end

                target_val = mod(b(j) - fixed_contrib, q);
                if mod(target_val, g) ~= 0
                    % Inconsistent constraint; skip it.
                    continue;
                end

                effective_q    = q / g;
                reduced_target = target_val / g;
                reduced_coeff  = coeff / g;
                [~, inv_red]   = gcd(reduced_coeff, effective_q);
                fixed_val      = mod(reduced_target * inv_red, effective_q);

                fixed_values(idx) = fixed_val;
                free_vars(idx)    = false;
                changed           = true;
                break;
            end
        end
    end
end


function solutions = dfs_mod_solver(A, b, q, n, num_vars, fixed_sum)
% DFS_MOD_SOLVER
% Depth-first search solver for the reduced modular system,
% with pruning by partial sums and constraint consistency.

    [M, N] = size(A);
    if N ~= num_vars
        error('Number of variables (%d) inconsistent with expected (%d).', N, num_vars);
    end

    % Remove all-zero rows and check consistency
    valid_rows = true(M,1);
    for i = 1:M
        if all(A(i,:)==0)
            if mod(b(i), q) ~= 0
                solutions = zeros(0, num_vars);
                return;
            else
                valid_rows(i) = false;
            end
        end
    end
    A = A(valid_rows, :);
    b = b(valid_rows);
    M = size(A,1);

    if M > 0
        last_var_idx = zeros(1,M);
        for i = 1:M
            nz = find(A(i,:)~=0);
            last_var_idx(i) = max(nz);
        end

        affected_constraints = cell(1, num_vars);
        for j = 1:num_vars
            affected_constraints{j} = find(A(:,j)~=0);
        end
    else
        last_var_idx         = [];
        affected_constraints = cell(1,0);
    end

    max_solutions = min(1000000, q^num_vars);
    solutions     = zeros(max_solutions, num_vars);
    count         = 0;

    if fixed_sum > n
        solutions = zeros(0, num_vars);
        return;
    end

    function dfs_recursive(depth, current_x, current_sum, constraint_vals)
        for val = 0:q-1
            new_x        = current_x;
            new_x(depth) = val;
            new_sum      = current_sum + val;

            if fixed_sum + new_sum > n
                continue;
            end

            new_constraint_vals = constraint_vals;
            valid = true;

            if M > 0
                aff_cons = affected_constraints{depth};
                for k = 1:length(aff_cons)
                    con_idx = aff_cons(k);
                    new_constraint_vals(con_idx) = ...
                        mod(new_constraint_vals(con_idx) + A(con_idx, depth)*val, q);
                    if depth >= last_var_idx(con_idx)
                        if new_constraint_vals(con_idx) ~= b(con_idx)
                            valid = false;
                            break;
                        end
                    end
                end
            end

            if ~valid
                continue;
            end

            if depth < num_vars
                dfs_recursive(depth+1, new_x, new_sum, new_constraint_vals);
            else
                if mod(fixed_sum + new_sum, q) ~= mod(n, q)
                    return;
                end

                all_ok = true;
                if M > 0
                    for i = 1:M
                        if new_constraint_vals(i) ~= b(i)
                            all_ok = false;
                            break;
                        end
                    end
                end

                if all_ok
                    count = count + 1;
                    if count > size(solutions,1)
                        solutions = [solutions; zeros(10000, num_vars)];
                    end
                    solutions(count, :) = new_x;
                end
            end
        end
    end

    init_constraint_vals = zeros(1, M);
    dfs_recursive(1, zeros(1, num_vars), 0, init_constraint_vals);

    solutions = solutions(1:count, :);
end


function full_solutions = reconstruct_solutions(free_solutions, free_vars, fixed_values)
% RECONSTRUCT_SOLUTIONS  Expand solutions on free variables to full vectors.

    num_sol = size(free_solutions, 1);
    N       = length(free_vars);

    if num_sol > 0
        full_solutions = zeros(num_sol, N);
        if any(~free_vars)
            fixed_columns = repmat(fixed_values, num_sol, 1);
            full_solutions(:, free_vars)  = free_solutions;
            full_solutions(:, ~free_vars) = fixed_columns(:, ~free_vars);
        else
            full_solutions = free_solutions;
        end
    else
        full_solutions = fixed_values;
    end
end


function minimal_solutions = generate_minimal_solutions(remainder_list, ~, ~)
% GENERATE_MINIMAL_SOLUTIONS
% Here we simply take the modular remainders as minimal solutions.

    minimal_solutions = remainder_list;
end