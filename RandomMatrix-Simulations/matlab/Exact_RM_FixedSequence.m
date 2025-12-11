%% Exact_RM_FixedSequence.m
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
% (sel_seq, coeff1) have been found for the given dimension.

clc;
clear;
close all;

%% ----- Parameters -----
q        = 2;    % alphabet size (binary phases)
m        = 15;   % RM code parameter (code length = q^m)
numPorts = 19;   % matrix dimension N

%% ----- Generate q-ary RM linear code -----
x1      = qary_rm_linear(q, m);
lengths = size(x1, 1);

%% ----- Test matrix and exact permanent (Ryser) -----
rng('shuffle');
A            = rand(numPorts, numPorts);    % test matrix (can be arbitrary N×N)
perm_correct = matrix_permanent(A);        % ground-truth permanent

%% ----- Fixed RM codeword indices (calibrated once for this N) -----
% The index set 'sel_seq' encodes a specific choice of RM codewords that
% has been found (offline, via multiple experiments) to admit a clean
% reconstruction through a pre-computed coefficient matrix 'coeff1'.
%
% Once such a pair (sel_seq, coeff1) is fixed for a given N, we no longer
% need to recompute the modular phase exponents phase_order.

sel_seq = [ ...
    6755  19312 15193 15759 16959 17056 17890 31004 ...
    1498   6923 18096 28749  2542   261 20519 21946 ...
    6126  28166 21342 ];

if numel(sel_seq) ~= numPorts
    error('Length of sel_seq (%d) must equal numPorts (%d).', ...
          numel(sel_seq), numPorts);
end

%% ----- Construct phase-encoding matrix from fixed codewords -----
sign_index     = x1(sel_seq, :);                 % integer codewords in Z_q
omega          = exp(2 * 1i * pi / q);
signMatrix_sub = omega .^ sign_index;            % numPorts × L phase matrix
lams           = prod(signMatrix_sub, 1);        %#ok<NASGU> % column products λ_j (not used here)

%% ----- Phase-encoding estimate of perm(A) -----
perm = matrix_perm(A, signMatrix_sub);

%% ----- Column-wise π phase shifts and encoded permanents perm_c -----
% For each column j, we multiply column j of A by e^{iπ} and recompute
% the phase-encoded permanent. This yields a vector perm_c of length N
% that will serve as the right-hand side of a small linear system.

perm_c = zeros(1, numPorts);
for col = 1:numPorts
    th         = pi;
    As         = A;
    As(:, col) = exp(1i * th) * As(:, col);
    perm_c(col) = matrix_perm(As, signMatrix_sub);
end

%% ----- Pre-computed coefficient matrix for this dimension -----
% coeff1 was originally derived from the modular phase exponents phase_order
% via coeff1 = 1 - 2 * phase_order (for q = 2). For this particular choice
% of (N, m, sel_seq), coeff1 can be *pre-stored* and reused for any A.

coeff1 = [ ...
     1   1   1   1  -1  -1  -1  -1;
     1   1  -1  -1   1   1  -1  -1;
     1  -1   1  -1   1  -1   1  -1;
    -1  -1  -1  -1  -1  -1  -1  -1;
    -1  -1   1   1   1   1  -1  -1;
     1  -1   1  -1   1  -1   1  -1;
    -1  -1  -1  -1  -1  -1  -1  -1;
    -1   1  -1   1   1  -1   1  -1;
    -1  -1  -1  -1  -1  -1  -1  -1;
    -1   1   1  -1  -1   1   1  -1;
    -1  -1   1   1   1   1  -1  -1;
    -1   1   1  -1  -1   1   1  -1;
    -1  -1   1   1   1   1  -1  -1;
     1   1  -1  -1   1   1  -1  -1;
     1  -1   1  -1   1  -1   1  -1;
    -1   1   1  -1  -1   1   1  -1;
     1   1  -1  -1   1   1  -1  -1;
     1  -1  -1   1  -1   1   1  -1;
    -1  -1  -1  -1  -1  -1  -1  -1 ];

% Sanity check: C has size m_eq × n_var for the small linear system
C = coeff1;                % size: (m_eq × n_var)
perm_c_mr = perm_c;        % right-hand side (row)

%% ----- Solve linear system and reconstruct permanent -----
% We solve a small (possibly over-/under-determined) system
%    C * X ≈ perm_c^T
% with an additional sum constraint S = perm (phase-encoded estimate).
% The solver returns only the last component X_n, which empirically
% matches the true permanent for this fixed setup.

X_sol = solve_system(C, perm_c_mr.', perm);

%% ----- Numerical comparison -----
fprintf('\n===== Fixed-sequence reconstruction (N = %d) =====\n', numPorts);
fprintf('Ryser exact permanent                                  = % .16e\n', perm_correct);
fprintf('Reconstructed X_sol                                    = % .16e\n', X_sol);
fprintf('X_sol / perm_correct                                   = % .16e\n', X_sol / perm_correct);
fprintf('Relative error |X_sol - perm_correct| / |perm_correct| = % .3e\n', ...
        abs(X_sol - perm_correct) / max(1, abs(perm_correct)));
fprintf('=================================================\n');


%% =====================================================================
%  Supporting functions (identical to the general version)
%% =====================================================================

function Perm = matrix_perm(matrix, signMatrix_sub)
    % MATRIX_PERM  Phase-encoding based estimator of the matrix permanent.
    L    = size(signMatrix_sub, 2);
    S    = matrix * signMatrix_sub;
    lams = prod(signMatrix_sub, 1);
    JC   = prod(S, 1);
    corr = JC * conj(lams).';
    Perm = corr / L;
end

function perm = matrix_permanent(A)
    % MATRIX_PERMANENT  Exact permanent via Ryser's formula.
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
    qm     = q^m;
    inputs = zeros(qm, m);
    for i = 1:qm
        idx = i-1;
        for j = m:-1:1
            inputs(i, j) = mod(floor(idx/q^(j-1)), q);
        end
    end
    coeffs = zeros(qm, m);
    for i = 1:qm
        idx = i-1;
        for j = m:-1:1
            coeffs(i, j) = mod(floor(idx/q^(j-1)), q);
        end
    end
    codewords = mod(coeffs * inputs.', q);
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
