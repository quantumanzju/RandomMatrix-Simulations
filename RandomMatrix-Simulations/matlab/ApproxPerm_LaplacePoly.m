%% ApproxPerm_LaplacePoly.m
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

clc; clear; close all;

%% ----------------- Parameters -----------------
numdim    = 15;    % matrix dimension N
numsample = 100;   % number of random samples

A          = cell(numsample,1);
true_perm  = zeros(numsample, 1);  % ground-truth permanents
Penc0      = zeros(numsample, 1);  % encoded estimate for A
Penc1      = zeros(numsample, 1);  % encoded estimate for A^T
r0         = zeros(numsample, 1);  % characteristic ratio r(A)
r1         = zeros(numsample, 1);  % characteristic ratio r(A^T)

% Dimension-aware global coefficients (trained offline)
% Each row is a polynomial in N; three rows correspond to [a_N, b_N, c_N]
coff = [15.291014416532047, -83.364932630744100; ...
        -31.598207421729818, 167.6832006771538; ...
        16.307121673277322,  -83.316091487811010];

%% ----------------- Exact permanents (Ryser) -----------------
tic;
for m = 1:numsample
    if mod(m,10) == 0
        fprintf('Computing true permanent (Ryser): sample %d / %d\n', m, numsample);
    end
    rng('shuffle');
    A{m} = rand(numdim, numdim);
    true_perm(m) = matrix_permanent(A{m});
end
elapsedTime_exact = toc;
fprintf('Ryser exact computation time: %.4f seconds\n', elapsedTime_exact);

%% ----------------- Encoded permanent + Laplace features -----------------
tic;
for m = 1:numsample
    if mod(m,10) == 0
        fprintf('Encoded permanent + Laplace: sample %d / %d\n', m, numsample);
    end

    % Original matrix A
    [tmp_Penc0, tmp_Perm_LP0] = matrix_perm(A{m});
    Penc0(m) = tmp_Penc0;
    r0(m)    = mean(tmp_Perm_LP0 ./ tmp_Penc0);  % characteristic ratio r(A)

    % Transposed matrix A^T
    [tmp_Penc1, tmp_Perm_LP1] = matrix_perm(A{m}');
    Penc1(m) = tmp_Penc1;
    r1(m)    = mean(tmp_Perm_LP1 ./ tmp_Penc1);  % r(A^T)
end
elapsedTime_approx = toc;
fprintf('Fast encoded approximation time: %.4f seconds\n', elapsedTime_approx);

%% ----------------- Ratios rho = P_enc / perm -----------------
rho0 = Penc0 ./ true_perm;  % for A
rho1 = Penc1 ./ true_perm;  % for A^T  (perm(A^T) = perm(A))

%% ----------------- Dimension-aware quadratic calibration g_N(r) -----------------
% For current N, compute coefficients [a_N, b_N, c_N] by evaluating coff polynomials
p0 = zeros(1,3);
p1 = zeros(1,3);
for k = 1:3
    p0(k) = polyval(coff(k,:), numdim);  % for A
    p1(k) = polyval(coff(k,:), numdim);  % for A^T (same N dependency)
end
% p0 = [a_N, b_N, c_N] for A;  p1 = [a_N, b_N, c_N] for A^T
a0 = p0(1); b0 = p0(2); c0 = p0(3);
a1 = p1(1); b1 = p1(2); c1 = p1(3);

fprintf('Dimension N = %d, calibrated coefficients (quadratic in r):\n', numdim);
fprintf('  For A   :  rho ~ %.6f * r^2 + %.6f * r + %.6f\n', a0, b0, c0);
fprintf('  For A^T :  rho ~ %.6f * r^2 + %.6f * r + %.6f\n', a1, b1, c1);

% Calibration functions g_N(r) for each sample
g0 = a0 * r0.^2 + b0 * r0 + c0;   % g_N(r(A))
g1 = a1 * r1.^2 + b1 * r1 + c1;   % g_N(r(A^T))

% Avoid extremely small denominators
eps_floor = 1e-8;
g0(abs(g0) < eps_floor) = eps_floor .* sign(g0(abs(g0) < eps_floor) + eps_floor);
g1(abs(g1) < eps_floor) = eps_floor .* sign(g1(abs(g1) < eps_floor) + eps_floor);

% One-sided calibrated estimators
Perm0_pred = Penc0 ./ g0;  % calibrated estimate using A
Perm1_pred = Penc1 ./ g1;  % calibrated estimate using A^T

% Combined estimator (A and A^T)
disp('Combine A and A^T (simple average)');
Perm_pred = (Perm0_pred + Perm1_pred) / 2;

%% ----------------- Error statistics -----------------
% Relative error in percentage
err_raw0   = 100 * (Penc0 ./ true_perm - 1);      % uncalibrated (A)
err_raw1   = 100 * (Penc1 ./ true_perm - 1);      % uncalibrated (A^T)
err_cal0   = 100 * (Perm0_pred ./ true_perm - 1); % calibrated (A)
err_cal1   = 100 * (Perm1_pred ./ true_perm - 1); % calibrated (A^T)
err_comb   = 100 * (Perm_pred   ./ true_perm - 1);% calibrated combined

% Focus on combined calibrated estimator (to match Chapter 5 text)
error = err_comb;

RMSE   = sqrt(mean(error.^2));
MAE    = mean(abs(error));
MaxAE  = max(abs(error));
StdErr = std(error);

fprintf('=== Error statistics for combined calibrated estimator (A & A^T) ===\n');
fprintf('RMSE      = %.4f %%\n', RMSE);
fprintf('MAE       = %.4f %%\n', MAE);
fprintf('MaxAE     = %.4f %%\n', MaxAE);
fprintf('Std(error)= %.4f %%\n', StdErr);

%% ----------------- Correlation and R^2 for rho vs r -----------------
% Pearson correlation between r and rho
corr_r_rho_A   = corr(r0, rho0);
corr_r_rho_AT  = corr(r1, rho1);

% R^2 for quadratic fit rho ~ g_N(r) on A
rho_fit0 = a0 * r0.^2 + b0 * r0 + c0;
SS_res0  = sum((rho0 - rho_fit0).^2);
SS_tot0  = sum((rho0 - mean(rho0)).^2);
R2_A     = 1 - SS_res0 / SS_tot0;

% R^2 for quadratic fit on A^T
rho_fit1 = a1 * r1.^2 + b1 * r1 + c1;
SS_res1  = sum((rho1 - rho_fit1).^2);
SS_tot1  = sum((rho1 - mean(rho1)).^2);
R2_AT    = 1 - SS_res1 / SS_tot1;

fprintf('--- Correlation and R^2 between r and rho (N = %d) ---\n', numdim);
fprintf('corr(r(A),  rho(A))      = %.6f\n', corr_r_rho_A);
fprintf('corr(r(A^T),rho(A^T))    = %.6f\n', corr_r_rho_AT);
fprintf('R^2 for quadratic fit (A)   = %.6f\n', R2_A);
fprintf('R^2 for quadratic fit (A^T) = %.6f\n', R2_AT);

%% ----------------- Figures for the paper (English labels) -----------------

% 1) rho vs r scatter + fitted curve for A (with corr and R^2 in title)
figure;
scatter(r0, rho0, 40, 'filled'); hold on;
r_line  = linspace(min(r0), max(r0), 200);
rho_fit = a0 * r_line.^2 + b0 * r_line + c0;
plot(r_line, rho_fit, 'r', 'LineWidth', 1.5);
xlabel('Laplace characteristic ratio r(A)', 'FontSize', 12);
ylabel('Encoding-truth ratio \rho(A) = P_{enc}(A) / perm(A)', 'FontSize', 12);
title(sprintf(['N = %d: r(A) vs \\rho(A)\n', ...
               'corr = %.4f,  R^2 (quadratic) = %.4f'], ...
              numdim, corr_r_rho_A, R2_A), ...
      'FontSize', 12);
legend({sprintf('Samples (N = %d)', numdim), ...
        'Quadratic fit g_N(r)'}, ...
       'Location', 'best');
grid on;
set(gca, 'FontSize', 12);
% print(gcf, sprintf('rho_vs_r_N%d.png', numdim), '-dpng', '-r300');

% 2) Relative error vs sample index (combined calibrated)
figure;
plot(1:numsample, error, 'o-', 'LineWidth', 1.2, 'MarkerSize', 5);
xlabel('Sample index', 'FontSize', 12);
ylabel('Relative error (%)', 'FontSize', 12);
title(sprintf(['Combined calibrated estimator (A and A^T), N = %d\n', ...
               'RMSE = %.3f%%, MaxAE = %.3f%%'], ...
              numdim, RMSE, MaxAE), ...
      'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);
% print(gcf, sprintf('error_curve_combined_N%d.png', numdim), '-dpng', '-r300');

% 3) Histogram of relative errors (combined calibrated)
figure;
histogram(error, 20, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
xlabel('Relative error (%)', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title(sprintf(['Error distribution of combined calibrated estimator, N = %d\n', ...
               'RMSE = %.3f%%, MAE = %.3f%%'], ...
              numdim, RMSE, MAE), ...
      'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);
% print(gcf, sprintf('error_hist_combined_N%d.png', numdim), '-dpng', '-r300');

% 4) Compare uncalibrated vs calibrated errors (boxplot)
figure;
boxplot([err_raw0, err_cal0, err_comb], ...
        'Labels', {'Raw(A)', 'Calibrated(A)', 'Calibrated combined'});
ylabel('Relative error (%)', 'FontSize', 12);
title(sprintf('Comparison of different estimators, N = %d', numdim), 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);
% print(gcf, sprintf('error_boxplot_N%d.png', numdim), '-dpng', '-r300');


%% ======================= Supporting functions =======================

function [Perm, Perm_Laplace] = matrix_perm(matrix)

    % 获取输入矩阵的行数
    numdim = size(matrix, 1);   
    q = numdim + 1;
    m = 1;
    seq_size = q^m;
    x1 = qary_rm_linear(q, m);

    rng(133);
%     rng('shuffle');                            % 基于当前时间设置随机种子，确保每次不同
    sel_seq = random_select(seq_size, numdim);
    signMatrix_sub = x1(sel_seq, :);    

    omega = exp(2* 1i * pi/ q);                % q进制单位根 (360°/q相位)
    signMatrix_sub = omega.^signMatrix_sub;    % 映射为q进制调制信号

    Perm = matrix_permS(matrix, signMatrix_sub);
    Perm_Laplace = matrix_perPL(matrix, signMatrix_sub);
    
end

function Perm_Laplace = matrix_perPL(matrix, signMatrix_sub)
    k = size(matrix,1);
    Perm_Laplace = zeros(1,k);
    % 预先计算所有可能的列索引组合（避免在循环中重复计算）
    cols_index = arrayfun(@(col) [1:col-1, col+1:k], 1:k, 'UniformOutput', false);
    % 预先计算每个col对应的子符号矩阵（避免在循环中重复删除行）
    precomputed_sign_sub = cell(1, k);
    for col = 1:k
        precomputed_sign_sub{col} = signMatrix_sub(setdiff(1:k, col), :);
    end
    for row = 1:k
        % 预先计算当前行对应的行索引（避免在列循环中重复计算）
        row_sub_index = setdiff(1:k, row);
        row_sub_matrix = matrix(row_sub_index, :);  % 预提取子矩阵的行
        for col = 1:k
            % 使用预计算的列索引（比setdiff快）
            col_sub_index = cols_index{col};
            % 直接索引替代setdiff（效率提升关键）
            sub_matrix = row_sub_matrix(:, col_sub_index);
            % 使用预计算的子符号矩阵（避免重复删除行）
            sub_signMatrix_sub = precomputed_sign_sub{col};
            % 调用独立函数计算子矩阵的永久
            sub_Perm = matrix_permS(sub_matrix, sub_signMatrix_sub);
            % 累加计算结果
            Perm_Laplace(row) = Perm_Laplace(row) + matrix(row, col) * sub_Perm;
        end
    end
end

function Perm = matrix_permS(matrix, signMatrix_sub)
    % 计算矩阵积和式 (Permanent) 原始特征函数实现
    dim = size(matrix, 1);
    numbit = size(signMatrix_sub, 2);    
    % 并行计算核心操作
    column_sums = sum(signMatrix_sub, 1);  % 1×M向量
    verify_JC = column_sums .^ dim;        % 直接计算验证矩阵乘积 (1×M)    
    S = matrix * signMatrix_sub;           % 并行矩阵乘法 (N×M)
    lams = prod(signMatrix_sub, 1);        % 并行列乘积 (1×M)
    JC = prod(S, 1);                       % 并行列乘积 (1×M)    
    % 标量计算
    correlation = JC * conj(lams)';
    verify_correlation = verify_JC * conj(lams)';
    verify = verify_correlation / (numbit * factorial(dim));
    Perm = correlation / numbit;
    Perm = real(Perm / verify);
end

function perm = matrix_permanent(A)
% 计算矩阵永久（Permanent）的正确Ryser公式实现
% 输入: A - n x n 矩阵
% 输出: perm - 矩阵永久值
    n = size(A, 1);    
    % 检查是否为方阵
    if size(A, 1) ~= size(A, 2)
        error('输入矩阵必须是方阵');
    end    
    % 处理特殊情况
    if n == 0
        perm = 1;  % 空矩阵的永久定义为1
        return;
    end    
    % 如果矩阵有零行或零列，永久为0
    if any(all(A == 0, 1)) || any(all(A == 0, 2))
        perm = 0;
        return;
    end    
    total = 0;    
    % 遍历所有非空子集 S ⊆ {1, 2, ..., n}
    % 注意：从1开始，跳过空集（k=0）
    for k = 1:n
        % 生成所有大小为k的子集
        combinations = nchoosek(1:n, k);        
        for idx = 1:size(combinations, 1)
            S = combinations(idx, :);            
            % 计算每行在子集S上的和
            row_sums = zeros(n, 1);
            for i = 1:n
                row_sums(i) = sum(A(i, S));
            end            
            % 计算乘积
            prod_val = prod(row_sums);            
            % 根据Ryser公式，符号为(-1)^k
            sign_val = (-1)^k;            
            total = total + sign_val * prod_val;
        end
    end    
    % 最终符号调整：perm(A) = (-1)^n * 上述求和
    perm = (-1)^n * total;
end

function codewords = qary_rm_linear(q, m)
    % 线性 Reed--Muller q-ary 码字生成
    qm = q^m;    
    % 生成所有输入向量 x ∈ Z_q^m
    inputs = zeros(qm, m);
    for i = 1:qm
        idx = i-1;
        for j = m:-1:1
            inputs(i, j) = mod(floor(idx/q^(j-1)), q);
        end
    end    
    % 生成所有系数向量 a ∈ Z_q^m
    coeffs = zeros(qm, m);
    for i = 1:qm
        idx = i-1;
        for j = m:-1:1
            coeffs(i, j) = mod(floor(idx/q^(j-1)), q);
        end
    end    
    % 计算线性函数值 (矩阵运算优化)
    codewords = mod(coeffs * inputs', q);
end

function selected = random_select(N, k)
    % 从1到N中随机选取k个不重复整数
    if k > N
        error('k 不能大于 N');
    end    
    full_set = 1:N;
    perm = full_set(randperm(N));    
    selected = perm(1:k);    
end