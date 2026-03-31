% =========================================================================
% CASE 2: Online Foreground-Background Separation
% Comparison: GeRoST (Proposed) vs. GRASTA vs. GREAT
% =========================================================================

clear; clc; close all;

% Apply IEEE Plot Settings
ieee_settings();

%% 1. Path Setup
% Add GeRoST class and utils
addpath('../'); 
if exist('../utils/', 'dir'), addpath('../utils/'); end
if exist('./results/', 'dir') == 0, mkdir('results'); end

% Add GRASTA path (Assumes grasta is cloned in a parallel directory)
grasta_path = '../../grasta/';
if exist(grasta_path, 'dir')
    addpath(genpath(grasta_path));
else
    warning('GRASTA directory not found at %s. Please update the path if you want to run the GRASTA comparison.', grasta_path);
end

%% 2. Synthetic Data Generation (Dynamic BG + Random Walk FG)
fprintf('\n--- Generating synthetic video data ---\n');

H = 64; W = 64;           % Image dimensions
n = H * W;                % Ambient dimension
N = 300;                  % Total number of frames
k = 5;                    % True background subspace dimension
T_init = 10;              % Initial frames strictly for SVD

rng(42);
[UV, ~] = qr(randn(n, 2*k), 0);
U0 = UV(:, 1:k);
V0 = UV(:, k+1:2*k);

U_true = zeros(n, k, N);
bg_true = zeros(n, N);
% Storage for sensitivity plot
rho_log = NaN(1, N);
lambda_log = NaN(1, N);

for t = 1:N
    theta = 0.5 * sin(2 * pi * t / N); 
    U_t = U0 * cos(theta) + V0 * sin(theta);
    U_true(:, :, t) = U_t;
    
    w_t = 10 * randn(k, 1);    
    bg_true(:, t) = U_t * w_t;
end

fg_true = zeros(n, N);
x_pos = 20; y_pos = 20;   
bw = 10; bh = 10;         
fg_intensity = 5.0;       

for t = 51:N 
    x_pos = x_pos + randi([-3, 3]);
    y_pos = y_pos + randi([-3, 3]);
    x_pos = max(1, min(W - bw + 1, x_pos));
    y_pos = max(1, min(H - bh + 1, y_pos));
    
    frame_fg = zeros(H, W);
    frame_fg(y_pos:y_pos+bh-1, x_pos:x_pos+bw-1) = fg_intensity;
    fg_true(:, t) = frame_fg(:);
end

noise_level = 0.01; 
Y = bg_true + fg_true + noise_level * randn(n, N);

%% 3. Tracker Initialization
fprintf('\n--- Initializing Trackers (Warm-up: Frames 1 to %d) ---\n', T_init);

[U_init, ~, ~] = svd(Y(:, 1:T_init), 'econ');
U0_hat = U_init(:, 1:k);

% Verify the baseline SVD actually found the true subspace
err_init = chordalDist(U_true(:,:,T_init), U0_hat);
fprintf('Baseline SVD Error at Frame %d: %.4f\n', T_init, err_init);

% --- Initialize GeRoST ---
d_est = 7; % Over-parameterize (d > k) to isolate outlier dimensions in the SVD window
window_len = 30;

% Initialize GeRoST (rho will be adaptively overridden inside the loop)
% Note: 'K', 5 specifies that the tracker will internally run 5 gradient 
% descent loops per frame during descent_step().
gerost_tracker = gerost(n, k, d_est, window_len, 'K', 5, 'rho', 0.1, 'max_steps', N);
gerost_tracker = gerost_tracker.initialize(U0_hat);

% --- Initialize GREAT ---
great_tracker = great(n, k, window_len, 'K', 5, 'max_steps', N);
great_tracker = great_tracker.initialize(U0_hat);

% --- Initialize GRASTA ---
run_grasta = exist('grasta_stream', 'file') || exist('grasta', 'file') || exist('run_grasta', 'file');
if run_grasta
    try
        OPTIONS.DIM_M = n;
        OPTIONS.RANK = k;
        OPTIONS.ITER_MIN = 5;
        OPTIONS.ITER_MAX = 20;
        OPTIONS.TOL = 1e-7;
        OPTIONS.USE_MEX = 0; 
        
        U_grasta = U0_hat;
        STATUS_grasta.init       = 1; 
        STATUS_grasta.curr_iter  = 0;
        STATUS_grasta.last_mu    = 1;
        STATUS_grasta.level      = 0;
        
        % Prevent division-by-zero on the converged, clean frame 11 gradient 
        % by seeding a safe, stable initial step scale.
        STATUS_grasta.step_scale = 0.01; 
        
        STATUS_grasta.last_w     = zeros(k, 1);
        STATUS_grasta.last_gamma = zeros(n, 1);
        
        OPTS_grasta.TOL      = OPTIONS.TOL;
        OPTS_grasta.MAX_ITER = OPTIONS.ITER_MIN;
        OPTS_grasta.QUIET    = 1;
        OPTS_grasta.RHO      = 1.8;
    catch
        warning('GRASTA initialization failed. Bypassing GRASTA loop.');
        run_grasta = false;
    end
end

% Set up storage
bg_gerost = zeros(n, N);
fg_gerost = zeros(n, N);
err_gerost = NaN(1, N);
err_gerost(1:T_init) = err_init;

bg_great = zeros(n, N);
fg_great = zeros(n, N);
err_great = NaN(1, N);
err_great(1:T_init) = err_init;

if run_grasta
    bg_grasta = zeros(n, N);
    fg_grasta = zeros(n, N);
    err_grasta = NaN(1, N);
    err_grasta(1:T_init) = err_init;
end

%% 4. Online Subspace Tracking Loop
fprintf('\n--- Starting Online Tracking Loop (Frames %d to %d) ---\n', T_init + 1, N);

for t = T_init+1:N
    y_t = Y(:, t);
    
    % --- [1] GeRoST Tracking Step ---
    % Dynamically compute instantaneous NSR (p_t) from tracking residual
    res_norm = norm(y_t - gerost_tracker.U * (gerost_tracker.U' * y_t));
    signal_norm = max(norm(gerost_tracker.U' * y_t), 1e-6);
    p_t = res_norm / signal_norm;
    
    % Safely retrieve the window's k-th singular value
    if ~isempty(gerost_tracker.sigma_vals_w) && length(gerost_tracker.sigma_vals_w) >= k
        sigma_k = gerost_tracker.sigma_vals_w(k);
    else
        sigma_k = signal_norm * sqrt(window_len); % Approximation for the very first frame
    end
    sigma_k = max(sigma_k, 1e-6);
    
    % Expand Grassmannian uncertainty ball gracefully for outliers
    % Hard-cap the radius to prevent tracking the outlier
    p_t_capped = min(p_t, 0.15); % Cap NSR explicitly
    rho_t = (sqrt(2) * p_t_capped) / (1 - p_t_capped) + sqrt(d_est-k); % Max rho_t ~ 1.414+0.25
    
    % Inject adaptive rho into tracker
    gerost_tracker.rho_param = rho_t;
    
    % Execute Tracking (Internally runs K=5 gradient descent loops)
    gerost_tracker = gerost_tracker.descent_step(y_t);
    U_gerost = gerost_tracker.U;

    rho_log(t) = rho_t;
    lambda_log(t) = gerost_tracker.lambda_star; 
    
    [bg_g, fg_g] = extract_fg_bg_ista(y_t, U_gerost, 1.0);
    bg_gerost(:, t) = bg_g;
    fg_gerost(:, t) = fg_g;
    err_gerost(t) = chordalDist(U_true(:,:,t), U_gerost);
    
    % --- [2] GREAT Tracking Step ---
    great_tracker = great_tracker.descent_step(y_t);
    U_great = great_tracker.U;
    
    [bg_grt, fg_grt] = extract_fg_bg_ista(y_t, U_great, 1.0);
    bg_great(:, t) = bg_grt;
    fg_great(:, t) = fg_grt;
    err_great(t) = chordalDist(U_true(:,:,t), U_great);

    % --- [3] GRASTA Tracking Step ---
    if run_grasta
        try
            idx_full = (1:n)';
            [U_grasta, STATUS_grasta, OPTS_grasta] = grasta_stream(y_t, idx_full, U_grasta, STATUS_grasta, OPTIONS, OPTS_grasta);
            
            bg_grasta(:, t) = U_grasta * STATUS_grasta.w;
            fg_grasta(:, t) = STATUS_grasta.s_t; 
            err_grasta(t) = chordalDist(U_true(:,:,t), U_grasta);
        catch ME
            warning('GRASTA step failed at frame %d: %s', t, ME.message);
            err_grasta(t) = NaN;
            run_grasta = false; 
        end
    end
    
    % --- Diagnostic Logging ---
    if mod(t, 20) == 0 || t == T_init + 1 || t == 51 || t == 52
        if run_grasta
            fprintf('Frame %3d | GeRoST Err: %.4f | GRASTA Err: %.4f | GREAT Err: %.4f\n', ...
                t, err_gerost(t), err_grasta(t), err_great(t));
        else
            fprintf('Frame %3d | GeRoST Err: %.4f | GRASTA: Not Run | GREAT Err: %.4f\n', ...
                t, err_gerost(t), err_great(t));
        end
        if t == 11 || t == 51 || t == 52
            fprintf('    [Diag t=%d] GeRoST Instantaneous p_t: %.3f -> Computed rho_t: %.3f\n', t, p_t, rho_t);
        end
    end
end

fprintf('\nTracking complete!\n');

%% 5. Visualization and Metrics
fprintf('Generating plots...\n');

figure('Name', 'Subspace Tracking Error', 'Position', [100, 100, 600, 400]);
plot(1:N, err_gerost, 'b-', 'LineWidth', 2, 'DisplayName', 'GeRoST');
hold on;
plot(1:N, err_great, 'g-.', 'LineWidth', 2, 'DisplayName', 'GREAT');
if run_grasta
    plot(1:N, err_grasta, 'r--', 'LineWidth', 2, 'DisplayName', 'GRASTA');
end
xline(50, 'k:', 'LineWidth', 1.5, 'DisplayName', 'Occlusion Starts');
xlabel('Frame Index (t)', 'Interpreter', 'latex');
ylabel('Chordal Distance $d_c(\mathcal{U}_t, \hat{\mathcal{U}}_t)$', 'Interpreter', 'latex');
title('Subspace Tracking Error under Moving Occlusions', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
grid on;
saveas(gcf, 'results/subspace_error_case2.png');

visual_frame = 250;
figure('Name', 'FG-BG Separation', 'Position', [150, 150, 1200, 800]);

subplot(3, 3, 1); imshow(reshape(Y(:, visual_frame), H, W), []); title('Original Frame');
subplot(3, 3, 2); imshow(reshape(bg_true(:, visual_frame), H, W), []); title('True BG');
subplot(3, 3, 3); imshow(reshape(fg_true(:, visual_frame), H, W), []); title('True Sparse FG');

subplot(3, 3, 4); imshow(reshape(bg_gerost(:, visual_frame), H, W), []); title('GeRoST BG');
subplot(3, 3, 5); imshow(reshape(fg_gerost(:, visual_frame), H, W), []); title('GeRoST FG');

if run_grasta
    subplot(3, 3, 6); imshow(reshape(fg_grasta(:, visual_frame), H, W), []); title('GRASTA FG');
else
    subplot(3, 3, 6); text(0.1, 0.5, 'GRASTA Not Run'); axis off;
end

subplot(3, 3, 7); imshow(reshape(bg_great(:, visual_frame), H, W), []); title('GREAT BG');
subplot(3, 3, 8); imshow(reshape(fg_great(:, visual_frame), H, W), []); title('GREAT FG');
saveas(gcf, 'results/fg_bg_visuals_case2.png');

% C. ROC Curve for Foreground Separation
fprintf('Computing ROC and AUC...\n');
eval_frames = 51:N; % Evaluate only when occlusion is present
mask_true = fg_true(:, eval_frames) > 0;

% Use the projection residuals as the anomaly score
score_gerost = abs(Y(:, eval_frames) - bg_gerost(:, eval_frames));
score_great  = abs(Y(:, eval_frames) - bg_great(:, eval_frames));

if run_grasta
    score_grasta = abs(Y(:, eval_frames) - bg_grasta(:, eval_frames));
    max_score = max([score_gerost(:); score_grasta(:); score_great(:)]);
else
    max_score = max([score_gerost(:); score_great(:)]);
end

% Generate thresholds
thresholds = linspace(0, max_score, 100);

tpr_gerost = zeros(1, 100); fpr_gerost = zeros(1, 100);
tpr_great  = zeros(1, 100); fpr_great  = zeros(1, 100);
tpr_grasta = zeros(1, 100); fpr_grasta = zeros(1, 100);

for i = 1:100
    th = thresholds(i);
    
    % GeRoST metrics
    pred_g = score_gerost > th;
    tp_g = sum(pred_g(:) & mask_true(:));
    fp_g = sum(pred_g(:) & ~mask_true(:));
    fn_g = sum(~pred_g(:) & mask_true(:));
    tn_g = sum(~pred_g(:) & ~mask_true(:));
    tpr_gerost(i) = tp_g / max(tp_g + fn_g, 1);
    fpr_gerost(i) = fp_g / max(fp_g + tn_g, 1);
    
    % GREAT metrics
    pred_grt = score_great > th;
    tp_grt = sum(pred_grt(:) & mask_true(:));
    fp_grt = sum(pred_grt(:) & ~mask_true(:));
    fn_grt = sum(~pred_grt(:) & mask_true(:));
    tn_grt = sum(~pred_grt(:) & ~mask_true(:));
    tpr_great(i) = tp_grt / max(tp_grt + fn_grt, 1);
    fpr_great(i) = fp_grt / max(fp_grt + tn_grt, 1);
    
    % GRASTA metrics
    if run_grasta
        pred_gr = score_grasta > th;
        tp_gr = sum(pred_gr(:) & mask_true(:));
        fp_gr = sum(pred_gr(:) & ~mask_true(:));
        fn_gr = sum(~pred_gr(:) & mask_true(:));
        tn_gr = sum(~pred_gr(:) & ~mask_true(:));
        tpr_grasta(i) = tp_gr / max(tp_gr + fn_gr, 1);
        fpr_grasta(i) = fp_gr / max(fp_gr + tn_gr, 1);
    end
end

% Sort for plotting and AUC
[fpr_gerost, sort_idx] = sort(fpr_gerost); tpr_gerost = tpr_gerost(sort_idx);
auc_gerost = trapz(fpr_gerost, tpr_gerost);

[fpr_great, sort_idx] = sort(fpr_great); tpr_great = tpr_great(sort_idx);
auc_great = trapz(fpr_great, tpr_great);

figure('Name', 'ROC Curve', 'Position', [200, 200, 500, 450]);
plot(fpr_gerost, tpr_gerost, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('GeRoST (AUC = %.3f)', auc_gerost));
hold on;
plot(fpr_great, tpr_great, 'g-.', 'LineWidth', 2, 'DisplayName', sprintf('GREAT (AUC = %.3f)', auc_great));
if run_grasta
    [fpr_grasta, sort_idx] = sort(fpr_grasta); tpr_grasta = tpr_grasta(sort_idx);
    auc_grasta = trapz(fpr_grasta, tpr_grasta);
    plot(fpr_grasta, tpr_grasta, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('GRASTA (AUC = %.3f)', auc_grasta));
end
plot([0, 1], [0, 1], 'k:', 'DisplayName', 'Random');
xlabel('False Positive Rate (FPR)', 'Interpreter', 'latex');
ylabel('True Positive Rate (TPR)', 'Interpreter', 'latex');
title('ROC for Foreground Separation', 'Interpreter', 'latex');
legend('Location', 'southeast', 'Interpreter', 'latex');
grid on;
saveas(gcf, 'results/roc_case2.png');

figure('Position', [100 100 560 280]);
yyaxis left
plot(1:N, rho_log, 'b-', 'LineWidth', 1.5);
ylabel('$\rho_t$', 'Interpreter', 'latex');
ylim([0 0.5]);

yyaxis right
plot(1:N, lambda_log, 'r--', 'LineWidth', 1.5);
ylabel('$\lambda^*_t$', 'Interpreter', 'latex');
ylim([2 inf]);

xline(50, 'k:', 'LineWidth', 1.2, ...
    'Label', 'Occlusion Starts', ...
    'Interpreter', 'latex');
yline(2, 'r:', 'LineWidth', 1.0); 
% uniqueness threshold lambda* > 2

xlabel('Frame Index $t$', 'Interpreter', 'latex');
title('Adaptive $\rho_t$ and $\lambda^*_t$ Evolution', ...
    'Interpreter', 'latex');
legend({'$\rho_t$', '$\lambda^*_t$'}, ...
    'Interpreter', 'latex', 'Location', 'northeast');
grid on;
saveas(gcf, 'results/rho_lambda_evolution.png');


%% ========================================================================
% HELPER FUNCTIONS
% =========================================================================

function [bg, fg] = extract_fg_bg_ista(y, U, lambda)
    w = U' * y; 
    for iter = 1:5
        fg = y - U * w;
        fg = sign(fg) .* max(abs(fg) - lambda, 0);
        w = U' * (y - fg);
    end
    bg = U * w;
end