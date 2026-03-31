%% GREAT - MATLAB Implementation Example
% This script demonstrates the GREAT algorithm and compares it with GeRoST
% using sliding window logic

addpath('./utils');
s = plot_settings();  % Load IEEE plot settings

%% Setup: Common parameters for both algorithms
% Problem dimensions
n = 50;        % ambient dimension
k = 10;         % true subspace dimension
T = 20;        % window length
K = 15;         % inner gradient descent iterations

t_0 = 50;  % data collection duration
num_steps = 100;
num_trials = 5;  % number of Monte Carlo trials for error bars

% Storage for results across multiple trials
errors_gerost_trials = zeros(num_trials, num_steps);
errors_great_trials = zeros(num_trials, num_steps);

fprintf('\nRunning %d Monte Carlo trials...\n', num_trials);

for trial = 1:num_trials
    fprintf('Trial %d/%d: ', trial, num_trials);
    
    % Generate true subspace (new for each trial)
    U_true = randn(n, k);
    U_true = orth(U_true);
    
    % Phase 1: Data collection (t = 0 to t_0)
    data_collection = zeros(n, t_0);
    
    for t = 1:t_0
        coeff = randn(k, 1);
        noise = 0.01 * randn(n, 1);
        u = U_true * coeff + noise;
        data_collection(:, t) = u;
    end
    
    % Phase 2: Compute initial subspace estimate from collected data
    [U_init, ~, ~] = svd(data_collection, 'econ');
    U_init = U_init(:, 1:k);
    
    % Phase 3: Initialize both trackers
    tracker_gerost = gerost(n, k, 8, T, 'K', K, 'rho', 0.1, 'fallback', 'data', ...
        'max_steps', num_steps);
    tracker_gerost = initialize(tracker_gerost, U_init);
    
    tracker_great = great(n, k, T, 'K', K, 'alpha', [], 'max_steps', num_steps);
    tracker_great = initialize(tracker_great, U_init);
    
    % Phase 4: Online tracking with sliding window
    for t = 1:num_steps
        % Generate data vector: mostly from true subspace + small noise
        coeff = randn(k, 1);
        noise = 0.01 * randn(n, 1);
        u = U_true * coeff + noise;
        
        % Process with GeRoST (using sliding window internally)
        tracker_gerost = tracker_gerost.descent_step(u);
        
        % Process with GREAT (using sliding window internally)
        tracker_great = tracker_great.descent_step(u);
        
        % Compute errors
        errors_gerost_trials(trial, t) = chordalDist(tracker_gerost.U, U_true);
        errors_great_trials(trial, t) = chordalDist(tracker_great.U, U_true);
    end
    fprintf('done\n');
end

% Aggregate results
error_gerost_mean = mean(errors_gerost_trials, 1);
error_great_mean = mean(errors_great_trials, 1);
error_gerost_std = std(errors_gerost_trials, 0, 1);
error_great_std = std(errors_great_trials, 0, 1);

%% Compute rolling window statistics
window_size = 20;
rolling_gerost_mean = movmean(error_gerost_mean, window_size, 'Endpoints', 'fill');
rolling_great_mean = movmean(error_great_mean, window_size, 'Endpoints', 'fill');
rolling_gerost_std = movstd(error_gerost_mean, window_size, 0, 'Endpoints', 'fill');
rolling_great_std = movstd(error_great_mean, window_size, 0, 'Endpoints', 'fill');

% Compute convergence metrics (final 20 steps)
final_window = max(1, num_steps - 19):num_steps;
gerost_final_mean = mean(error_gerost_mean(final_window));
gerost_final_std = mean(error_gerost_std(final_window));
great_final_mean = mean(error_great_mean(final_window));
great_final_std = mean(error_great_std(final_window));

%% Print statistics
fprintf('\n========== RESULTS FROM %d MONTE CARLO TRIALS ==========\n', num_trials);
fprintf('GeRoST (with ball constraint):\n');
fprintf('  Mean error (final window): %.6f ± %.6f\n', gerost_final_mean, gerost_final_std);
fprintf('  Min/Max across trials: %.6f / %.6f\n', ...
    min(errors_gerost_trials(:, end)), max(errors_gerost_trials(:, end)));

fprintf('GREAT (baseline, no constraint):\n');
fprintf('  Mean error (final window): %.6f ± %.6f\n', great_final_mean, great_final_std);
fprintf('  Min/Max across trials: %.6f / %.6f\n', ...
    min(errors_great_trials(:, end)), max(errors_great_trials(:, end)));

improvement_pct = 100 * (great_final_mean - gerost_final_mean) / great_final_mean;
fprintf('\nGeRoST advantage: %.2f%% lower error than GREAT\n', improvement_pct);

%% Create comprehensive comparison plots
figure('Position', s.fig_pos);

% Plot 1: Error curves with error bars
subplot(2, 2, 1);
t_axis = 1:num_steps;
errorbar(t_axis(1:5:end), error_gerost_mean(1:5:end), error_gerost_std(1:5:end), ...
    'b-o', 'LineWidth', s.line_width, 'MarkerSize', s.marker_size, 'CapSize', 4, 'DisplayName', 'GeRoST');
hold on;
errorbar(t_axis(1:5:end), error_great_mean(1:5:end), error_great_std(1:5:end), ...
    'r-s', 'LineWidth', s.line_width, 'MarkerSize', s.marker_size, 'CapSize', 4, 'DisplayName', 'GREAT');
xlabel('Time step', 'FontSize', s.font_label, 'Interpreter', s.interpreter);
ylabel('Chordal distance', 'FontSize', s.font_label, 'Interpreter', s.interpreter);
title('Subspace Tracking Error with Error Bars', 'FontSize', s.font_title, 'FontWeight', 'bold', 'Interpreter', s.interpreter);
legend('FontSize', s.font_legend, 'Location', 'best', 'Interpreter', s.interpreter);
grid on;
set(gca, 'FontSize', s.font_main);
xlim([0, num_steps+1]);

% Plot 2: Convergence rate (log scale)
subplot(2, 2, 2);
semilogy(t_axis, error_gerost_mean, 'b-', 'LineWidth', s.line_width, 'DisplayName', 'GeRoST');
hold on;
semilogy(t_axis, error_great_mean, 'r-', 'LineWidth', s.line_width, 'DisplayName', 'GREAT');
% Confidence bands using fill (set HandleVisibility to off to prevent legend entry)
f1 = fill([t_axis, fliplr(t_axis)], ...
    [error_gerost_mean + error_gerost_std, fliplr(max(error_gerost_mean - error_gerost_std, 1e-8))], ...
    'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
set(f1, 'HandleVisibility', 'off');
f2 = fill([t_axis, fliplr(t_axis)], ...
    [error_great_mean + error_great_std, fliplr(max(error_great_mean - error_great_std, 1e-8))], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
set(f2, 'HandleVisibility', 'off');
xlabel('Time step', 'FontSize', s.font_label, 'Interpreter', s.interpreter);
ylabel('Chordal distance (log scale)', 'FontSize', s.font_label, 'Interpreter', s.interpreter);
title('Convergence Rate', 'FontSize', s.font_title, 'FontWeight', 'bold', 'Interpreter', s.interpreter);
legend('FontSize', s.font_legend, 'Location', 'best', 'Interpreter', s.interpreter);
grid on;
set(gca, 'FontSize', s.font_main);
xlim([0, num_steps+1]);

% Plot 3: Rolling mean (smoothed) with confidence bands
subplot(2, 2, 3);
f3 = fill([t_axis, fliplr(t_axis)], ...
    [rolling_gerost_mean + rolling_gerost_std, fliplr(rolling_gerost_mean - rolling_gerost_std)], ...
    'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
set(f3, 'HandleVisibility', 'off');
hold on;
f4 = fill([t_axis, fliplr(t_axis)], ...
    [rolling_great_mean + rolling_great_std, fliplr(rolling_great_mean - rolling_great_std)], ...
    'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
set(f4, 'HandleVisibility', 'off');
plot(t_axis, rolling_gerost_mean, 'b-', 'LineWidth', s.line_width, 'DisplayName', 'GeRoST');
plot(t_axis, rolling_great_mean, 'r-', 'LineWidth', s.line_width, 'DisplayName', 'GREAT');
xlabel('Time step', 'FontSize', s.font_label, 'Interpreter', s.interpreter);
ylabel('Chordal distance', 'FontSize', s.font_label, 'Interpreter', s.interpreter);
title(sprintf('Smoothed Error (%d-step rolling window)', window_size), ...
    'FontSize', s.font_title, 'FontWeight', 'bold', 'Interpreter', s.interpreter);
legend('FontSize', s.font_legend, 'Location', 'best', 'Interpreter', s.interpreter);
grid on;
set(gca, 'FontSize', s.font_main);
xlim([0, num_steps+1]);

% Plot 4: Box plot of final errors (last 20 steps for each trial)
subplot(2, 2, 4);
gerost_final = mean(errors_gerost_trials(:, final_window), 2);
great_final = mean(errors_great_trials(:, final_window), 2);
final_errors = [gerost_final, great_final];
bp = boxplot(final_errors, 'Labels', {'GeRoST', 'GREAT'}, 'Widths', 0.5);
set(findobj(bp, 'type', 'line'), 'LineWidth', 2);
set(findobj(bp, 'type', 'patch'), 'FaceAlpha', 0.6);
ylabel('Final Chordal Distance (mean of last 20 steps)', 'FontSize', s.font_label, 'Interpreter', s.interpreter);
title(sprintf('Final Error Distribution (%d trials)', num_trials), 'FontSize', s.font_title, 'FontWeight', 'bold', 'Interpreter', s.interpreter);
grid on;
set(gca, 'FontSize', s.font_main);
hold on;
% Overlay individual trial points with transparency
scatter(ones(num_trials, 1), gerost_final, 60, 'b', 'filled', 'MarkerEdgeAlpha', 0.5, 'MarkerFaceAlpha', 0.6);
scatter(2*ones(num_trials, 1), great_final, 60, 'r', 'filled', 'MarkerEdgeAlpha', 0.5, 'MarkerFaceAlpha', 0.6);

% Set figure paper properties for PDF export
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [11, 8.5]);  % Letter size landscape
set(gcf, 'PaperPosition', [0.5, 0.5, 10, 7.5]);  % Margins and content area

% Save figure as PDF with best fit
print(gcf, 'results/gerost_great_comparison.pdf', '-dpdf', '-r300', '-bestfit');
hold off;

fprintf('\nFigure 1 shows the comparison between GeRoST and GREAT.\n');
