% LTV_TRACKING
% Compares GeRoST and GREAT on online subspace prediction for a
% spring-mass-damper with sinusoidally time-varying stiffness.
%
% File-tree assumed:
%   <root>/
%     gerost.m, great.m
%     utils/   gradf.m, inner_max.m, lowrank_topd_eig.m, ieee_settings.m ...
%     case1/   ltv_tracking.m  (this file)
%              construct_hankel.m, generate_ltv_spring_data.m
%              subspace_initialize.m, subspace_predict.m, gerost_validate.m
%     results/
%
% External dependency: Manopt (grassmannfactory, steepestdescent)

clearvars;
close all;
rng(0);

% Apply IEEE Plot Settings
ieee_settings();

% Path setup
base = fileparts(mfilename('fullpath'));
addpath(fullfile(base, '..'));
addpath(fullfile(base, '..', 'utils'));

%% Generate data
large_error = true;
% Generates data with 50 Monte Carlo scenarios (size(y_test, 3) = 50)
[u_train, y_train, u_val, y_val, u_test, y_test] = ...
    generate_ltv_spring_data(large_error);

T_train  = size(u_train, 2);
T_val_d  = size(u_val,   2);
T_test_d = size(u_test,  2);
T_val    = T_train + T_val_d;

m = size(u_train, 1);
p = size(y_train, 1);

u_tv = [u_train, u_val];
y_tv = [y_train, y_val];

%% Parameters
ss_params.L           = 10;
ss_params.T_ini       = 5;
ss_params.T_fut       = ss_params.L - ss_params.T_ini;
ss_params.K           = 3;
ss_params.max_steps   = T_val + T_test_d + 50;

% Fixed parametes
ss_params.order_range = [3];
ss_params.T_d_range   = [100]; 
ss_params.rho_range   = [0.2];

% Redundantly set explicit single values in case other functions rely on them
ss_params.rho         = 0.2;
ss_params.order       = 3; % System order
ss_params.T_d         = 100; % Ensures T_d > m*L + n
ss_params.d           = m * ss_params.L + ss_params.order; 

%% Initialise
[~, ~, U_0, ss_params] = subspace_initialize(u_train, y_train, ss_params);

%% Validate Online Models
[gerost_val, great_val, ss_params] = ...
    gerost_validate(U_0, u_tv, y_tv, T_train, ss_params);

% =========================================================================
% ADAPTIVE RHO (Based on Remark 4.7)
% =========================================================================
gerost_val.rho_param = @(t, sigs, What, Y) compute_dynamic_rho(sigs, gerost_val.k, gerost_val.d);

%% Online test
nr_scenarios = size(y_test, 3);
T_steps      = T_test_d - ss_params.T_fut;

err_gerost = zeros(T_steps, nr_scenarios);
err_great  = zeros(T_steps, nr_scenarios);

for scenario = 1:nr_scenarios
    fprintf('Testing scenario %i/%i\n', scenario, nr_scenarios);

    u_full = [u_tv, u_test(:, :, scenario)];
    y_full = [y_tv, y_test(:, :, scenario)];

    gerost_obj = gerost_val;
    great_obj  = great_val;

    for i = T_val + 1 : T_val + T_steps
        % Online Subspace Updates
        gerost_obj = gerost_obj.descent_step(make_sample(u_full, y_full, i, ss_params.L, m, p));
        great_obj  =  great_obj.descent_step(make_sample(u_full, y_full, i, ss_params.L, m, p));

        u_ini  = u_full(:, i-ss_params.T_ini+1:i);
        y_ini  = y_full(:, i-ss_params.T_ini+1:i);
        u_fut  = u_full(:, i+1:i+ss_params.T_fut);
        y_true = y_full(:, i+1:i+ss_params.T_fut);

        % Subspace Predictions (Enforce real() to drop numerical complex artifacts)
        gerost_pred_y = real(subspace_predict(gerost_obj.U, u_ini, y_ini, u_fut, ss_params));
        great_pred_y  = real(subspace_predict(great_obj.U,  u_ini, y_ini, u_fut, ss_params));

        t_idx = i - T_val;
        
        % Explicit relative prediction error calculation per the requested formula:
        err_den = sum(sum(abs(y_true).^2));
        
        gerost_err_num = sum(sum(abs(gerost_pred_y(:, ss_params.T_ini+1:end) - y_true).^2));
        err_gerost(t_idx, scenario) = real(sqrt(gerost_err_num / err_den));
        
        great_err_num  = sum(sum(abs(great_pred_y(:, ss_params.T_ini+1:end) - y_true).^2));
        err_great(t_idx, scenario) = real(sqrt(great_err_num / err_den));
    end
end

%% Save results
err_gerost_a = real(mean(err_gerost, 2));   std_err_gerost = real(std(err_gerost, 0, 2));
err_great_a  = real(mean(err_great,  2));   std_err_great  = real(std(err_great,  0, 2));

if large_error
    file_name = fullfile(base, '..', 'results', 'ltv_spring_large_error.csv');
else
    file_name = fullfile(base, '..', 'results', 'ltv_spring_clean.csv');
end

writematrix([(1:T_steps)', ...
    err_gerost_a, err_gerost_a+std_err_gerost, err_gerost_a-std_err_gerost, ...
    err_great_a,  err_great_a +std_err_great,  err_great_a -std_err_great], ...
    file_name);
fprintf('Results saved to %s\n', file_name);

%% Plot
t_axis = (1:T_steps)';

figure('Name', 'Relative prediction error', 'NumberTitle', 'off', 'Position', [100, 100, 700, 450]);
hold on; grid on;

% Define explicit colors for better contrast
col_gerost = [0 0.4470 0.7410];       % Blue
col_great  = [0.8500 0.3250 0.0980];  % Red

% Create standard deviation color bands (shaded area)
fill([t_axis; flipud(t_axis)], ...
     [err_great_a + std_err_great; flipud(max(0, err_great_a - std_err_great))], ...
     col_great, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

fill([t_axis; flipud(t_axis)], ...
     [err_gerost_a + std_err_gerost; flipud(max(0, err_gerost_a - std_err_gerost))], ...
     col_gerost, 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% Plot the average errors on top of the shaded regions
plot(t_axis, err_great_a,  '-', 'Color', col_great,  'LineWidth', 2, 'DisplayName', 'GREAT');
plot(t_axis, err_gerost_a, '-', 'Color', col_gerost, 'LineWidth', 2, 'DisplayName', 'GeRoST');

if large_error
    % Plot a dashed black line to signify the sensor fault
    xline(80, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Sensor fault');
end

xlabel('$t$');
ylabel('Relative prediction error');
xlim([0, T_steps]);
ylim([0, 15]);
legend('Location', 'best');

% Force the axis box and ticks to be on top of the shaded patches
set(gca, 'Layer', 'top');

%% Quantify Effectiveness (Post-Fault)
if large_error
    fault_idx = 80;
    
    % Calculate post-fault mean and std across all time steps and scenarios
    post_fault_gerost = real(err_gerost(fault_idx+1:end, :));
    post_fault_great  = real(err_great(fault_idx+1:end, :));
    
    mean_pf_gerost = real(mean(post_fault_gerost, 'all'));
    std_pf_gerost  = real(std(post_fault_gerost, 0, 'all'));
    
    mean_pf_great  = real(mean(post_fault_great, 'all'));
    std_pf_great   = real(std(post_fault_great, 0, 'all'));

    improvement_great = (mean_pf_great - mean_pf_gerost) / mean_pf_great * 100;
    
    % Print quantification
    fprintf('\n--- Post-Fault Performance (t > %d) ---\n', fault_idx);
    fprintf('GREAT  : Mean Error = %.3f, Std Dev = %.3f\n', mean_pf_great, std_pf_great);
    fprintf('GeRoST : Mean Error = %.3f, Std Dev = %.3f\n', mean_pf_gerost, std_pf_gerost);
    fprintf('GeRoST improves avg post-fault error by %.1f%% over GREAT\n', improvement_great);
    fprintf('----------------------------------------\n');
    
    % Plot Bar Chart with Error Bars
    figure('Name', 'Post-Fault Performance', 'NumberTitle', 'off', 'Position', [820, 100, 450, 400]);
    hold on; grid on;
    
    % Force purely real arrays into bar/errorbar functions
    means = real([mean_pf_gerost, mean_pf_great]);
    stds  = real([std_pf_gerost, std_pf_great]);
    
    % Create bar chart
    b = bar(1:2, means, 0.5);
    b.FaceColor = 'flat';
    b.CData(1,:) = col_gerost;
    b.CData(2,:) = col_great;
    
    % Add standard deviation error bars
    errorbar(1:2, means, stds, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 10);
    
    set(gca, 'XTick', 1:2, 'XTickLabel', {'GeRoST (Proposed)', 'GREAT'});
    ylabel('Average Relative Error');
    title(sprintf('Post-Fault Performance ($t > %d$)', fault_idx));
    xlim([0.5, 2.5]);
end

%% Helpers

function s = make_sample(u_, y_, i_, L_, m_, p_)
    s = [reshape(u_(:, i_-L_+1:i_), [L_*m_, 1]);
         reshape(y_(:, i_-L_+1:i_), [L_*p_, 1])];
end

function rho = compute_dynamic_rho(sigma_vals, k, d)
    % Computes the minimum uncertainty ball radius 
    if length(sigma_vals) > k
        sigma_k  = max(sigma_vals(k), 1e-8);
        sigma_k1 = max(sigma_vals(k+1), 0);
        p_t = sigma_k1 / sigma_k;
    else
        p_t = 0; % Edge case: No noise components captured
    end
    p_t = min(p_t, 0.95);
    gamma = 0.5; 
    rho = (sqrt(2) * p_t) / (1 - p_t) + gamma * sqrt(max(d - k, 0));
end