%% GeRoST - MATLAB Implementation Example
% This script demonstrates how to use the GeRoST class for online subspace tracking

addpath('./utils');

%% Example 1: Basic Usage with Fixed Subspace
rng(42);

% Problem dimensions
n = 50;        % ambient dimension
k = 5;         % true subspace dimension
d = 8;         % data subspace dimension
T = 10;        % window length
K = 5;         % inner gradient descent iterations

% Initialize GeRoST object
tracker = gerost(n, k, d, T, 'K', K, 'rho', 0.1, 'fallback', 'data');

% Generate true subspace
U_true = randn(n, k);
U_true = orth(U_true);  % ensure it's an orthonormal basis

% Phase 1: Data collection (t = 0 to t_0)
t_0 = 20;  % data collection duration
data_collection = zeros(n, t_0);

for t = 1:t_0
    coeff = randn(k, 1);
    noise = 0.01 * randn(n, 1);
    u = U_true * coeff + noise;
    data_collection(:, t) = u;
end

% Phase 2: Initialize tracker with collected data and start tracking at t = t_0+1
% Compute initial subspace estimate from collected data
[U_init, ~, ~] = svd(data_collection, 'econ');
U_init = U_init(:, 1:k);

% Initialize tracker
tracker = initialize(tracker, U_init);

% Phase 3: Subspace tracking (t = t_0+1 to t = t_0+num_steps)
num_steps = 100;
U_estimates = zeros(n, k, num_steps);

for t = 1:num_steps
    % Generate data vector: mostly from true subspace + small noise
    coeff = randn(k, 1);
    noise = 0.01 * randn(n, 1);
    u = U_true * coeff + noise;
    
    % Process step
    tracker = tracker.descent_step(u);
    U_estimates(:, :, t) = tracker.U;
end

% Compute subspace error (chordal distance)
subspace_error = zeros(1, num_steps);
for t = 1:num_steps
    subspace_error(t) = chordalDist(U_estimates(:, :, t), U_true);
end

% Plot results
figure(1);
time_axis = (t_0 + 1):(t_0 + num_steps);  % Time axis starting from t_0+1
plot(time_axis, subspace_error, 'linewidth', 1.5);
xlabel('Time Step');
ylabel('Chordal Distance to True Subspace');
title(sprintf('GeRoST Subspace Tracking Error (Data collection: t=1 to t=%d, Tracking: t=%d to t=%d)', ...
    t_0, t_0+1, t_0+num_steps));
grid on;
