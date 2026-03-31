function [u_train, y_train, u_val, y_val, u_test, y_test] = ...
         generate_ltv_spring_data(large_error)
% GENERATE_LTV_SPRING_DATA   Simulate a spring-mass-damper with sinusoidally
% time-varying stiffness — a standard LTV benchmark in control literature.
%
%   Continuous-time model:
%       mass * x'' + c * x' + k(t) * x  =  u
%       k(t)  =  k_base  +  k_amp * sin(2*pi*t / T_period)
%   State  :  [x; x']   (position and velocity)
%   Output :  y  =  x   (position measurement)
%   Input  :  u  =  applied force
%
%   The system is discretised step-by-step via zero-order hold so that
%   the time-varying A(t) matrix is correctly captured.
%
%   Returns normalised train / validation / test splits whose structure
%   mirrors generate_airplane_data.m.  When large_error = true a 20x
%   noise-magnitude spike is injected at t = error_time within every
%   test scenario (simulating a transient sensor fault).
%
%   Outputs
%   -------
%   u_train  : (m x T_train)
%   y_train  : (p x T_train)
%   u_val    : (m x T_val)
%   y_val    : (p x T_val)
%   u_test   : (m x T_test x nr_scenarios)
%   y_test   : (p x T_test x nr_scenarios)

if nargin < 1
    large_error = false;
end

% --- Simulation settings ------------------------------------------------
d_T          = 0.1;                   % discretisation step (s)
T_final      = 900;                   % total simulation steps
T_train      = floor(T_final / 3);   % 300
T_val_end    = floor(2*T_final / 3); % 600  (val data: steps 301-600)
nr_scenarios = 50;
error_time   = 80;                    % spike location within test trajectory

snr       = 5; % sensor fault magnitude
signal_sd = 1;
noise_sd  = signal_sd / snr;

% --- LTV spring-mass-damper parameters ----------------------------------
%   k(t) oscillates between k_base - k_amp = 2  and  k_base + k_amp = 8
%   giving a roughly 2x change in natural frequency over one period.
mass     = 1.0;    % kg
c_damp   = 0.3;    % N s/m — lightly damped
k_base   = 5.0;    % N/m  — nominal stiffness
k_amp    = 3.0;    % N/m  — variation amplitude
T_period = 400;    % steps per stiffness cycle (~slower than system dynamics)

n_state = 2;   m_in = 1;   p_out = 1;

k_fn  = @(step) k_base + k_amp * sin(2*pi*step / T_period);
Ac_fn = @(step) [0, 1; -k_fn(step)/mass, -c_damp/mass];
Bc    = [0; 1/mass];   % input matrix (constant)
Cc    = [1, 0];        % output: position only
Dc    = 0;

% --- Training + validation simulation -----------------------------------
rng(0);
u_tv = randn(m_in, T_val_end) * signal_sd;

x = zeros(n_state, T_final + 1);   % zero initial condition
for t = 1:T_val_end
    sys_d    = c2d(ss(Ac_fn(t), Bc, Cc, Dc), d_T);
    x(:,t+1) = sys_d.A * x(:,t) + sys_d.B * u_tv(:,t);
end

y_tv = Cc * x(:,1:T_val_end) + randn(p_out, T_val_end) * noise_sd;

u_train_raw = u_tv(:, 1:T_train);
y_train_raw = y_tv(:, 1:T_train);
u_val_raw   = u_tv(:, T_train+1:T_val_end);
y_val_raw   = y_tv(:, T_train+1:T_val_end);

% Normalise using training statistics only
mean_u = mean(u_train_raw, 2);   std_u = std(u_train_raw, 0, 2);
mean_y = mean(y_train_raw, 2);   std_y = std(y_train_raw, 0, 2);

u_train = (u_train_raw - mean_u) ./ std_u;
y_train = (y_train_raw - mean_y) ./ std_y;
u_val   = (u_val_raw   - mean_u) ./ std_u;
y_val   = (y_val_raw   - mean_y) ./ std_y;

% --- Test scenarios -----------------------------------------------------
T_test = T_final - T_val_end;   % 300 steps per scenario
y_test = zeros(p_out, T_test, nr_scenarios);
u_test = zeros(m_in,  T_test, nr_scenarios);

for scenario = 1:nr_scenarios
    u_sc       = randn(m_in, T_test) * signal_sd;
    x_sc       = zeros(n_state, T_test+1);
    x_sc(:,1)  = x(:, T_val_end+1);   % continue from end of train+val state

    for t = 1:T_test
        t_global   = T_val_end + t;
        sys_d      = c2d(ss(Ac_fn(t_global), Bc, Cc, Dc), d_T);
        x_sc(:,t+1) = sys_d.A * x_sc(:,t) + sys_d.B * u_sc(:,t);
    end

    y_sc = Cc * x_sc(:,1:T_test) + randn(p_out, T_test) * noise_sd;

    if large_error
        % inject a large spike at a fixed time to simulate a sensor fault
        y_sc(:, error_time) = y_sc(:, error_time) + randn(p_out,1) * noise_sd * 20;
    end

    y_test(:,:,scenario) = (y_sc   - mean_y) ./ std_y;
    u_test(:,:,scenario) = (u_sc   - mean_u) ./ std_u;
end
end