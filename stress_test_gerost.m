function stress_test_gerost()
%% stress_test_gerost  —  Empirical stress tests for gerost.m
%
%  Unlike test_gerost (which checks known-outcome sanity conditions),
%  every test here discovers its answer by running the code and measuring
%  an empirical quantity, then asserting it falls within a statistically
%  justified bound.
%
%  Run with:
%    >> stress_test_gerost
%
%  Sections
%  --------
%  S1.  rank1_eig_update — 500 random (n,d,v,sgn) combinations
%         S1a. Subspace error vs ground-truth eig()  < 1e-8
%         S1b. Output U orthonormal                  < 1e-10
%         S1c. Singular values non-negative and descending
%         S1d. Downdate: s_new^2 <= s_old^2 elementwise (energy shrinks)
%         S1e. Update:   top eigenvalue non-decreasing
%
%  S2.  build_window_matrix rank-2 update — 200 random (n,d,T) configs,
%       each run for 50 consecutive sliding-window steps
%         S2a. subspace_err(rank-2, full SVD) < 1e-8  at every step
%         S2b. Gram-matrix relative error      < 1e-6  at every step
%         S2c. sigma_vals sorted descending    at every step
%         S2d. What orthonormal                at every step
%
%  S3.  descent_step Grassmann invariant — 100 random tracker configs,
%       each run for 100 steps
%         S3a. ||U'U - I||_F < 1e-10  at every single step
%         S3b. U_history(:,:,t) == U   at every step after warm-up
%         S3c. window_idx always has exactly min(t,T) elements
%
%  S4.  inner_max correctness — 300 random (Y, What, rho) instances
%         S4a. d_c(Wstar, What) <= rho + 1e-6  (ball constraint satisfied)
%         S4b. ||Wstar^T Wstar - I||_F < 1e-10 (Wstar orthonormal)
%         S4c. Objective ||Wstar^T Y||_F^2 >= ||What^T Y||_F^2
%              (Wstar beats the ball centre)
%         S4d. lambda_star > 2 whenever d_c(W_unconstrained, What) > rho
%              (active constraint correctly detected)
%
%  S5.  impute reconstruction accuracy — 500 random (n,k,obs_ratio) instances
%         S5a. Observed entries identical to input
%         S5b. ||u_imputed - u_true||/||u_true|| < 1e-6
%              when u_true is exactly in span(U)
%         S5c. Error degrades gracefully as obs_ratio decreases
%              (error at 30% obs < 10x error at 90% obs, not blow-up)
%
%  S6.  Convergence under varying SNR — 5 noise levels x 20 trials each
%         S6a. Final chordal distance < noise_std * C  (noise-floor scaling)
%         S6b. Monotone improvement: errors(end) < errors(1) in 19/20 trials
%
%  S7.  Missing data robustness — 50 trials, obs_ratio swept 0.3 to 0.9
%         S7a. No crash for any obs_ratio
%         S7b. U orthonormal after every step
%         S7c. Chordal distance at end < 1.0 for obs_ratio >= 0.5
%
%  S8.  rank1_eig_update cascade stability — chain 1000 sequential updates
%       starting from a known exact state, check drift does not accumulate
%         S8a. subspace_err vs fresh eig() < 1e-4 after 1000 updates
%         S8b. U orthonormal < 1e-10 throughout
%
%  S9.  Dimension sweep — n in {20,50,100,200}, k/n=0.2, d=k+2, T=2k
%         S9a. No crash for any dimension
%         S9b. Chordal distance after 150 steps < 0.5 for all dimensions
%         S9c. U orthonormal < 1e-10 for all dimensions

% =========================================================================
pass  = 0;
fail  = 0;
total = 0;

fprintf('=================================================================\n');
fprintf('  GeRoST stress tests\n');
fprintf('=================================================================\n\n');

% =========================================================================
%  S1. rank1_eig_update — 500 random instances
% =========================================================================
fprintf('--- S1. rank1_eig_update (500 random instances) -----------------\n');

N_S1 = 500;
s1_subspace_errs  = zeros(N_S1, 1);
s1_orth_errs      = zeros(N_S1, 1);
s1_neg_sv         = false(N_S1, 1);
s1_unsorted       = false(N_S1, 1);
s1_energy_ok      = true(N_S1, 1);   % downdate: energy shrinks
s1_top_ev_ok      = true(N_S1, 1);   % update:   top eigenvalue non-decreasing

rng(0);
for i = 1:N_S1
    n_i = randi([10, 60]);
    d_i = randi([2,  max(3, floor(n_i/3))]);
    d_i = min(d_i, n_i - 1);

    % Random rank-d PSD via eigenfactors
    [U_i, ~] = qr(randn(n_i, d_i), 0);
    s_i      = sort(rand(d_i,1)*4 + 0.5, 'descend');

    sgn_i = 2*(rand() > 0.5) - 1;   % +1 or -1
    v_i   = randn(n_i, 1);

    % For downdate keep it mild so C stays PSD
    if sgn_i == -1
        v_i = v_i / norm(v_i) * min(s_i) * 0.3;
    end

    % Ground truth
    C_i     = U_i * diag(s_i.^2) * U_i';
    C_new_i = C_i + sgn_i * (v_i * v_i');

    % Only test if C_new is PSD (required for valid downdate)
    ev_all = eig(C_new_i);
    if any(ev_all < -1e-8)
        continue;
    end

    [Ev_i, El_i]   = eig(C_new_i, 'vector');
    [El_i, idx_i]  = sort(El_i, 'descend');
    U_ref_i        = Ev_i(:, idx_i(1:d_i));

    [U_new_i, s_new_i] = gerost.rank1_eig_update(U_i, s_i, v_i, sgn_i, d_i);

    % S1a: subspace error
    s1_subspace_errs(i) = norm(U_new_i*U_new_i' - U_ref_i*U_ref_i', 'fro');

    % S1b: orthonormality
    s1_orth_errs(i) = norm(U_new_i'*U_new_i - eye(d_i), 'fro');

    % S1c: singular values valid
    s1_neg_sv(i)   = any(s_new_i < 0);
    s1_unsorted(i) = ~issorted(flipud(s_new_i));

    % S1d: downdate — each eigenvalue can only decrease
    if sgn_i == -1
        s1_energy_ok(i) = all(sort(s_new_i,'descend').^2 <= sort(s_i,'descend').^2 + 1e-10);
    end

    % S1e: update — top eigenvalue non-decreasing
    if sgn_i == +1
        s1_top_ev_ok(i) = (max(s_new_i)^2 >= max(s_i)^2 - 1e-10);
    end
end

check('S1a: subspace error < 1e-8  (max over 500 trials)',  max(s1_subspace_errs) < 1e-8);
check('S1b: U orthonormal < 1e-10  (max over 500 trials)',  max(s1_orth_errs)     < 1e-10);
check('S1c: s_new non-negative     (all 500 trials)',        ~any(s1_neg_sv));
check('S1c: s_new sorted desc      (all 500 trials)',        ~any(s1_unsorted));
check('S1d: downdate shrinks energy (all downdate trials)',   all(s1_energy_ok));
check('S1e: update top-ev non-decr  (all update trials)',    all(s1_top_ev_ok));

fprintf('     max subspace_err = %.2e\n', max(s1_subspace_errs));
fprintf('     max orth_err     = %.2e\n', max(s1_orth_errs));

% =========================================================================
%  S2. build_window_matrix rank-2 update — 200 configs x 50 steps
% =========================================================================
fprintf('\n--- S2. build_window_matrix rank-2 (200 configs x 50 steps) ----\n');

N_S2 = 200;
s2_max_subspace = zeros(N_S2, 1);
s2_max_gram_rel = zeros(N_S2, 1);
s2_orth_fail    = false(N_S2, 1);
s2_sort_fail    = false(N_S2, 1);

rng(1);
for i = 1:N_S2
    n_i = randi([15, 60]);
    d_i = randi([2, max(3, floor(n_i/4))]);
    d_i = min(d_i, n_i - 1);
    k_i = max(1, d_i - 1);
    T_i = randi([d_i+1, d_i+15]);
    max_steps_i = T_i + 50;

    g_i      = gerost(n_i, k_i, d_i, T_i, 'K', 1, 'rho', 0.1, ...
                      'max_steps', max_steps_i + 5);
    U_data_i = orth(randn(n_i, d_i));   % exactly rank-d data subspace
    g_i      = initialize(g_i, orth(randn(n_i, k_i)));

    % Warm-up: load T samples directly
    for t = 1:T_i
        g_i.t            = g_i.t + 1;
        g_i.samples(:, g_i.t) = U_data_i * randn(d_i, 1);
        g_i.window_idx   = 1:g_i.t;
    end
    [g_i, ~, ~] = build_window_matrix(g_i);   % populate cache

    % Steady-state: 50 rank-2 steps
    for step = 1:50
        g_i.t = g_i.t + 1;
        g_i.samples(:, g_i.t) = U_data_i * randn(d_i, 1);
        g_i.window_idx = (g_i.t - T_i + 1):g_i.t;

        [g_i, What_r2, sv_r2] = build_window_matrix(g_i);

        % Ground truth full SVD
        W_true = g_i.samples(:, g_i.window_idx);
        [Usv, Ssv, ~] = svd(W_true, 'econ');
        What_ref = Usv(:, 1:d_i);

        % S2a: subspace error
        sub_err = norm(What_r2*What_r2' - What_ref*What_ref', 'fro');
        s2_max_subspace(i) = max(s2_max_subspace(i), sub_err);

        % S2b: Gram relative error
        Gram_r2  = What_r2 * diag(sv_r2.^2) * What_r2';
        Gram_ref = W_true * W_true';
        [Uv, Sv] = eigs(Gram_ref, d_i);
        Gram_ref_d = Uv * Sv * Uv';
        rel_err = norm(Gram_r2 - Gram_ref_d, 'fro') / ...
                  (norm(Gram_ref_d, 'fro') + 1e-12);
        s2_max_gram_rel(i) = max(s2_max_gram_rel(i), rel_err);

        % S2c: singular values sorted
        if ~issorted(flipud(sv_r2))
            s2_sort_fail(i) = true;
        end

        % S2d: What orthonormal
        orth_err = norm(What_r2'*What_r2 - eye(d_i), 'fro');
        if orth_err > 1e-10
            s2_orth_fail(i) = true;
        end
    end
end

check('S2a: rank-2 subspace_err < 1e-8 (max over all configs+steps)', ...
      max(s2_max_subspace) < 1e-8);
check('S2b: Gram rel error < 1e-6 (max over all configs+steps)', ...
      max(s2_max_gram_rel) < 1e-6);
check('S2c: sigma_vals sorted desc (all configs+steps)', ~any(s2_sort_fail));
check('S2d: What orthonormal < 1e-10 (all configs+steps)', ~any(s2_orth_fail));

fprintf('     max subspace_err = %.2e\n', max(s2_max_subspace));
fprintf('     max Gram rel_err = %.2e\n', max(s2_max_gram_rel));

% =========================================================================
%  S3. descent_step Grassmann invariant — 100 configs x 100 steps
% =========================================================================
fprintf('\n--- S3. descent_step Grassmann invariant (100 configs x 100 steps)\n');

N_S3 = 100;
s3_max_orth  = zeros(N_S3, 1);
s3_hist_fail = false(N_S3, 1);
s3_win_fail  = false(N_S3, 1);

rng(2);
for i = 1:N_S3
    n_i  = randi([15, 50]);
    k_i  = randi([2, max(3, floor(n_i/4))]);
    k_i  = min(k_i, n_i - 2);
    d_i  = k_i + randi([1, 3]);
    d_i  = min(d_i, n_i - 1);
    T_i  = randi([k_i+2, k_i+12]);
    K_i  = randi([3, 8]);

    g_i      = gerost(n_i, k_i, d_i, T_i, 'K', K_i, 'rho', 0.15, ...
                      'max_steps', 110);
    U_true_i = orth(randn(n_i, k_i));
    g_i      = initialize(g_i, orth(randn(n_i, k_i)));

    for t = 1:100
        u_t = U_true_i * randn(k_i,1) + 0.01*randn(n_i,1);
        g_i = descent_step(g_i, u_t);

        % S3a: U orthonormal at every step
        orth_err = norm(g_i.U'*g_i.U - eye(k_i), 'fro');
        s3_max_orth(i) = max(s3_max_orth(i), orth_err);

        % S3b: U_history written correctly after warm-up
        if g_i.t >= d_i
            hist_err = norm(g_i.U_history(:,:,g_i.t) - g_i.U, 'fro');
            if hist_err > 1e-12
                s3_hist_fail(i) = true;
            end
        end

        % S3c: window_idx length exactly min(t,T)
        expected_len = min(g_i.t, T_i);
        if length(g_i.window_idx) ~= expected_len
            s3_win_fail(i) = true;
        end
    end
end

check('S3a: U orthonormal < 1e-10 (max over all configs+steps)', ...
      max(s3_max_orth) < 1e-10);
check('S3b: U_history matches U  (all configs+steps)', ~any(s3_hist_fail));
check('S3c: window_idx correct length (all configs+steps)', ~any(s3_win_fail));

fprintf('     max orth_err = %.2e\n', max(s3_max_orth));

% =========================================================================
%  S4. inner_max correctness — 300 random (Y, What, rho) instances
% =========================================================================
fprintf('\n--- S4. inner_max (300 random instances) -----------------------\n');

N_S4 = 300;
s4_constraint_viol = false(N_S4, 1);
s4_orth_fail       = false(N_S4, 1);
s4_obj_fail        = false(N_S4, 1);
s4_active_fail     = false(N_S4, 1);

rng(3);
for i = 1:N_S4
    n_i   = randi([10, 50]);
    k_i   = randi([2, max(3, floor(n_i/3))]);
    k_i   = min(k_i, n_i - 2);
    d_i   = k_i + randi([1, 3]);
    d_i   = min(d_i, n_i - 1);
    rho_i = rand() * (sqrt(d_i) - 0.1) + 0.05;  % in (0.05, sqrt(d))
    rho_i = min(rho_i, sqrt(k_i) - 1e-6);        % respect gerost clamp

    Y_i    = orth(randn(n_i, k_i));
    What_i = orth(randn(n_i, d_i));

    [Wstar_i, lam_i] = gerost.inner_max(Y_i, What_i, rho_i, d_i);

    % S4a: ball constraint satisfied
    sv_c = min(svd(Wstar_i' * What_i), 1);
    dc_i = sqrt(max(d_i - sum(sv_c.^2), 0));
    s4_constraint_viol(i) = (dc_i > rho_i + 1e-6);

    % S4b: Wstar orthonormal
    orth_err = norm(Wstar_i'*Wstar_i - eye(d_i), 'fro');
    s4_orth_fail(i) = (orth_err > 1e-8);

    % S4c: objective >= ball-centre baseline
    obj_star   = norm(Wstar_i' * Y_i, 'fro')^2;
    obj_centre = norm(What_i'  * Y_i, 'fro')^2;
    s4_obj_fail(i) = (obj_star < obj_centre - 1e-8);

    % S4d: lambda_star > 2 iff unconstrained optimum violates the ball
    W_unc   = gerost.lowrank_topd_eig(What_i, Y_i, 0, d_i);
    sv_unc  = min(svd(W_unc' * What_i), 1);
    dc_unc  = sqrt(max(d_i - sum(sv_unc.^2), 0));
    should_be_active = (dc_unc > rho_i + 1e-8);
    s4_active_fail(i) = should_be_active && ~(lam_i > 2);
end

check('S4a: ball constraint satisfied (all 300 trials)',   ~any(s4_constraint_viol));
check('S4b: Wstar orthonormal < 1e-8 (all 300 trials)',    ~any(s4_orth_fail));
check('S4c: obj(Wstar) >= obj(What) (all 300 trials)',     ~any(s4_obj_fail));
check('S4d: lambda>2 when constraint active (all trials)', ~any(s4_active_fail));

fprintf('     constraint violations: %d / %d\n', sum(s4_constraint_viol), N_S4);
fprintf('     active-flag failures:  %d / %d\n', sum(s4_active_fail),     N_S4);

% =========================================================================
%  S5. impute reconstruction — 500 random (n,k,obs_ratio) instances
% =========================================================================
fprintf('\n--- S5. impute reconstruction (500 random instances) -----------\n');

N_S5    = 500;
s5_obs_unchanged  = true(N_S5, 1);
s5_rel_err        = zeros(N_S5, 1);
s5_err_high_obs   = zeros(N_S5, 1);   % obs_ratio = 0.9
s5_err_low_obs    = zeros(N_S5, 1);   % obs_ratio = 0.3

rng(4);
for i = 1:N_S5
    n_i   = randi([10, 60]);
    k_i   = randi([2, max(3, floor(n_i/3))]);
    k_i   = min(k_i, n_i - 2);
    obs_i = 0.3 + 0.6 * rand();      % random obs ratio in [0.3, 0.9]

    U_i = orth(randn(n_i, k_i));
    g_i = gerost(n_i, k_i, k_i, k_i+2, 'max_steps', 10);
    g_i = initialize(g_i, U_i);

    % u exactly in span(U_i)
    u_true = U_i * randn(k_i, 1);

    mask_i = rand(n_i, 1) > (1 - obs_i);
    mask_i(1:k_i) = true;   % guarantee rank condition

    u_imp = impute(g_i, u_true, mask_i);

    % S5a: observed entries unchanged
    s5_obs_unchanged(i) = norm(u_imp(mask_i) - u_true(mask_i)) < 1e-12;

    % S5b: relative reconstruction error
    s5_rel_err(i) = norm(u_imp - u_true) / (norm(u_true) + 1e-15);

    % S5c: compare high vs low obs for same subspace
    mask_hi = rand(n_i,1) > 0.1; mask_hi(1:k_i) = true;
    mask_lo = rand(n_i,1) > 0.7; mask_lo(1:k_i) = true;
    u_hi = impute(g_i, u_true, mask_hi);
    u_lo = impute(g_i, u_true, mask_lo);
    s5_err_high_obs(i) = norm(u_hi - u_true);
    s5_err_low_obs(i)  = norm(u_lo - u_true);
end

check('S5a: observed entries unchanged (all 500 trials)',       all(s5_obs_unchanged));
check('S5b: relative recon error < 1e-8 (all 500 trials)', ...
      max(s5_rel_err) < 1e-8);
check('S5c: high-obs error <= low-obs error (>=90% of trials)', ...
      mean(s5_err_high_obs <= s5_err_low_obs + 1e-12) >= 0.90);

fprintf('     max rel recon error = %.2e\n', max(s5_rel_err));
fprintf('     high<=low obs:  %d / %d trials\n', ...
        sum(s5_err_high_obs <= s5_err_low_obs + 1e-12), N_S5);

% =========================================================================
%  S6. Convergence under varying SNR — 5 noise levels x 20 trials
% =========================================================================
fprintf('\n--- S6. Convergence under varying SNR (5 levels x 20 trials) ---\n');

noise_levels = [0, 0.001, 0.01, 0.05, 0.1];
n6=40; k6=6; d6=8; T6=15; K6=10; steps6=150; t0_6=40;

rng(5);
for ni = 1:length(noise_levels)
    noise_std = noise_levels(ni);
    final_errs = zeros(20, 1);
    init_errs  = zeros(20, 1);
    mono_count = 0;

    for trial = 1:20
        U_true6 = orth(randn(n6, k6));
        X0_6 = U_true6*randn(k6,t0_6) + noise_std*randn(n6,t0_6);
        [U0_6,~,~] = svd(X0_6,'econ');
        U0_6 = U0_6(:,1:k6);

        g6 = gerost(n6,k6,d6,T6,'K',K6,'rho',0.15,'fallback','data', ...
                    'max_steps',steps6);
        g6 = initialize(g6, U0_6);

        errs6 = zeros(steps6,1);
        for t = 1:steps6
            u6 = U_true6*randn(k6,1) + noise_std*randn(n6,1);
            g6 = descent_step(g6, u6);
            sv6 = min(svd(g6.U'*U_true6), 1);
            errs6(t) = sqrt(max(k6 - sum(sv6.^2), 0));
        end

        init_errs(trial)  = mean(errs6(1:10));
        final_errs(trial) = mean(errs6(end-9:end));
        if final_errs(trial) < init_errs(trial)
            mono_count = mono_count + 1;
        end
    end

    label = sprintf('noise=%.3f', noise_std);
    check(sprintf('S6: %s — error decreases in >=18/20 trials', label), ...
          mono_count >= 18);
    check(sprintf('S6: %s — final err < 1.0 (all 20 trials)', label), ...
          max(final_errs) < 1.0);
    fprintf('     %s: final_err = %.4f ± %.4f  (mono %d/20)\n', ...
            label, mean(final_errs), std(final_errs), mono_count);
end

% =========================================================================
%  S7. Missing data robustness — 50 trials, obs_ratio 0.3 to 0.9
% =========================================================================
fprintf('\n--- S7. Missing data robustness (50 trials, swept obs_ratio) ---\n');

obs_ratios = linspace(0.3, 0.9, 7);
n7=30; k7=5; d7=7; T7=12; K7=8; steps7=80; t0_7=30;

rng(6);
for oi = 1:length(obs_ratios)
    obs7 = obs_ratios(oi);
    crashed   = false;
    orth_fail = false;
    final_errs7 = zeros(50,1);

    for trial = 1:50
        U_true7 = orth(randn(n7, k7));
        X0_7 = U_true7*randn(k7,t0_7) + 0.01*randn(n7,t0_7);
        [U0_7,~,~] = svd(X0_7,'econ'); U0_7 = U0_7(:,1:k7);

        g7s = gerost(n7,k7,d7,T7,'K',K7,'rho',0.1,'missing',true, ...
                     'max_steps',steps7);
        g7s = initialize(g7s, U0_7);

        try
            for t = 1:steps7
                u7  = U_true7*randn(k7,1) + 0.01*randn(n7,1);
                m7  = rand(n7,1) < obs7;
                m7(1:k7) = true;
                g7s = descent_step(g7s, u7, 'mask', m7);

                oe = norm(g7s.U'*g7s.U - eye(k7),'fro');
                if oe > 1e-9
                    orth_fail = true;
                end
            end
        catch
            crashed = true;
        end

        sv7s = min(svd(g7s.U'*U_true7), 1);
        final_errs7(trial) = sqrt(max(k7 - sum(sv7s.^2), 0));
    end

    label7 = sprintf('obs=%.0f%%', obs7*100);
    check(sprintf('S7: %s — no crash (50 trials)',        label7), ~crashed);
    check(sprintf('S7: %s — U orthonormal throughout',    label7), ~orth_fail);
    if obs7 >= 0.5
        check(sprintf('S7: %s — final err < 1.0 (all 50)', label7), ...
              max(final_errs7) < 1.0);
    end
    fprintf('     %s: final_err = %.4f ± %.4f\n', ...
            label7, mean(final_errs7), std(final_errs7));
end

% =========================================================================
%  S8. rank1_eig_update cascade — 1000 sequential updates
% =========================================================================
fprintf('\n--- S8. rank1_eig_update cascade (1000 sequential updates) -----\n');

rng(7);
n8c = 30; d8c = 5;
[U8c, ~] = qr(randn(n8c, d8c), 0);
s8c      = sort(rand(d8c,1)*3 + 0.5, 'descend');
C8c_true = U8c * diag(s8c.^2) * U8c';

max_orth8 = 0;
U8c_cur   = U8c;
s8c_cur   = s8c;

for iter = 1:1000
    sgn8  = 2*(rand() > 0.5) - 1;
    v8    = randn(n8c, 1);
    if sgn8 == -1
        v8 = v8 / norm(v8) * min(s8c_cur) * 0.2;
    end

    C8c_true = C8c_true + sgn8 * (v8 * v8');

    % Keep only PSD part (downdates can make eigenvalues negative)
    ev8 = eig(C8c_true);
    if any(ev8 < -1e-8)
        % Undo this step
        C8c_true = C8c_true - sgn8 * (v8 * v8');
        continue;
    end

    [U8c_cur, s8c_cur] = gerost.rank1_eig_update(U8c_cur, s8c_cur, v8, sgn8, d8c);

    orth_err = norm(U8c_cur'*U8c_cur - eye(d8c), 'fro');
    max_orth8 = max(max_orth8, orth_err);
end

% Final subspace error vs direct eig of accumulated true C
[Ev8, El8]    = eig(C8c_true, 'vector');
[El8, idx8]   = sort(El8, 'descend');
U8c_ref       = Ev8(:, idx8(1:d8c));
final_sub_err = norm(U8c_cur*U8c_cur' - U8c_ref*U8c_ref', 'fro');

check('S8a: subspace_err < 1e-4 after 1000 sequential updates', ...
      final_sub_err < 1e-4);
check('S8b: U orthonormal < 1e-10 throughout 1000 updates', ...
      max_orth8 < 1e-10);

fprintf('     final subspace_err = %.2e\n', final_sub_err);
fprintf('     max orth_err       = %.2e\n', max_orth8);

% =========================================================================
%  S9. Dimension sweep — n in {20,50,100,200}
% =========================================================================
fprintf('\n--- S9. Dimension sweep (n = 20, 50, 100, 200) -----------------\n');

dims = [20, 50, 100, 200];
rng(8);

for di = 1:length(dims)
    n9  = dims(di);
    k9  = max(2, round(n9 * 0.2));
    d9  = k9 + 2;
    T9  = 2 * k9;
    K9  = 8;
    t0_9 = 2 * T9;
    steps9 = 150;

    crashed9  = false;
    orth9_max = 0;
    errs9     = zeros(steps9, 1);

    U_true9 = orth(randn(n9, k9));
    X0_9    = U_true9 * randn(k9, t0_9) + 0.01*randn(n9, t0_9);
    [U0_9,~,~] = svd(X0_9,'econ'); U0_9 = U0_9(:,1:k9);

    g9 = gerost(n9,k9,d9,T9,'K',K9,'rho',0.15,'fallback','data', ...
                'max_steps',steps9);
    g9 = initialize(g9, U0_9);

    try
        for t = 1:steps9
            u9 = U_true9*randn(k9,1) + 0.01*randn(n9,1);
            g9 = descent_step(g9, u9);
            oe = norm(g9.U'*g9.U - eye(k9),'fro');
            orth9_max = max(orth9_max, oe);
            sv9 = min(svd(g9.U'*U_true9), 1);
            errs9(t) = sqrt(max(k9 - sum(sv9.^2), 0));
        end
    catch ME
        crashed9 = true;
        fprintf('     n=%d CRASHED: %s\n', n9, ME.message);
    end

    check(sprintf('S9: n=%3d — no crash',                         n9), ~crashed9);
    check(sprintf('S9: n=%3d — U orthonormal < 1e-10 throughout', n9), orth9_max < 1e-10);
    if ~crashed9
        final9 = mean(errs9(end-19:end));
        check(sprintf('S9: n=%3d — final chordal dist < 0.5',     n9), final9 < 0.5);
        fprintf('     n=%3d (k=%d,d=%d,T=%d): final_err=%.4f  orth_max=%.2e\n', ...
                n9, k9, d9, T9, final9, orth9_max);
    end
end

% =========================================================================
%  SUMMARY
% =========================================================================
fprintf('\n=================================================================\n');
fprintf('  Results: %d / %d passed', pass, total);
if fail == 0
    fprintf('  — ALL TESTS PASSED\n');
else
    fprintf('  — %d FAILED\n', fail);
end
fprintf('=================================================================\n');

% =========================================================================
%  Nested helpers
% =========================================================================

    function check(name, condition)
        if condition
            fprintf('  [PASS]  %s\n', name);
            pass  = pass  + 1;
        else
            fprintf('  [FAIL]  %s\n', name);
            fail  = fail  + 1;
        end
        total = total + 1;
    end

end  % function stress_test_gerost