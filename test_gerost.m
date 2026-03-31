function test_gerost()
%% test_gerost  —  Sanity test suite for gerost.m
%
%  Run from the MATLAB command window with:
%    >> test_gerost
%
%  Requires Manopt on the path (for grassmannfactory used by gerost).
%
%  Sections
%  --------
%  1.  Helpers         (nested: check, subspace_err)
%  2.  rank1_eig_update (static method)
%        2a. Update (+1), v outside subspace  (beta > 0 branch)
%        2b. Update (+1), v inside  subspace  (beta ~ 0 branch)
%        2c. Downdate (-1)
%        2d. Output columns orthonormal
%        2e. Singular values non-negative and sorted descending
%  3.  Constructor     — every stored property
%  4.  initialize()   — subspace assignment
%  5.  get_rho()      — scalar, 4-arg handle, 3-arg handle
%  6.  impute()       — full passthrough, empty mask, partial mask
%  7.  build_window_matrix()
%        7a. Warm-up: What is (n x d) orthonormal
%        7b. Warm-up: sigma_vals non-negative and sorted descending
%        7c. Cache (U_what / sigma_vals_w) populated after warm-up
%        7d. Rank-2 path: subspace matches full SVD
%        7e. Rank-2 path: Gram matrix matches ground-truth top-d approx
%  8.  descent_step()
%        8a. t increments by 1 every call
%        8b. sample stored at correct column
%        8c. window_idx grows correctly for t <= T
%        8d. window_idx slides correctly for t >  T
%        8e. U stays on Grassmann (U'U = I_k) after every step
%        8f. U_history written at each step (after warm-up)
%        8g. U unchanged during warm-up skip (window < d)
%        8h. Cache non-empty after warm-up completes
%  9.  rho clamping   — values outside [1e-6, sqrt(k)-1e-6] don't crash
%  10. Fallback modes — 'data' and 'subspace' both complete without error
%  11. Missing data   — descent_step runs with partial observation masks
%  12. Function-handle rho — adaptive radius works inside descent_step
%  13. Integration    — chordal distance decreases over 200 steps

% =========================================================================
%  Shared counters  (visible to nested functions check / subspace_err)
% =========================================================================
pass  = 0;
fail  = 0;
total = 0;

fprintf('=================================================================\n');
fprintf('  GeRoST sanity tests\n');
fprintf('=================================================================\n\n');

tol    = 1e-8;   % tight tolerance for exact operations
tol_sv = 1e-5;   % looser tolerance for rank-2 SVD comparisons

rng(42);         % reproducible random seed

% =========================================================================
%  SECTION 2 — rank1_eig_update  (static method, no Manopt required)
% =========================================================================
fprintf('--- 2. rank1_eig_update (static) --------------------------------\n');

n_r = 40;  d_r = 6;

% Random rank-d PSD matrix expressed via its eigen-factorisation
[U_r, ~] = qr(randn(n_r, d_r), 0);             % (n x d) orthonormal
s_r      = sort(5*rand(d_r,1)+1, 'descend');    % (d x 1) positive sing vals
C_r      = U_r * diag(s_r.^2) * U_r';           % (n x n) PSD, rank d

% --- 2a. Update (+1), v outside subspace (beta > 0) ----------------------
v_out = randn(n_r, 1);
v_out = v_out - U_r*(U_r'*v_out);    % make v exactly perp to span(U_r)
v_out = v_out / norm(v_out) * 3;

C_ref         = C_r + v_out*v_out';
[Ev, El]      = eig(C_ref, 'vector');
[El, idx]     = sort(El, 'descend');
U_ref         = Ev(:, idx(1:d_r));

[U_upd, s_upd] = gerost.rank1_eig_update(U_r, s_r, v_out, +1, d_r);

check('2a: update (+1) outside subspace — subspace error', ...
      subspace_err(U_upd, U_ref) < tol_sv);
check('2a: update (+1) outside subspace — eigenvalues', ...
      norm(sort(s_upd,'descend').^2 - El(1:d_r)) < tol_sv);

% --- 2b. Update (+1), v inside subspace (beta ~ 0) -----------------------
v_in = U_r * randn(d_r, 1);          % exactly in span(U_r)

C_in          = C_r + v_in*v_in';
[Ev_i, El_i]  = eig(C_in, 'vector');
[El_i, idx_i] = sort(El_i, 'descend');
U_ref_in      = Ev_i(:, idx_i(1:d_r));

[U_upd_in, ~] = gerost.rank1_eig_update(U_r, s_r, v_in, +1, d_r);

check('2b: update (+1) inside subspace — subspace error', ...
      subspace_err(U_upd_in, U_ref_in) < tol_sv);

% --- 2c. Downdate (-1) ---------------------------------------------------
% Keep v small enough that C_r - v*v' stays PSD
v_dn  = U_r * (randn(d_r,1) * 0.3);
C_dn  = C_r - v_dn*v_dn';

ev_all = eig(C_dn);
if all(ev_all >= -1e-10)
    [Ev_d, El_d]  = eig(C_dn, 'vector');
    [El_d, idx_d] = sort(El_d, 'descend');
    U_ref_dn      = Ev_d(:, idx_d(1:d_r));
    [U_dn, ~]     = gerost.rank1_eig_update(U_r, s_r, v_dn, -1, d_r);
    check('2c: downdate (-1) — subspace error', ...
          subspace_err(U_dn, U_ref_dn) < tol_sv);
else
    fprintf('  [SKIP]  2c: C_dn not PSD for this random seed — skipping\n');
end

% --- 2d. Output columns orthonormal --------------------------------------
check('2d: update (+1) — U_new orthonormal', ...
      norm(U_upd'*U_upd - eye(d_r), 'fro') < tol);

% --- 2e. Singular values non-negative and sorted descending --------------
check('2e: s_new all non-negative',       all(s_upd >= 0));
check('2e: s_new sorted descending',      issorted(flipud(s_upd)));

% =========================================================================
%  SECTION 3 — Constructor
% =========================================================================
fprintf('\n--- 3. Constructor ----------------------------------------------\n');

n=30; k=5; d=7; T=15; K_it=10; max_s=200;
g = gerost(n, k, d, T, 'K', K_it, 'rho', 0.2, 'alpha', 0.01, ...
           'missing', false, 'fallback', 'data', 'max_steps', max_s);

check('3: n',                             g.n  == n);
check('3: k',                             g.k  == k);
check('3: d',                             g.d  == d);
check('3: T',                             g.T  == T);
check('3: K',                             g.K  == K_it);
check('3: rho_param scalar',              g.rho_param   == 0.2);
check('3: alpha_param scalar',            g.alpha_param == 0.01);
check('3: missing = false',               g.missing == false);
check('3: fallback = data',               strcmp(g.fallback,'data'));
check('3: t = 0',                         g.t == 0);
check('3: U empty',                       isempty(g.U));
check('3: U_what empty',                  isempty(g.U_what));
check('3: sigma_vals_w empty',            isempty(g.sigma_vals_w));
check('3: samples size (n x max_s)',      isequal(size(g.samples),   [n, max_s]));
check('3: masks_full size (n x max_s)',   isequal(size(g.masks_full),[n, max_s]));
check('3: masks_full all true',           all(g.masks_full(:)));
check('3: U_history size (n x k x max_s)', ...
      isequal(size(g.U_history), [n, k, max_s]));

% =========================================================================
%  SECTION 4 — initialize()
% =========================================================================
fprintf('\n--- 4. initialize() --------------------------------------------\n');

U0 = orth(randn(n, k));
g  = initialize(g, U0);

check('4: U equals U0',        norm(g.U - U0, 'fro') < tol);
check('4: U orthonormal',      norm(g.U'*g.U - eye(k), 'fro') < tol);

% =========================================================================
%  SECTION 5 — get_rho()
% =========================================================================
fprintf('\n--- 5. get_rho() -----------------------------------------------\n');

g_rho      = gerost(n, k, d, T, 'rho', 0.35, 'max_steps', max_s);
g_rho      = initialize(g_rho, U0);
sigma_dummy = rand(d,1) + 1;
What_dummy  = orth(randn(n, d));

rho_out = get_rho(g_rho, sigma_dummy, What_dummy, U0);
check('5: scalar rho returned unchanged', abs(rho_out - 0.35) < tol);

% 4-argument handle
rho_fn4  = @(t, sv, W, Y) sv(1) / (t + 1);
g_rho4   = gerost(n, k, d, T, 'rho', rho_fn4, 'max_steps', max_s);
g_rho4.t = 3;
rho4     = get_rho(g_rho4, sigma_dummy, What_dummy, U0);
check('5: 4-arg handle evaluated correctly', ...
      abs(rho4 - sigma_dummy(1)/4) < tol);

% 3-argument handle (fallback via try/catch inside get_rho)
rho_fn3  = @(t, sv, W) sv(1) / (t + 1);
g_rho3   = gerost(n, k, d, T, 'rho', rho_fn3, 'max_steps', max_s);
g_rho3.t = 3;
rho3     = get_rho(g_rho3, sigma_dummy, What_dummy, U0);
check('5: 3-arg handle evaluated correctly', ...
      abs(rho3 - sigma_dummy(1)/4) < tol);

% =========================================================================
%  SECTION 6 — impute()
% =========================================================================
fprintf('\n--- 6. impute() ------------------------------------------------\n');

g_imp = gerost(n, k, d, T, 'missing', true, 'max_steps', max_s);
g_imp = initialize(g_imp, U0);

u_full          = randn(n, 1);
u_out_full      = impute(g_imp, u_full, true(n,1));
u_out_empty     = impute(g_imp, u_full, []);

check('6: fully-observed passthrough',    norm(u_out_full  - u_full) < tol);
check('6: empty-mask passthrough',        norm(u_out_empty - u_full) < tol);

% Partial mask: vector exactly in span(U0) must be perfectly reconstructed
u_sub       = U0 * randn(k, 1);          % exactly in subspace
mask_part   = true(n, 1);
mask_part(floor(n/2)+1:end) = false;     % upper half observed

% Rank condition: need at least k observed rows for U_obs \ u_obs to be valid
assert(sum(mask_part) >= k, 'Test setup error: not enough observed entries');

u_imp = impute(g_imp, u_sub, mask_part);
check('6: partial mask — observed entries unchanged', ...
      norm(u_imp(mask_part) - u_sub(mask_part)) < tol);
check('6: partial mask — in-subspace vector fully recovered', ...
      norm(u_imp - u_sub) < 1e-6);

% =========================================================================
%  SECTION 7 — build_window_matrix()
% =========================================================================
fprintf('\n--- 7. build_window_matrix() -----------------------------------\n');

g7      = gerost(n, k, d, T, 'K', 3, 'rho', 0.1, 'max_steps', 200);
% Use a d-dimensional data subspace (not k with noise).
% The window matrix W then has rank exactly d, so the top-d truncation
% inside rank1_eig_update is lossless and the rank-2 update is
% algebraically exact (floating-point error ~1e-13, well within 1e-4).
U_data7 = orth(randn(n, d));             % d-dim basis for data generation
g7      = initialize(g7, orth(randn(n, k)));

% Manually load exactly T samples (warm-up path, bypass descent_step)
for t = 1:T
    g7.t = g7.t + 1;
    g7.samples(:, g7.t) = U_data7 * randn(d, 1);   % exactly rank-d, no noise
    g7.window_idx = 1:g7.t;
end

[g7, What7, sv7] = build_window_matrix(g7);

% 7a. Shape and orthonormality of What
check('7a: What size is (n x d)',         isequal(size(What7), [n, d]));
check('7a: What columns orthonormal',     norm(What7'*What7 - eye(d),'fro') < tol);

% 7b. Singular values
check('7b: sigma_vals non-negative',      all(sv7 >= -tol));
check('7b: sigma_vals sorted descending', issorted(flipud(sv7(1:d))));

% 7c. Cache populated
check('7c: U_what non-empty',             ~isempty(g7.U_what));
check('7c: sigma_vals_w non-empty',       ~isempty(g7.sigma_vals_w));
check('7c: U_what size (n x d)',          isequal(size(g7.U_what), [n, d]));
check('7c: sigma_vals_w length = d',      length(g7.sigma_vals_w) == d);

% 7d & 7e. Add one more sample to trigger the rank-2 steady-state path
g7.t = g7.t + 1;
g7.samples(:, g7.t) = U_data7 * randn(d, 1);       % exactly rank-d, no noise
g7.window_idx = (g7.t - T + 1):g7.t;

[g7, What7_r2, sv7_r2] = build_window_matrix(g7);   % rank-2 branch

% Reference: full SVD of the true current window
W_ref7          = g7.samples(:, g7.window_idx);
[Usvd7, S7, ~] = svd(W_ref7, 'econ');
What7_ref       = Usvd7(:, 1:d);

check('7d: rank-2 subspace matches full SVD', ...
      subspace_err(What7_r2, What7_ref) < 1e-4);

% 7e. Gram matrix identity
Gram_r2  = What7_r2 * diag(sv7_r2.^2) * What7_r2';
Gram_ref = W_ref7 * W_ref7';
[Uv, Sv] = eigs(Gram_ref, d);
Gram_ref_d = Uv * Sv * Uv';
rel_err = norm(Gram_r2 - Gram_ref_d,'fro') / (norm(Gram_ref_d,'fro') + 1e-12);
check('7e: rank-2 Gram matches true Gram (top-d, relative error < 1e-3)', ...
      rel_err < 1e-3);

% =========================================================================
%  SECTION 8 — descent_step()
% =========================================================================
fprintf('\n--- 8. descent_step() ------------------------------------------\n');

n8=30; k8=5; d8=6; T8=12; steps8=30;
g8      = gerost(n8, k8, d8, T8, 'K', 5, 'rho', 0.15, 'max_steps', 60);
U_true8 = orth(randn(n8, k8));
g8      = initialize(g8, orth(randn(n8, k8)));

for t8 = 1:steps8
    u8       = U_true8*randn(k8,1) + 0.01*randn(n8,1);
    t_before = g8.t;
    g8       = descent_step(g8, u8);

    % 8a. time counter
    check(sprintf('8a: t increments (step %2d)', t8), g8.t == t_before + 1);

    % 8b. sample stored
    check(sprintf('8b: sample stored at col %2d', t8), ...
          norm(g8.samples(:, g8.t) - u8) < tol);

    % 8c / 8d. window indices
    if g8.t <= T8
        check(sprintf('8c: window_idx growing (t=%2d)', g8.t), ...
              isequal(g8.window_idx, 1:g8.t));
    else
        check(sprintf('8d: window_idx sliding (t=%2d)', g8.t), ...
              isequal(g8.window_idx, (g8.t-T8+1):g8.t));
    end

    % 8e. U on Grassmannian
    if ~isempty(g8.U)
        check(sprintf('8e: U orthonormal (step %2d)', t8), ...
              norm(g8.U'*g8.U - eye(k8),'fro') < 1e-10);
    end

    % 8f. U_history written (after warm-up skip is past)
    if g8.t >= d8
        check(sprintf('8f: U_history written (step %2d)', t8), ...
              norm(g8.U_history(:,:,g8.t) - g8.U,'fro') < tol);
    end
end

% 8g. Warm-up skip: U must be unchanged for the first d-1 samples
g8g       = gerost(n8, k8, d8, T8, 'K', 5, 'rho', 0.15, 'max_steps', 60);
g8g       = initialize(g8g, orth(randn(n8, k8)));
U_skip    = g8g.U;
for ii = 1:(d8-1)
    g8g = descent_step(g8g, randn(n8,1));
end
check('8g: U unchanged during warm-up skip (window < d)', ...
      norm(g8g.U - U_skip,'fro') < tol);

% 8h. Cache non-empty after warm-up
check('8h: U_what non-empty after full run',       ~isempty(g8.U_what));
check('8h: sigma_vals_w non-empty after full run', ~isempty(g8.sigma_vals_w));

% =========================================================================
%  SECTION 9 — rho clamping
% =========================================================================
fprintf('\n--- 9. rho clamping --------------------------------------------\n');

U_true9 = orth(randn(n, k));

g9a = gerost(n, k, d, T, 'rho', sqrt(k)+5, 'K', 3, 'max_steps', 100);
g9a = initialize(g9a, orth(randn(n,k)));
ok9a = true;
try
    for ii = 1:d+1
        g9a = descent_step(g9a, U_true9*randn(k,1));
    end
catch ME
    ok9a = false;
    fprintf('    Error (rho too large): %s\n', ME.message);
end
check('9: rho_too_large — no crash', ok9a);

g9b = gerost(n, k, d, T, 'rho', -3, 'K', 3, 'max_steps', 100);
g9b = initialize(g9b, orth(randn(n,k)));
ok9b = true;
try
    for ii = 1:d+1
        g9b = descent_step(g9b, U_true9*randn(k,1));
    end
catch ME
    ok9b = false;
    fprintf('    Error (rho too small): %s\n', ME.message);
end
check('9: rho_too_small — no crash', ok9b);

% =========================================================================
%  SECTION 10 — fallback modes
% =========================================================================
fprintf('\n--- 10. fallback modes -----------------------------------------\n');

for fb = {'data', 'subspace'}
    g10 = gerost(n, k, d, T, 'K', 3, 'rho', 0.1, ...
                 'fallback', fb{1}, 'max_steps', 100);
    g10 = initialize(g10, orth(randn(n,k)));
    ok  = true;
    try
        for ii = 1:T+5
            g10 = descent_step(g10, U_true9*randn(k,1) + 0.01*randn(n,1));
        end
    catch ME
        ok = false;
        fprintf('    Error (%s fallback): %s\n', fb{1}, ME.message);
    end
    check(sprintf('10: fallback="%s" — no runtime error',   fb{1}), ok);
    check(sprintf('10: fallback="%s" — U orthonormal',      fb{1}), ...
          norm(g10.U'*g10.U - eye(k),'fro') < 1e-10);
end

% =========================================================================
%  SECTION 11 — missing data
% =========================================================================
fprintf('\n--- 11. missing data -------------------------------------------\n');

g11  = gerost(n, k, d, T, 'K', 3, 'rho', 0.1, 'missing', true, 'max_steps', 100);
g11  = initialize(g11, orth(randn(n,k)));
ok11 = true;
try
    for ii = 1:T+5
        u11   = U_true9*randn(k,1) + 0.01*randn(n,1);
        m11   = rand(n,1) > 0.3;    % ~70 % observed
        m11(1:k) = true;             % guarantee rank condition
        g11 = descent_step(g11, u11, 'mask', m11);
    end
catch ME
    ok11 = false;
    fprintf('    Error (missing): %s\n', ME.message);
end
check('11: missing=true — no runtime error',  ok11);
check('11: missing=true — U orthonormal', ...
      norm(g11.U'*g11.U - eye(k),'fro') < 1e-10);

% =========================================================================
%  SECTION 12 — function-handle rho inside descent_step
% =========================================================================
fprintf('\n--- 12. function-handle rho in descent_step --------------------\n');

rho_fn = @(t, sv, W, Y) max(sv(1) / (2*t+1), 0.05);
g12    = gerost(n, k, d, T, 'K', 3, 'rho', rho_fn, 'max_steps', 100);
g12    = initialize(g12, orth(randn(n,k)));
ok12   = true;
try
    for ii = 1:T+10
        g12 = descent_step(g12, U_true9*randn(k,1) + 0.01*randn(n,1));
    end
catch ME
    ok12 = false;
    fprintf('    Error (fn-handle rho): %s\n', ME.message);
end
check('12: function-handle rho — no runtime error', ok12);
check('12: function-handle rho — U orthonormal', ...
      norm(g12.U'*g12.U - eye(k),'fro') < 1e-10);

% =========================================================================
%  SECTION 13 — Integration / convergence
% =========================================================================
fprintf('\n--- 13. Integration: convergence --------------------------------\n');

n13=50; k13=8; d13=10; T13=20; K13=15; num_steps13=200;

U_true13 = orth(randn(n13, k13));

% Warm-start from 60 samples of pre-collected data
t0   = 60;
X0   = U_true13*randn(k13,t0) + 0.01*randn(n13,t0);
[U0_13,~,~] = svd(X0,'econ');
U0_13 = U0_13(:, 1:k13);

g13 = gerost(n13, k13, d13, T13, 'K', K13, 'rho', 0.15, ...
             'fallback', 'data', 'max_steps', num_steps13);
g13 = initialize(g13, U0_13);

errors13 = zeros(1, num_steps13);
for t13 = 1:num_steps13
    u13 = U_true13*randn(k13,1) + 0.01*randn(n13,1);
    g13 = descent_step(g13, u13);
    sv13 = min(svd(g13.U'*U_true13), 1);
    errors13(t13) = sqrt(max(k13 - sum(sv13.^2), 0));
end

early_err = mean(errors13(1:20));
final_err = mean(errors13(end-19:end));

check('13: final chordal distance < 0.5',                  final_err < 0.5);
check('13: error decreases (final window < early window)',  final_err < early_err);
check('13: U orthonormal after 200 steps', ...
      norm(g13.U'*g13.U - eye(k13),'fro') < 1e-10);

fprintf('\n  Early error  (steps   1-20): %.6f\n', early_err);
fprintf('  Final error  (steps 181-200): %.6f\n', final_err);

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
%  Nested helper functions  (share pass / fail / total with parent scope)
% =========================================================================

    function check(name, condition)
        % Report a single pass/fail result and update counters.
        if condition
            fprintf('  [PASS]  %s\n', name);
            pass  = pass  + 1;
        else
            fprintf('  [FAIL]  %s\n', name);
            fail  = fail  + 1;
        end
        total = total + 1;
    end

    function result = subspace_err(A, B)
        % Frobenius distance between projection matrices P_A and P_B.
        result = norm(A*A' - B*B', 'fro');
    end

end  % function test_gerost