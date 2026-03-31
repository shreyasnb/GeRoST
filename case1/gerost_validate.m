function [gerost_val, great_val, params] = ...
         gerost_validate(U_0, u, y, T_train, params)
% GEROST_VALIDATE  Joint hyperparameter search for GeRoST and GREAT.
%
%   [gerost_val, great_val, params] = gerost_validate(U_0, u, y, T_train, params)
%
%   Structure:
%     - outer loop over order_range  (sets subspace dimension d = m*L + order)
%     - inner loop over T_d_range    (sliding-window length)
%     - innermost loop over rho_range (GeRoST only)
%
%   At each (order, T_d) pair GREAT is also evaluated so both methods
%   share the same optimal order and T_d before rho is selected.
%
%   Required params fields
%   ----------------------
%   L, T_ini, T_fut, K, max_steps
%   order_range  : vector of order values to sweep
%   T_d_range    : vector of window lengths to sweep
%   rho_range    : vector of ball radii to sweep (GeRoST)
%
%   Sets in params on return
%   ------------------------
%   order, T_d, rho, d  (optimal values)
%
%   Returns
%   -------
%   gerost_val : gerost object warmed up on all of u/y with optimal params
%   great_val  : great  object warmed up on all of u/y with optimal params

m = size(u, 1);
p = size(y, 1);
n = params.L * (m + p);

nO   = length(params.order_range);
nTd  = length(params.T_d_range);
nRho = length(params.rho_range);

err_great  = nan(nO, nTd);
err_gerost = nan(nO, nTd, nRho);

% ---- sweep order × T_d -----------------------------------------------
for idx_o = 1:nO
    fprintf('GeRoST/GREAT validate: order %i/%i\n', idx_o, nO);

    order_i = params.order_range(idx_o);
    d_i     = m*params.L + order_i;         % subspace dim (ss_initialize convention)

    % Trim U_0 columns to current d_i
    U_hat_0 = U_0(:, 1:d_i);

    for idx_td = 1:nTd
        Td_i = params.T_d_range(idx_td);

        % ---- GREAT ----
        gr = great(n, d_i, Td_i, 'K', params.K, 'max_steps', params.max_steps);
        gr = gr.initialize(U_hat_0);
        err_great(idx_o, idx_td) = 0;

        for i = T_train + 1 : size(u, 2)
            w_i = make_sample(u, y, i, params.L, m, p);
            gr  = gr.descent_step(w_i);

            if i <= size(u, 2) - params.T_fut
                pred = subspace_predict(gr.U, ...
                           u(:, i-params.T_ini+1:i), ...
                           y(:, i-params.T_ini+1:i), ...
                           u(:, i+1:i+params.T_fut), params);
                err_great(idx_o, idx_td) = err_great(idx_o, idx_td) + ...
                    norm(pred(:, params.T_ini+1:end) - y(:, i+1:i+params.T_fut), 'fro');
            end
        end

        % ---- GeRoST — sweep rho at this (order, T_d) ------------------
        d_win_i = min(d_i + order_i, n - 1);

        for idx_rho = 1:nRho
            rho_i = params.rho_range(idx_rho);

            gst = gerost(n, d_i, d_win_i, Td_i, ...
                         'K', params.K, 'rho', rho_i, ...
                         'missing', false, 'max_steps', params.max_steps);
            gst = gst.initialize(U_hat_0);
            err_gerost(idx_o, idx_td, idx_rho) = 0;

            for i = T_train + 1 : size(u, 2)
                w_i  = make_sample(u, y, i, params.L, m, p);
                gst  = gst.descent_step(w_i);

                if i <= size(u, 2) - params.T_fut
                    pred = subspace_predict(gst.U, ...
                               u(:, i-params.T_ini+1:i), ...
                               y(:, i-params.T_ini+1:i), ...
                               u(:, i+1:i+params.T_fut), params);
                    err_gerost(idx_o, idx_td, idx_rho) = ...
                        err_gerost(idx_o, idx_td, idx_rho) + ...
                        norm(pred(:, params.T_ini+1:end) - y(:, i+1:i+params.T_fut), 'fro');
                end
            end
        end
    end
end

% ---- Select optimal hyperparameters ----------------------------------

% GREAT: best (order, T_d)
[~, idx_gr] = min(err_great(:));
[io_gr, itd_gr] = ind2sub([nO, nTd], idx_gr);
params.order = params.order_range(io_gr);
params.T_d   = params.T_d_range(itd_gr);
params.d     = m*params.L + params.order;
fprintf('GREAT optimal: order=%i, T_d=%i\n', params.order, params.T_d);

% GeRoST: best (order, T_d, rho)  — use same (order, T_d) as GREAT for
%         fair comparison; only tune rho at those fixed values.
rho_errors = squeeze(err_gerost(io_gr, itd_gr, :));
[~, idx_rho] = min(rho_errors);
params.rho = params.rho_range(idx_rho);
fprintf('GeRoST optimal: rho=%.3f (at order=%i, T_d=%i)\n', ...
        params.rho, params.order, params.T_d);

% ---- Rebuild final objects warmed up on ALL of u/y -------------------
U_hat_0_opt = U_0(:, 1:params.d);
d_win_opt   = min(params.d + params.order, n - 1);

great_val = great(n, params.d, params.T_d, ...
                  'K', params.K, 'max_steps', params.max_steps);
great_val = great_val.initialize(U_hat_0_opt);

gerost_val = gerost(n, params.d, d_win_opt, params.T_d, ...
                    'K', params.K, 'rho', params.rho, ...
                    'missing', false, 'max_steps', params.max_steps);
gerost_val = gerost_val.initialize(U_hat_0_opt);

for i = T_train + 1 : size(u, 2)
    w_i        = make_sample(u, y, i, params.L, m, p);
    great_val  = great_val.descent_step(w_i);
    gerost_val = gerost_val.descent_step(w_i);
end

% ---- Local helper ----------------------------------------------------
function s = make_sample(u_, y_, i_, L_, m_, p_)
    s = [reshape(u_(:, i_-L_+1:i_), [L_*m_, 1]);
         reshape(y_(:, i_-L_+1:i_), [L_*p_, 1])];
end

end