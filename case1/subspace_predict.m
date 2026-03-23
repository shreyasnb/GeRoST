function pred_y = subspace_predict(U, u_ini, y_ini, u_fut, params)
% SUBSPACE_PREDICT  Predict future outputs from a behavioural subspace.
%
%   pred_y = subspace_predict(U, u_ini, y_ini, u_fut, params)
%
%   Drop-in equivalent of ss_predict.m for gerost / great subspace
%   estimates.  Logic is IDENTICAL to ss_predict.m.
%
%   Row layout of U  (n = L*(m+p) rows)
%   ------------------------------------
%   1  :  m*L              full L-step input window  (u_ini stacked with u_fut)
%   m*L+1  :  m*L+p*T_ini  past T_ini output rows    (known)
%   m*L+p*T_ini+1  :  end  future output rows         (to predict)
%
%   Parameters
%   ----------
%   U      : (n x d)  orthonormal subspace basis
%   u_ini  : (m x T_ini)  past inputs
%   y_ini  : (p x T_ini)  past outputs
%   u_fut  : (m x T_fut)  future inputs
%   params : struct with fields  L, T_ini
%
%   Returns
%   -------
%   pred_y : (p x L)  predicted output trajectory over the full L-step window.
%            Caller slices  pred_y(:, T_ini+1:end)  to get T_fut future outputs,
%            mirroring the  ss_pred_y(:, T_ini+1:end)  pattern in sysID_main.m.

m = size(u_ini, 1);
p = size(y_ini, 1);

% Identical to ss_predict.m
u_given = reshape([u_ini, u_fut], [m*params.L, 1]);
y_given = reshape(y_ini,          [p*params.T_ini, 1]);
w_given = [u_given; y_given];

U_given = U(1 : m*params.L + p*params.T_ini, :);

pred_traj = U * pinv(U_given) * w_given;
pred_y    = reshape(pred_traj(m*params.L+1:end, 1), [p, params.L]);
end