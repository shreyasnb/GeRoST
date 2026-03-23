function [gerost_obj, great_obj, U0, params] = ...
         subspace_initialize(u, y, params)
% SUBSPACE_INITIALIZE  Build initial behavioural subspace and seed tracker objects.
%
%   [gerost_obj, great_obj, U0, params] = subspace_initialize(u, y, params)
%
%   Mirrors ss_initialize.m exactly for the Hankel construction and SVD.
%   Additionally instantiates gerost and great objects seeded with U0.
%
%   Required params fields
%   ----------------------
%   L          : Hankel window length
%   order      : system-order estimate  ->  d = m*L + order  (as in ss_initialize)
%   T_d        : sliding-window length (number of Hankel columns)
%   K          : gradient-descent iterations per step
%   rho        : GeRoST ball radius (scalar or function handle)
%   T_ini      : past steps supplied at prediction time
%   T_fut      : prediction horizon
%   max_steps  : pre-allocation depth (default: 2000)
%
%   Augments params with:  m, p, n, d (subspace dim), Gr (Grassmannfactory)

if ~isfield(params, 'max_steps')
    params.max_steps = 2000;
end

m = size(u, 1);
p = size(y, 1);
n = params.L * (m + p);

params.m = m;
params.p = p;
params.n = n;

% --- Mirror ss_initialize.m exactly -----------------------------------
H_U = construct_hankel(u, params.L);
H_Y = construct_hankel(y, params.L);
W_0 = [H_U; H_Y];

d   = m*params.L + params.order;      % subspace dimension (ss_initialize convention)
params.d = d;

[U_svd, ~, ~] = svd(W_0, 'econ');
U0 = U_svd(:, 1:d);

params.Gr = grassmannfactory(n, d);   % also stored so ss_update still works if called

% --- Initialise GREAT (k = d, same subspace dimension) ----------------
great_obj = great(n, d, params.T_d, ...
                  'K', params.K, 'max_steps', params.max_steps);
great_obj = great_obj.initialize(U0);

% --- Initialise GeRoST
%   gerost k  = d  (subspace to track)
%   gerost d  = d_win  (data-subspace dim for inner max; d_win >= k)
%   We set d_win = min(d + params.order, n-1) to give the window matrix
%   a slightly larger rank budget than the tracked subspace.
d_win = min(d + params.order, n - 1);
gerost_obj = gerost(n, d, d_win, params.T_d, ...
                    'K', params.K, 'rho', params.rho, ...
                    'missing', false, 'max_steps', params.max_steps);
gerost_obj = gerost_obj.initialize(U0);

end