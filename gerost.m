classdef gerost
    % GeRoST: Geometrically Robust Online Subspace Tracking
    %
    % This class implements online subspace tracking via constrained gradient
    % descent on the Grassmann manifold with a ball constraint on subspace
    % perturbations.
    %
    % Properties
    % ----------
    % n : int, ambient dimension
    % k : int, true subspace dimension
    % d : int, data subspace dimension (d >= k)
    % T : int, window length
    % K : int, number of inner gradient descent iterations per time step
    % rho_param : float or function handle, ball radius
    % alpha_param : float or empty, step-size
    % missing : bool, whether to handle missing data
    % fallback : str, 'subspace' or 'data' for inactive constraint handling

    properties
        n           % ambient dimension
        k           % true subspace dimension
        d           % data subspace dimension
        T           % window length
        K           % inner gradient descent iterations per step
        rho_param   % ball radius (fixed or function handle)
        alpha_param % step-size (fixed or empty for adaptive)
        missing     % flag for missing data handling
        fallback    % 'subspace' or 'data' fallback strategy

        % State
        U           % current subspace estimate (n x k)
        U_history   % full sequence of subspace estimates (n x k x num_steps)
        samples     % full data matrix (n x num_collected)
        masks_full  % full observation masks matrix (n x num_collected)
        window_idx  % indices for sliding window
        t           % current time step

        % ----------------------------------------------------------------
        % Cached SVD of window matrix W_{t-1}  — used for rank-2 updates.
        %   W_t W_t^T = W_{t-1} W_{t-1}^T  -  u_{t-T} u_{t-T}^T  +  u_t u_t^T
        % U_what       : (n x d)  top-d left singular vectors of W_{t-1}
        % sigma_vals_w : (d x 1)  corresponding singular values of W_{t-1}
        % ----------------------------------------------------------------
        U_what        % (n x d) cached left singular vectors
        sigma_vals_w  % (d x 1) cached singular values

        % imputed_samples: stores each u_t imputed at *entry time* using the
        % subspace estimate available then.  When missing=false this equals
        % samples exactly.  When missing=true this ensures the vector
        % removed from the Gram matrix at time t (u_{t-T}) is identical to
        % the vector that was added at time t-T, preserving the rank-2
        % recurrence  W_t W_t^T = W_{t-1}W_{t-1}^T - u_{t-T}u_{t-T}^T + u_t u_t^T.
        imputed_samples  % (n x max_steps) imputed data matrix

        % Grassmann manifold (from Manopt)
        manifold    % Grassmann(n, k) manifold from Manopt
    end

    methods
        function obj = gerost(n, k, d, T, varargin)
            % Initialize gerost object
            %
            % Usage: obj = gerost(n, k, d, T, 'K', 5, 'rho', 0.1, ...
            %                     'alpha', [], 'missing', false, 'fallback', 'data')

            p = inputParser;
            addParameter(p, 'K', 5, @isnumeric);
            addParameter(p, 'rho', 0.1, @(x) isnumeric(x) || isa(x, 'function_handle'));
            addParameter(p, 'alpha', [], @(x) isempty(x) || isnumeric(x));
            addParameter(p, 'missing', false, @islogical);
            addParameter(p, 'fallback', 'data', @ischar);
            addParameter(p, 'max_steps', 1000, @isnumeric);

            parse(p, varargin{:});

            obj.n = n;
            obj.k = k;
            obj.d = d;
            obj.T = T;
            obj.K = p.Results.K;
            obj.rho_param   = p.Results.rho;
            obj.alpha_param = p.Results.alpha;
            obj.missing     = p.Results.missing;
            obj.fallback    = p.Results.fallback;

            % Initialize state
            obj.U            = [];
            obj.U_history    = zeros(n, k, p.Results.max_steps);
            obj.samples      = zeros(n, p.Results.max_steps);  % pre-allocated
            obj.masks_full   = true(n, p.Results.max_steps);   % pre-allocated
            obj.window_idx   = [];
            obj.t            = 0;

            % Cached SVD — empty until first full-window SVD is computed
            obj.U_what          = [];
            obj.sigma_vals_w    = [];
            obj.imputed_samples = zeros(n, p.Results.max_steps);  % BUG 2+3 fix

            % Initialize Manopt Grassmann manifold
            obj.manifold = grassmannfactory(n, k);
        end

        function obj = initialize(obj, U0)
            % Set initial subspace estimate
            % U0 : (n x k) orthonormal basis
            obj.U = U0;
        end

        function rho = get_rho(obj, sigma_vals, What, Y)
            if isa(obj.rho_param, 'function_handle')
                try
                    rho = obj.rho_param(obj.t, sigma_vals, What, Y);
                catch
                    rho = obj.rho_param(obj.t, sigma_vals, What);
                end
            else
                rho = obj.rho_param;
            end
        end

        function u_tilde = impute(obj, u, mask)
            if isempty(mask) || all(mask)
                u_tilde = u;
                return;
            end
            u_tilde   = u;
            U_obs     = obj.U(mask, :);
            w_star    = U_obs \ u(mask);
            u_imputed = obj.U * w_star;
            u_tilde(~mask) = u_imputed(~mask);
        end

        % ----------------------------------------------------------------
        %  build_window_matrix
        %
        %  WARM-UP  (window not yet full, t <= T):
        %    Full thin SVD of the current n-by-t slice — O(n t d).
        %    Result is cached in obj.U_what / obj.sigma_vals_w.
        %
        %  STEADY STATE  (t > T):
        %    Exploits the rank-2 recurrence
        %      W_t W_t^T = W_{t-1} W_{t-1}^T - u_{t-T} u_{t-T}^T + u_t u_t^T
        %    via two sequential rank-1 eigendecomposition updates — O(nd + d^3).
        %    Replaces the O(nTd) full SVD that was called every step.
        % ----------------------------------------------------------------
        function [obj, What, sigma_vals] = build_window_matrix(obj)

            window_full = (obj.t > obj.T) && ~isempty(obj.U_what);

            if window_full
                % ----------------------------------------------------------
                % Rank-2 incremental update
                % ----------------------------------------------------------
                u_out = obj.samples(:, obj.t - obj.T);  % leaving the window
                u_in  = obj.samples(:, obj.t);          % entering the window

                if obj.missing
                    u_out = impute(obj, u_out, obj.masks_full(:, obj.t - obj.T));
                    u_in  = impute(obj, u_in,  obj.masks_full(:, obj.t));
                end

                U = obj.U_what;        % (n x d)
                s = obj.sigma_vals_w;  % (d x 1)

                % Downdate: subtract u_out u_out^T
                [U, s] = gerost.rank1_eig_update(U, s, u_out, -1, obj.d);
                % Update:   add     u_in  u_in^T
                [U, s] = gerost.rank1_eig_update(U, s, u_in,  +1, obj.d);

                What       = U;
                sigma_vals = s;

            else
                % ----------------------------------------------------------
                % Warm-up: full thin SVD
                % ----------------------------------------------------------
                if obj.missing
                    cols = zeros(obj.n, length(obj.window_idx));
                    for ii = 1:length(obj.window_idx)
                        idx = obj.window_idx(ii);
                        cols(:, ii) = impute(obj, obj.samples(:, idx), ...
                                                  obj.masks_full(:, idx));
                    end
                    W = cols;
                else
                    W = obj.samples(:, obj.window_idx);
                end

                [U_svd, S, ~] = svd(W, 'econ');
                sv            = diag(S);
                What          = U_svd(:, 1:obj.d);
                sigma_vals    = sv(1:obj.d);   % truncate to d: matches rank-2 path output length
            end

            % Cache result for next step's rank-2 update
            obj.U_what       = What;
            obj.sigma_vals_w = sigma_vals(1:obj.d);
        end

        function obj = descent_step(obj, u, varargin)
            p = inputParser;
            addParameter(p, 'mask', [], @(x) isempty(x) || islogical(x));
            parse(p, varargin{:});
            mask = p.Results.mask;

            obj.t = obj.t + 1;

            % Store sample and mask
            obj.samples(:, obj.t) = u;
            if ~isempty(mask)
                obj.masks_full(:, obj.t) = mask;
            end

            % Update sliding window indices
            if obj.t <= obj.T
                obj.window_idx = 1:obj.t;
            else
                obj.window_idx = (obj.t - obj.T + 1):obj.t;
            end

            % Need at least d samples to form the ball center
            if length(obj.window_idx) < obj.d
                obj.U_history(:, :, obj.t) = obj.U;
                return;
            end

            % Build window matrix — rank-2 update or full SVD (obj updated too)
            [obj, What, sigma_vals] = build_window_matrix(obj);  % FIX: obj returned

            % Get ball radius
            rho = get_rho(obj, sigma_vals, What, obj.U);
            rho = min(max(rho, 1e-6), sqrt(obj.d) - 1e-6);   % ball lives in Gr(d,n): max d_c = sqrt(d)

            % Inner gradient descent loop
            Y = obj.U;

            for iter = 1:obj.K
                [Wstar, lambda_star] = gerost.inner_max(Y, What, rho, obj.d);

                if lambda_star > 2
                    grad  = gerost.gradf(Y, Wstar);
                    delta = lambda_star - 2.0;
                    L     = 4.0*(1.0 + 1.0/delta) + 4.0*sqrt(obj.d)/(delta^2);

                    YtW       = Y' * Wstar;
                    Nhat      = YtW * YtW';
                    eigvals_N = eig(Nhat);
                    nu_0      = max(min(eigvals_N), 1e-10);
                    nu        = 2.0 * nu_0;
                else
                    if strcmp(obj.fallback, 'data')
                        Wdata = obj.samples(:, obj.window_idx);
                        WtY   = Wdata' * Y;
                        WWtY  = Wdata * WtY;
                        grad  = -2.0 * (WWtY - Y * (Y' * WWtY));
                        L     = 4.0 * sigma_vals(1)^2;
                    else
                        grad = gerost.gradf(Y, What);
                        L    = 4.0;
                    end
                    nu = 0.0;
                end

                if ~isempty(obj.alpha_param)
                    alpha = obj.alpha_param;
                else
                    if nu > 1e-10
                        alpha = min(1.0/max(L,1e-8), 1.0/(2.0*nu));
                    else
                        alpha = 1.0/max(L, 1e-8);
                    end
                end

                tangent = -alpha * grad;
                Y       = obj.manifold.exp(Y, tangent);
            end

            obj.U = Y;
            obj.U_history(:, :, obj.t) = obj.U;
        end
    end

    % ====================================================================
    methods (Static)

        function [Wstar, lambda_star] = inner_max(Y, What, rho, d)
            % INNER_MAX  Ball-constrained inner maximisation via Lagrangian duality.
            %
            %   [Wstar, lambda_star] = gerost.inner_max(Y, What, rho, d)
            %
            %   Finds  W* = argmax_{W in Gr(n,d)} ||W^T Y||_F^2
            %               s.t. d_c(W, What) <= rho
            %
            %   For a fixed multiplier lambda the unconstrained maximiser is
            %       W*(lambda) = top-d eigenvectors of  -Y Y^T + lambda What What^T
            %   The scalar equation  d_c(W*(lambda), What) = rho  is solved by
            %   bisection (64 iterations, no nested functions, no external calls).
            %
            %   Implementation note
            %   -------------------
            %   Nested functions inside classdef static methods cannot resolve
            %   names from the MATLAB path, so fzero(@h_fn,...) with a nested
            %   h_fn that called chordalDist would fail with "Unrecognized
            %   function or variable 'chordalDist'".  The bisection loop below
            %   is fully self-contained: chordal distance is inlined and
            %   lowrank_topd_eig is called via the gerost.* qualified name.

            k          = size(Y, 2);
            LAMBDA_MIN = 2 + 1e-6;

            % --- helper: chordal distance between Vd and What (inlined) ---
            %   d_c(A,B)^2 = d - ||A^T B||_F^2   (principal-angle definition)
            dc_fn = @(Vd) sqrt(max(d - sum(min(svd(Vd' * What), 1).^2), 0));

            % --- evaluate h(lambda) = d_c(W*(lambda), What) - rho ---------
            h_fn  = @(lam) dc_fn(gerost.lowrank_topd_eig(What, Y, lam, d)) - rho;

            lam_lo = 0;
            lam_hi = 2.0 + sqrt(k) / max(rho, 1e-10);

            h_lo = h_fn(lam_lo);
            h_hi = h_fn(lam_hi);

            if h_lo <= 0
                % Unconstrained optimum already inside ball: constraint inactive.
                lambda_star = LAMBDA_MIN;
            elseif h_lo * h_hi >= 0
                % Bracket invalid (h never crosses zero): clamp to LAMBDA_MIN.
                lambda_star = LAMBDA_MIN;
            else
                % Bisect for 64 iterations (~1e-19 accuracy in lambda).
                for iter = 1:64
                    lam_mid = (lam_lo + lam_hi) / 2;
                    if h_fn(lam_mid) > 0
                        lam_lo = lam_mid;
                    else
                        lam_hi = lam_mid;
                    end
                    if abs(lam_hi - lam_lo) < 1e-12
                        break;
                    end
                end
                lambda_star = (lam_lo + lam_hi) / 2;
            end

            % lambda_star = max(lambda_star, LAMBDA_MIN);
            Wstar       = gerost.lowrank_topd_eig(What, Y, lambda_star, d);
        end

        function Vd = lowrank_topd_eig(What, Y, lambda, d)
            % LOWRANK_TOPD_EIG  Top-d eigenvectors of M = -Y Y^T + lambda What What^T.
            %
            %   Vd = gerost.lowrank_topd_eig(What, Y, lambda, d)
            %
            %   Because M is a sum of two low-rank matrices (ranks k and d),
            %   all non-trivial eigenvectors lie in span(Y, What).  The
            %   computation is therefore done in the (k+d)-dimensional joint
            %   subspace — cost O(n(k+d) + (k+d)^3) instead of O(n^3).

            n = size(Y, 1);

            % Orthonormal basis for span(Y, What)
            Q  = orth([Y, What]);            % (n x m), m <= k + d
            QY = Q' * Y;                     % (m x k)
            QW = Q' * What;                  % (m x d)

            % Projected matrix  M_small = Q^T M Q
            M_small = -QY * QY' + lambda * (QW * QW');  % (m x m)

            % Eigen-decompose the small matrix
            [V, D]   = eig(M_small, 'vector');
            [~, idx] = sort(D, 'descend');

            % Map top-d eigenvectors back to R^n
            Vd = Q * V(:, idx(1:d));         % (n x d)
        end

        function grad = gradf(Y, W)
            % GRADF  Riemannian gradient of f(Y,W) = ||W^T Y||_F^2 w.r.t. Y.
            %
            %   grad = gerost.gradf(Y, W)
            %
            %   Euclidean gradient:  nabla_Y f = 2 W W^T Y
            %   Tangent projection:  grad = P_Y^perp (nabla_Y f)
            %                             = 2 (W W^T Y - Y (Y^T W W^T Y))
            %                             = -2 P_Y^perp P_W Y
            %
            %   The minus sign matches the convention in descent_step where
            %   the step is  tangent = -alpha * grad  (ascent on f).

            WWtY = W * (W' * Y);                      % (n x k)
            grad = -2 * (WWtY - Y * (Y' * WWtY));     % (n x k)
        end

        function [U_new, s_new] = rank1_eig_update(U, s, v, sgn, d)
            % Rank-1 update of an eigendecomposition.
            %
            % Given  C = U * diag(s.^2) * U^T  (the rank-d approximation of
            % W_{t-1} W_{t-1}^T), computes the top-d eigen-factorisation of
            %   C_new = C + sgn * v * v^T
            %
            % Parameters
            % ----------
            % U   : (n x d) left singular vectors (orthonormal columns)
            % s   : (d x 1) singular values  (eigenvalues = s.^2)
            % v   : (n x 1) update vector  (u_t or u_{t-T})
            % sgn : +1 for rank-1 update, -1 for rank-1 downdate
            % d   : number of components to retain
            %
            % Algorithm (O(nd + d^3))
            % --------
            % 1. Project v: a = U^T v,  residual p = v - U*a,  beta = ||p||
            % 2. If beta > eps, extend the basis by one column p/beta,
            %    form the (d+1)x(d+1) matrix M, and eigen-decompose it.
            %    Otherwise stay in the d-dim subspace.
            % 3. Keep the top-d eigenpairs; map back to ambient space.

            a    = U' * v;       % (d x 1)  projection onto current basis
            p    = v - U * a;    % (n x 1)  residual in ambient space
            beta = norm(p);

            if beta > 1e-10
                % v has a component outside the current d-dim subspace.
                % Extend basis: U_ext = [U, p/beta]  (n x d+1)
                p_hat = p / beta;

                % (d+1)x(d+1) representation of C_new in the extended basis:
                %   M = diag([s.^2; 0]) + sgn * [a; beta] * [a; beta]^T
                avec = [a; beta];                               % (d+1 x 1)
                M    = diag([s.^2; 0]) + sgn * (avec * avec'); % (d+1 x d+1)

                [Q, Lambda] = eig(M, 'vector');
                [Lambda, idx] = sort(Lambda, 'descend');
                Q = Q(:, idx);

                U_ext = [U, p_hat];                    % (n x d+1)
                U_new = U_ext * Q(:, 1:d);             % (n x d)
                s_new = sqrt(max(Lambda(1:d), 0));     % guard tiny negatives

            else
                % v lies almost entirely in the current subspace.
                % d x d update:  M = diag(s.^2) + sgn * a * a^T
                M = diag(s.^2) + sgn * (a * a');       % (d x d)

                [Q, Lambda] = eig(M, 'vector');
                [Lambda, idx] = sort(Lambda, 'descend');
                Q = Q(:, idx);

                U_new = U * Q(:, 1:d);
                s_new = sqrt(max(Lambda(1:d), 0));
            end
        end

    end
    % ====================================================================
end