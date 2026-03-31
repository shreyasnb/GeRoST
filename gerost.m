classdef gerost
    % GeRoST: Geometrically Robust Online Subspace Tracking
    %
    % This class implements online subspace tracking via constrained gradient
    % descent on the Grassmann manifold with a ball constraint on subspace
    % perturbations.

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

        % Cached SVD of window matrix W_{t-1}
        U_what        % (n x d) cached left singular vectors
        sigma_vals_w  % (d x 1) cached singular values
        lambda_star   % last bisection result (scalar), for diagnostics

        % Grassmann manifold (from Manopt)
        manifold    % Grassmann(n, k) manifold from Manopt
    end

    methods
        function obj = gerost(n, k, d, T, varargin)
            p = inputParser;
            addParameter(p, 'K', 5, @isnumeric);
            addParameter(p, 'rho', 0.1, @(x) isnumeric(x) || isa(x, 'function_handle'));
            addParameter(p, 'alpha', [], @(x) isempty(x) || isnumeric(x));
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
            obj.missing     = false;
            obj.fallback    = p.Results.fallback;

            % Initialize state
            obj.U            = [];
            obj.U_history    = zeros(n, k, p.Results.max_steps);
            obj.samples      = zeros(n, p.Results.max_steps);  
            obj.masks_full   = true(n, p.Results.max_steps);   
            obj.window_idx   = [];
            obj.t            = 0;
            obj.lambda_star   = NaN;

            % Cached SVD
            obj.U_what          = [];
            obj.sigma_vals_w    = [];

            % Initialize Manopt Grassmann manifold
            obj.manifold = grassmannfactory(n, k);
        end

        function obj = initialize(obj, U0)
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

        function [obj, What, sigma_vals] = build_window_matrix(obj)

            window_full = (obj.t > obj.T) && ~isempty(obj.U_what);

            if window_full
                u_out = obj.samples(:, obj.t - obj.T);  
                u_in  = obj.samples(:, obj.t);          

                U = obj.U_what;        
                s = obj.sigma_vals_w;  

                [U, s] = gerost.rank1_eig_update(U, s, u_out, -1, obj.d);
                [U, s] = gerost.rank1_eig_update(U, s, u_in,  +1, obj.d);

                What       = U;
                sigma_vals = s;
            else
                W = obj.samples(:, obj.window_idx);

                [U_svd, S, ~] = svd(W, 'econ');
                sv            = diag(S);
                What          = U_svd(:, 1:obj.d);
                sigma_vals    = sv(1:obj.d);   
            end

            obj.U_what       = What;
            obj.sigma_vals_w = sigma_vals(1:obj.d);
        end

        function obj = descent_step(obj, u, varargin)
            p = inputParser;
            addParameter(p, 'mask', [], @(x) isempty(x) || islogical(x));
            parse(p, varargin{:});
            mask = p.Results.mask;

            obj.t = obj.t + 1;
            obj.samples(:, obj.t) = u;
            if ~isempty(mask)
                obj.masks_full(:, obj.t) = mask;
            end

            if obj.t <= obj.T
                obj.window_idx = 1:obj.t;
            else
                obj.window_idx = (obj.t - obj.T + 1):obj.t;
            end

            if length(obj.window_idx) < obj.d
                obj.U_history(:, :, obj.t) = obj.U;
                obj.lambda_star = NaN;
                return;
            end

            [obj, What, sigma_vals] = build_window_matrix(obj);  

            rho = get_rho(obj, sigma_vals, What, obj.U);
            rho = min(max(rho, 1e-6), sqrt(obj.d) - 1e-6);   

            Y = obj.U;

            for iter = 1:obj.K
                [Wstar, lambda_star] = gerost.inner_max(Y, What, rho, obj.d);

                if lambda_star > 2.0
                    grad  = gerost.gradf(Y, Wstar);
                    
                    % Bound delta to prevent gradient step sizes vanishing entirely near the boundary
                    delta = max(lambda_star - 2.0, 1e-2); 
                    L     = 4.0*(1.0 + 1.0/delta) + 4.0*sqrt(obj.d)/(delta^2);

                    YtW       = Y' * Wstar;
                    Nhat      = YtW * YtW';
                    Nhat      = (Nhat + Nhat') / 2; % Ensure symmetry for robust eig
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
            obj.lambda_star = lambda_star(1);
            obj.U_history(:, :, obj.t) = obj.U;
        end
    end

    % ====================================================================
    methods (Static)

        function [Wstar, lambda_star] = inner_max(Y, What, rho, d)
            k          = size(Y, 2);
            LAMBDA_MIN = 2 + 1e-6;

            dc_fn = @(Vd) sqrt(max(d - sum(min(svd(Vd' * What), 1).^2), 0));
            h_fn  = @(lam) dc_fn(gerost.lowrank_topd_eig(What, Y, lam, d)) - rho;

            % Start bound at LAMBDA_MIN to guarantee strict spectral gap (Lemma 4.1)
            lam_lo = LAMBDA_MIN;
            lam_hi = 2.0 + sqrt(k) / max(rho, 1e-10);

            h_lo = h_fn(lam_lo);
            h_hi = h_fn(lam_hi);

            if h_lo <= 0
                % Unconstrained optimum already inside ball: constraint inactive.
                % Force lambda_star to 2.0 to trigger fallback gradient behavior.
                lambda_star = 2.0; 
            elseif h_hi >= 0
                % Function never crossed zero due to numeric precision; clamp to upper bound.
                lambda_star = lam_hi;
            else
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

            Wstar = gerost.lowrank_topd_eig(What, Y, lambda_star, d);
        end

        function Vd = lowrank_topd_eig(What, Y, lambda, d)
            n = size(Y, 1);

            Q  = orth([Y, What]);            
            QY = Q' * Y;                     
            QW = Q' * What;                  

            M_small = -QY * QY' + lambda * (QW * QW');  
            M_small = (M_small + M_small') / 2; % Important Guard: Prevent complex eigenvalues

            [V, D]   = eig(M_small, 'vector');
            [~, idx] = sort(D, 'descend');

            Vd = Q * V(:, idx(1:d));         
        end

        function grad = gradf(Y, W)
            WWtY = W * (W' * Y);                      
            grad = -2 * (WWtY - Y * (Y' * WWtY));     
        end

        function [U_new, s_new] = rank1_eig_update(U, s, v, sgn, d)
            a    = U' * v;       
            p    = v - U * a;    
            beta = norm(p);

            if beta > 1e-10
                p_hat = p / beta;

                avec = [a; beta];                               
                M    = diag([s.^2; 0]) + sgn * (avec * avec'); 
                M    = (M + M') / 2; % Prevent complex output

                [Q, Lambda] = eig(M, 'vector');
                [Lambda, idx] = sort(Lambda, 'descend');
                Q = Q(:, idx);

                U_ext = [U, p_hat];                    
                U_new = U_ext * Q(:, 1:d);             
                s_new = sqrt(max(Lambda(1:d), 0));     

            else
                M = diag(s.^2) + sgn * (a * a');       
                M = (M + M') / 2; % Prevent complex output

                [Q, Lambda] = eig(M, 'vector');
                [Lambda, idx] = sort(Lambda, 'descend');
                Q = Q(:, idx);

                U_new = U * Q(:, 1:d);
                s_new = sqrt(max(Lambda(1:d), 0));
            end
        end

    end
end