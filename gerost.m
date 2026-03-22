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
            addParameter(p, 'max_steps', 1000, @isnumeric);  % pre-allocate space
            
            parse(p, varargin{:});
            
            obj.n = n;
            obj.k = k;
            obj.d = d;
            obj.T = T;
            obj.K = p.Results.K;
            obj.rho_param = p.Results.rho;
            obj.alpha_param = p.Results.alpha;
            obj.missing = p.Results.missing;
            obj.fallback = p.Results.fallback;
            
            % Initialize state
            obj.U = [];
            obj.U_history = zeros(n, k, p.Results.max_steps);
            obj.samples = zeros(n, 0);  % empty initially
            obj.masks_full = [];
            obj.window_idx = [];
            obj.t = 0;
            
            % Initialize Manopt Grassmann manifold
            obj.manifold = grassmannfactory(n, k);
        end
        
        function obj = initialize(obj, U0)
            % Set initial subspace estimate
            %
            % Parameters
            % ----------
            % U0 : (n x k) orthonormal basis
            
            obj.U = U0;
        end
        
        function rho = get_rho(obj, sigma_vals, What, Y)
            % Get ball radius for current time step
            %
            % Parameters
            % ----------
            % sigma_vals : vector of singular values
            % What : (n x d) ball center
            % Y : (n x k) current subspace (optional)
            %
            % Returns
            % -------
            % rho : float, ball radius
            
            if isa(obj.rho_param, 'function_handle')
                % Try calling with 4 arguments first, then fall back to 3
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
            % Impute missing entries using current estimate
            %
            % Parameters
            % ----------
            % u : (n,) data vector
            % mask : (n,) boolean array, True = observed
            %
            % Returns
            % -------
            % u_tilde : (n,) imputed vector
            
            if isempty(mask) || all(mask)
                u_tilde = u;
                return;
            end
            
            u_tilde = u;
            observed_idx = mask;
            U_obs = obj.U(observed_idx, :);  % (|Omega| x k)
            
            % Solve least-squares on observed entries
            u_obs = u(observed_idx);
            w_star = U_obs \ u_obs;
            
            % Fill missing entries with projection
            missing_idx = ~mask;
            u_imputed = obj.U * w_star;
            u_tilde(missing_idx) = u_imputed(missing_idx);
        end
        
        function [What, sigma_vals] = build_window_matrix(obj)
            % Build the (imputed) window matrix and compute ball center
            %
            % Returns
            % -------
            % What : (n x d) orthonormal basis (ball center)
            % sigma_vals : singular values of the window
            
            if obj.missing
                cols = [];
                for i = obj.window_idx
                    u = obj.samples(:, i);
                    m = obj.masks_full(:, i);
                    cols = [cols, impute(obj, u, m)];
                end
                W = cols;
            else
                W = obj.samples(:, obj.window_idx);
            end
            
            % SVD to get top-d left singular subspace
            [U_svd, S, ~] = svd(W, 'econ');
            sigma_vals = diag(S);
            What = U_svd(:, 1:obj.d);
        end
        
        function obj = descent_step(obj, u, varargin)
            % Process one data vector and update subspace estimate
            %
            % Parameters
            % ----------
            % u : (n,) data vector
            % mask : (n,) boolean array (optional), True = observed
            %
            % Returns
            % -------
            % obj : updated gerost object with new subspace estimate
            
            p = inputParser;
            addParameter(p, 'mask', [], @(x) isempty(x) || islogical(x));
            parse(p, varargin{:});
            mask = p.Results.mask;
            
            obj.t = obj.t + 1;
            
            % Update full data storage and sliding window indices
            obj.samples(:, obj.t) = u;
            if ~isempty(mask)
                obj.masks_full(:, obj.t) = mask;
            else
                obj.masks_full(:, obj.t) = true(obj.n, 1);
            end
            
            % Update sliding window indices (keep last T indices)
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
            
            % Build window matrix and ball center
            [What, sigma_vals] = build_window_matrix(obj);
            
            % Get ball radius
            rho = get_rho(obj, sigma_vals, What, obj.U);
            rho = clip(rho, 1e-6, sqrt(obj.k) - 1e-6);
            
            % Inner gradient descent loop
            Y = obj.U;
            
            for iter = 1:obj.K
                % Solve inner maximization (low-rank eigendecomposition)
                [Wstar, lambda_star] = inner_max(Y, What, rho, obj.d);
                
                if lambda_star > 2
                    % Constraint active: use paper's gradient
                    % grad_Y f(Y, W*) = -2 P_Y^perp P_{W*} Y
                    grad = gradf(Y, Wstar);
                    delta = lambda_star - 2.0;
                    L = 4.0 * (1.0 + 1.0 / delta) + 4.0 * sqrt(obj.d) / (delta ^ 2);
                    
                    % PL constant: Nhat = (Y^T W*)(W*^T Y)
                    YtW = Y' * Wstar;  % (k x d)
                    Nhat = YtW * YtW';  % (k x k)
                    eigvals_N = eig(Nhat);
                    nu_0 = max(min(eigvals_N), 1e-10);
                    nu = 2.0 * nu_0;
                else
                    % Constraint inactive (lambda* <= 2)
                    if strcmp(obj.fallback, 'data')
                        % GREAT fallback: use data-level gradient
                        % grad = -2 P_Y^perp W W^T Y
                        Wdata = obj.samples(:, obj.window_idx);
                        WtY = Wdata' * Y;  % (T x k)
                        WWtY = Wdata * WtY;  % (n x k)
                        grad = -2.0 * (WWtY - Y * (Y' * WWtY));
                        sig1_sq = sigma_vals(1) ^ 2;
                        L = 4.0 * sig1_sq;
                    else
                        % Subspace fallback: use W* = What (ball center)
                        % grad = -2 P_Y^perp P_{What} Y
                        grad = gradf(Y, What);
                        L = 4.0;
                    end
                    nu = 0.0;
                end
                
                % Compute step size
                if ~isempty(obj.alpha_param)
                    alpha = obj.alpha_param;
                else
                    if nu > 1e-10
                        alpha = min(1.0 / max(L, 1e-8), 1.0 / (2.0 * nu));
                    else
                        alpha = 1.0 / max(L, 1e-8);
                    end
                end
                
                % Retraction step (Grassmann manifold using Manopt)
                tangent = -alpha * grad;
                Y = obj.manifold.exp(Y, tangent);
            end
            
            obj.U = Y;
            obj.U_history(:, :, obj.t) = obj.U;
        end
    end
end