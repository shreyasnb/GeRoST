classdef great
    % GREAT: Grassmannian Recursive Estimation and Tracking
    %
    % Non-robust baseline algorithm: min_Y ||P_Y^perp W_t||_F^2
    %
    % This is a simplified version of GeRoST without the ball constraint,
    % serving as a baseline for comparison.
    %
    % Parameters
    % ----------
    % n : int, ambient dimension
    % k : int, subspace dimension
    % T : int, window length
    % K : int, number of gradient descent iterations per time step (default: 5)
    % alpha : float or [], step-size. If [], uses 1/L (default: [])
    
    properties
        n           % ambient dimension
        k           % subspace dimension
        T           % window length
        K           % gradient descent iterations per step
        alpha_param % step-size (fixed or empty for adaptive)
        
        % State
        U           % current subspace estimate (n x k)
        U_history   % full sequence of subspace estimates (n x k x num_steps)
        samples     % full data matrix (n x num_collected)
        window_idx  % indices for sliding window
        t           % current time step
        
        % Grassmann manifold (from Manopt)
        manifold    % Grassmann(n, k) manifold from Manopt
    end
    
    methods
        function obj = great(n, k, T, varargin)
            % Initialize GREAT object
            %
            % Usage: obj = great(n, k, T, 'K', 5, 'alpha', [])
            
            p = inputParser;
            addParameter(p, 'K', 5, @isnumeric);
            addParameter(p, 'alpha', [], @(x) isempty(x) || isnumeric(x));
            addParameter(p, 'max_steps', 1000, @isnumeric);  % pre-allocate space
            
            parse(p, varargin{:});
            
            obj.n = n;
            obj.k = k;
            obj.T = T;
            obj.K = p.Results.K;
            obj.alpha_param = p.Results.alpha;
            
            % Initialize state
            obj.U = [];
            obj.U_history = zeros(n, k, p.Results.max_steps);
            obj.samples = zeros(n, 0);  % empty initially
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
        
        function obj = descent_step(obj, u, varargin)
            % Process one data vector and update subspace estimate
            %
            % Parameters
            % ----------
            % u : (n,) data vector
            % mask : (n,) boolean array (optional, ignored for GREAT)
            %
            % Returns
            % -------
            % obj : updated great object with new subspace estimate and stored history
            
            p = inputParser;
            addParameter(p, 'mask', [], @(x) isempty(x) || islogical(x));
            parse(p, varargin{:});
            
            obj.t = obj.t + 1;
            
            % Append new sample to the full data matrix
            obj.samples = [obj.samples, u(:)];
            
            % Update sliding window indices (last T samples)
            num_samples = size(obj.samples, 2);
            if num_samples <= obj.T
                obj.window_idx = 1:num_samples;
            else
                obj.window_idx = (num_samples - obj.T + 1):num_samples;
            end
            
            % Need at least k samples to form the window matrix
            if length(obj.window_idx) < obj.k
                return;
            end
            
            % Build window matrix from current sliding window
            W = obj.samples(:, obj.window_idx);  % (n x T) matrix
            
            % Compute singular value for step size
            [~, S, ~] = svd(W, 'econ');
            svals = diag(S);
            sig1_sq = svals(1) ^ 2;
            
            % Inner gradient descent loop
            Y = obj.U;
            
            for iter = 1:obj.K
                % Use gradf utility function: grad = -2 * P_Y^perp * Wstar * Wstar' * Y
                % For GREAT, Wstar = W (no ball constraint)
                grad = gradf(Y, W);
                
                % Compute step size
                L = 4.0 * sig1_sq;
                if ~isempty(obj.alpha_param)
                    alpha = obj.alpha_param;
                else
                    alpha = 1.0 / max(L, 1e-8);
                end
                
                % Exponential map step (no negative sign, gradf already includes it)
                Y = obj.manifold.exp(Y, -alpha * grad);
            end
            
            obj.U = Y;
            obj.U_history(:, :, obj.t) = Y;
        end
    end
end
