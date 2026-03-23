function H = construct_hankel(W,L)
% CONSTRUCT_HANKEL  Constructs a Hankel matrix from time-series data
%
%   inputs:
%   - W times-series data in matrix. The columns contain the time 
%   instances of the signal. Each row is one component of the signal for 
%   the whole time horizont
%   - L horizon length (integer)
%
%   output: Hankel matrix of horizon L

% Extract signal size
q = size(W,1);

% Extract dataset size
D = size(W,2);

% Determine number of columns
col_nr = D-L+1;

% Initialize Hankel matrix
H = zeros(q*L,col_nr);

% Fill each column
for i = 1:col_nr
    H(:,i) = reshape(W(:,i:i+L-1),[1,q*L]);
end
end
