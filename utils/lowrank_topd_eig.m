function Vd = lowrank_topd_eig(What, Y, lam, d)
    % % Compute top-d eigenspace of B = lam * What @ What^T - Y @ Y^T
    % % using the low-rank trick with lambda absorbed into What.
    % % 
    % % B = Wtilde @ Wtilde^T - Y @ Y^T, where Wtilde = sqrt(lam) * What.
    % % B has rank <= d+k, so we project to a (d+k) x (d+k) problem.
    % % 
    % % Steps:
    % %     1. Scale: Wtilde = sqrt(lam) * What
    % %     2. QR of [Wtilde | Y] to get orthonormal basis Q for range(B)
    % %     3. Project: AQ = Q^T Wtilde, YQ = Q^T Y
    % %     4. Small B: B_small = AQ AQ^T - YQ YQ^T
    % %     5. Eigendecompose B_small, take top-d eigenvectors
    % %     6. Lift back: Vd = Q @ Vd_small
    % % 
    % % Parameters
    % % ----------
    % % What : (n, d) orthonormal basis
    % % Y : (n, k) orthonormal basis
    % % lam : float, multiplier
    % % d : int, number of top eigenvectors to return
    % % 
    % % Returns
    % % -------
    % % Vd : (n, d) orthonormal top-d eigenvectors of B
    % % 
    
    Wtilde = sqrt(lam) * What;  
    [Q, ~] = qr([Wtilde, Y]);

    AQ = Q'*Wtilde;                             
    YQ = Q'*Y;                        
    B_small = (AQ*AQ') - (YQ*YQ');
    [Vs, ~] = eigs(B_small, d);
    Vd_small = Vs(:, 1:d);
    Vd = Q * Vd_small;

end