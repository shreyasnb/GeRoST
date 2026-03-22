function cd = chordalDist(U, V)
    % % Chordal distance between subspaces spanned by U and V.
    % % 
    % % d_c = sqrt(sum sin^2(theta_i)) for equal dimensions,
    % % d_c = sqrt(|k-d| + sum sin^2(theta_i)) for unequal.
    % % 
    % % Parameters
    % % ----------
    % % U : (n, k) orthonormal basis
    % % V : (n, d) orthonormal basis
    % % 
    % % Returns
    % % -------
    % % float : chordal distance
    
    [~,S,~] = svd(U'*V);
    S = diag(S);
    cos_sq = clip(S.^2, 0, 1);
    sin_sq = 1-cos_sq;
    k = size(U,2);
    d = size(V,2);
    cd = sqrt(abs(k-d) + sum(sin_sq));
end