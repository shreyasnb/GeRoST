function Vd = topd_eig(What, Y, lam, d)
     % % Compute top-d eigenspace of B = lam * What @ What^T - Y @ Y^T
     % % Direct method

     B = lam * (What * What') - (Y * Y');
     [Vd, ~] = eigs(B, d, 'largestabs');

end