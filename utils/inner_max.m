function [Wstar, lambda_star] = inner_max(Y, What, rho, d)
    k = size(Y,2);
    LAMBDA_MIN = 2 + 1e-6;

    lam_lo = 0;
    lam_hi = 2.0 + sqrt(k) / max(rho, 1e-10);

    h_lo = h(lam_lo);
    h_hi = h(lam_hi);

    if (sign(h_lo*h_hi)<0)
        [lambda_star, fval, ~, output] = fzero(@(l) h(l), [lam_lo, lam_hi]);
        % disp('Function value');
        % disp(fval);
        % disp('Output');
        % disp(output);
    else
        lambda_star = LAMBDA_MIN;
    end

    lambda_star = max(lambda_star, LAMBDA_MIN);
    Wstar = lowrank_topd_eig(What, Y, lambda_star, d);

    function dc = h(lam)
        Vd = lowrank_topd_eig(What, Y, lam, d);
        dc = chordalDist(Vd, What) - rho;
    end

end