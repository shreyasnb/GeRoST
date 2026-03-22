function V = gradf(Y, Wstar)
    n = size(Y,1);
    V = -2*(eye(n)-(Y*Y'))*(Wstar*Wstar')*Y;
end