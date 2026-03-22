%% Utils test script
rng(42);
oldpath = path;
path('./utils',oldpath)

n = 10;
k = 3;
d = 5;

A = randn(n,n);
A = (A+A')/2;
E = 0.1*randn(n,n);
E = (E+E')/2;

[U,~] = eigs(A,k);
[V, ~] = eigs(A+E,d);
rho = sqrt(abs(d-k)+1e-8);


bisect_options = optimset('Display','iter');
[Wstar, lambda_star] = inner_max(U,V,rho, d,bisect_options);

% Perform additional analysis or visualization on the results
disp('W* and lambda*:');
disp(chordalDist(U,Wstar));
disp(lambda_star);
