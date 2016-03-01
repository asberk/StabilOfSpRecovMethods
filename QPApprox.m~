function [output, lambda] = QPApprox(A,b,x_c, Nlambda) 
% QPApprox approximates the quadratic unconstrained LASSO using the solution of the constrained LASSO
%          In particular, Given the solution x_c = x_tau to 
%                  min { TwoNorm(A*x-b) : OneNorm(x) ? tau } 
%          compute 
%                  output = TwoNorm(A*x-b)^2 + lambda*OneNorm(x)
%          for Nlambda-many values of lambda, where 0 ? lambda ? Infinity
%          is computed from lambda* = linspace(0, 1, Nlambda) via
%          lambda = lambda* / (1 - lambda*), and where 
%                  output = output(tau, lambda)
%          is inherently a function of tau and lambda. The goal is then to 
%          find, for given lambda, (tau*, x_tau*) such that
%                  x_tau* ~ argmin { output(tau, lambda) : x_tau \in x_c }
%    Nlambda : at how many points for lambda do we want to run the method?
%    
%    lambda* : a grid lying in [0,1] that maps to a parameter lambda in [0, infinity] 
%             by lambda = lambda* / (1 - lambda*)

lambda_star = linspace(0,1, Nlambda).^(1/4);
lambda = lambda_star./(1-lambda_star);
sigma_empir = norm(A*x_c - b);
tau_empir = norm(x_c, 1);

output = sigma_empir.^2 + lambda.*tau_empir;


end