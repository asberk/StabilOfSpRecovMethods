function [x_star, primal_min] = cvx_bpdn(A, b, sigma)
% UCLASSO computes the solution to the unconstrained convex optimization problem 
% x_star = argmin_x { .5 * || A*x - b ||_2^2 + lambda * ||x||_1
% using the CVX package. 
%
% % % % % % 
% Input:
% ======
%     A: the measurement matrix
%     b: the vector of measurements
% sigma: the parameter constraining the size of the residual
% 
% % % % % %
% Output:
% =======
%     x_star: the recovered signal; the vector returned as the result of
%             the optimization problem.
% primal_min: the value of the objective function evaluated at the optimum
%             vector x_star

% number of columns of A is equal to length of x
n = size(A, 2);

cvx_begin quiet
    variable x(n);
    minimize( norm(x, 1) );
    subject to
        norm(A*x-b) <= sigma;
cvx_end
        
x_star = x;
primal_min = cvx_optval;

end