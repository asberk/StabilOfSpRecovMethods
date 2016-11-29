function [x_star, primal_min] = cvx_ucLasso(A, b, lambda)
% UCLASSO computes the solution to the unconstrained convex optimization problem 
% x_star = argmin_x { .5 * || A*x - b ||_2^2 + lambda * ||x||_1
% using the CVX package. 
%
% % % % % % 
% Input:
% ======
%      A: the measurement matrix
%      b: the vector of measurements
% lambda: the parameter controlling trade-off between norm of recovered
%         signal and norm of residual
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
    minimize( .5*norm(A*x - b) + lambda*norm(x, 1) );
cvx_end
        
x_star = x;
primal_min = cvx_optval;

end