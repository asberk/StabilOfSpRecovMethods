function [lasso_error, x_tau] = batchLassoError(A,b, x0, tau, opts, verbose)
%BATCHLASSOERROR computes the lasso error given A, the vectors stored in b and the original vector x0 subject to the constraint vector tau
%         A : an m-by-n matrix, typically m << n. 
%        x0 : an n-dimensional vector, typically x0 is k-sparse.
%       tau : a vector whose elements correspond to the constraint imposed on
%             the one-norm of the solution x_tau to the minimization problem. 
%      opts : options passed to the spg solver. 
%   verbose : if 0, no verbosity; positive integer corresponds with:
%             if mod(currentIterationNumber, verbose)==0
%                  display(currentIterationNumber)
%     


% lasso_error = TwoNorm(x_tau - x0), 
% where x_tau is the solution the Lasso minimization problem
nTau = size(tau, 2);
totalIterations = size(b,2);

lasso_error = zeros(nTau, totalIterations); % rows will be plotted as independent variable; columns stratified as individual lines.

for t = 1:nTau
    
    for ni = 1:totalIterations
        
        x_tau = spg_lasso(A,b(:,ni),tau(t),opts);
        lasso_error(t, ni) = norm(x_tau - x0);
        
    end
    
    if verbose > 0
        if mod(t, verbose) == 0
            display(t);
        end
    end
    
end


end