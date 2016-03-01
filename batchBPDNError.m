function [bpdn_error, x_sigma] = batchBPDNError(A,b, x0, sigma, opts, verbose)
%BATCHBPDNERROR computes the error TwoNorm(x_sigma - x0) between the initial data x0 and the minimizer of the basis pursuit de-noise program given A, the vectors stored in b and the original vector x0 subject to the constraint vector sigma.
%         A : an m-by-n matrix, typically m << n. 
%        x0 : an n-dimensional vector, typically x0 is k-sparse.
%       sigma : a vector whose elements correspond to the constraint imposed on
%               the two-norm of the difference between the
%               approximated measurement values and the observed
%               measurements 
%      opts : options passed to the spg solver. 
%   verbose : if 0, no verbosity; positive integer corresponds with:
%             if mod(currentIterationNumber, verbose)==0
%                  display(currentIterationNumber)
%     


% bpdn_error = TwoNorm(x_sigma - x0), 
% where x_sigma is the solution the BPDN minimization problem
nSigma = size(sigma, 2);
totalIterations = size(b,2);

bpdn_error = zeros(nSigma, totalIterations); % rows will be plotted as independent variable; columns stratified as individual lines.

for s = 1:nSigma
    
    for ni = 1:totalIterations
        
        x_sigma = spg_lasso(A,b(:,ni),sigma(s),opts);
        bpdn_error(s, ni) = norm(x_sigma - x0);
        
    end
    
    if verbose > 0
        if mod(s, verbose) == 0
            display(s);
        end
    end
    
end


end