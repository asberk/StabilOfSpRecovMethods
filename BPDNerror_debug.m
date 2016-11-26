function [residual, primal_min] = BPDNerror_debug(A,b,x0, sigma)


% residual = TwoNorm(x_c - x0),
% where x_c is the solution the minimization problem spg_program
n = size(x0,1); % dimension of the solution x
len_sigma = size(sigma, 2); % length of constraint vector sigma
nInversions = size(b,2); % number of b's for which to look for a solution x

residual = zeros(len_sigma, nInversions); % rows will be plotted as independent variable; columns stratified as individual lines.
primal_min = zeros(len_sigma, nInversions);

x_sigma = zeros(n, len_sigma);

opts = spgSetParms('verbosity', 0);

for ni = 1:nInversions
    
    display(sprintf('Computing for b(:, %d)...', ni));
    
    for s = 1:len_sigma
        
        % display(constraint(t));
        x_sigma(:,s) = spg_bpdn(A, b(:,ni), sigma(s), opts);
        residual(s, ni) = norm(x_sigma(:,s) - x0);
        primal_min(s,ni) = norm(x_sigma(:,s), 1);
        
    end
        
end
    
end