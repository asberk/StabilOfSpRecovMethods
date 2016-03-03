function qp_error = batchQPError(A,b,x0, tau, lambda, opts, verbose)
% BATCHQPERROR uses a dense tau-grid to approximate the solution the unconstrained-LASSO QP-lambda for the values in the vector lambda.
%


% Format inputs
if ~isrow(tau)
    tau = tau.';
    if ~isrow(tau)
        error('QPError:VectorFormatIssue', 'tau must be a row vector, but appears to be a matrix');
    end
end

if ~iscolumn(lambda)
    lambda = lambda.';
    if ~iscolumn(lambda)
        error('QPError:VectorFormatIssue', 'lambda must be a column vector, but appears to be a matrix');
    end
end

% Define necessary settings
n_tau = size(tau, 2);
n_lambda = size(lambda,1);
n = size(x0, 1);
        
total_iterations = size(b,2);

qp_objective = zeros(n_lambda, n_tau);
x_tau = zeros(n, n_tau);

% Solve

x_tau = spg_lasso(A, b, tau, opts);


end