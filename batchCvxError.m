function [residual, qp_lambda] = batchCvxError(A,b,x0, program, constraint, varargin)
% BATCHCVXERROR computes the error TwoNorm(x_c - x0) between the true data x0 and the solution to the optimization program `program`.
%    A : an m-by-n measurement matrix; typically m << n
%    x0 : the n-dimensional input data
%    b = b0 + z for b0 = A*x0 the result of a linear measurement process,
%        and z a vector modelling the measurement noise.
%    program : any of 'lasso' or 'ls'; 'basis pursuit' or 'bpdn'; 'unconstrained' or 'qp'
%    cvec : the constraint vector - i.e., the `tau`, `sigma` or `lambda` corresponding
%           with ls, bpdn, or qp, respectively
%    varargin: {opts, verbose}
%    opts : options to be passed to the spg solver
%    verbose : no progress output if non-positive; if positive,
%              outputs according to mod(iterationNumber, verbose) == 0


if nargin < 6 % then verbose wasn't included
    verbose = 20;
else
    verbose = varargin{2};
end

if nargin < 5 || isempty(varargin{1})
    opts = spgSetParms('verbosity', 0);
else
    opts = varargin{1};
end


switch program
    
    case {'lasso', 'ls'}
        
        spg_program = @spg_lasso;
        
    case {'basis pursuit', 'bpdn'}
        
        spg_program = @spg_bpdn;
        
    case {'unconstrained', 'qp'}
        % does nothing for now

end


% residual = TwoNorm(x_tau - x0),
% where x_tau is the solution the minimization problem spg_program
nC = size(constraint, 2);
totalIterations = size(b,2);

residual = zeros(nC, totalIterations); % rows will be plotted as independent variable; columns stratified as individual lines.
primal_approx = zeros(nC, totalIterations);
Nlambda = 25;
qp_lambda = zeros(nC, totalIterations, Nlambda);



for t = 1:nC
    
    for ni = 1:totalIterations
        
        x_c = spg_program(A,b(:,ni),constraint(t),opts);
        residual(t, ni) = norm(x_c - x0);
        
        switch program
            
            case {'ls', 'lasso'}
                
                primal_approx(t,ni) = norm(A*x_c - b(:,ni));
                qp_lambda(t, ni, :) = QPApprox(A,b(:,ni),x_c,Nlambda);
                
            case {'bpdn', 'basis pursuit'}
                
                primal_approx(t,ni) = norm(x_c, 1);
                
            case {'unconstrained', 'qp'}
                
        end
        
    end
    
    if verbose > 0
        if mod(t, verbose) == 0
            display(t);
        end
    end
    
end



end