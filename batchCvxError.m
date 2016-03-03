function [residual, primal_min, varargout] = batchCvxError(A,b,x0, program, constraint, varargin)
% BATCHCVXERROR computes the error TwoNorm(x_c - x0) between the true data x0 and the solution to the optimization program `program`.
%    A : an m-by-n measurement matrix; typically m << n
%    x0 : the n-dimensional input data
%    b = b0 + z for b0 = A*x0 the result of a linear measurement process,
%        and z a vector modelling the measurement noise.
%    program : any of 'lasso' or 'ls'; 'basis pursuit' or 'bpdn'; 'unconstrained' or 'qp'
%    cvec : the constraint vector - i.e., the `tau`, `sigma` or `lambda` corresponding
%           with ls, bpdn, or qp, respectively
%    varargin: (program=='qp') ? {lambda, opts, verbose} : {opts, verbose}
%    opts : options to be passed to the spg solver
%    verbose : no progress output if non-positive; if positive,
%              outputs according to mod(iterationNumber, verbose) == 0

debug_qp = 1;

doQP = 0;

switch program
    
    case {'lasso', 'ls'}
        
        spg_program = @spg_lasso;
        
    case {'basis pursuit', 'bpdn'}
        
        spg_program = @spg_bpdn;
        
    case {'unconstrained', 'qp'}

        spg_program = @spg_lasso;
        doQP = 1;

end

if doQP
    lambda = varargin{1};
    nlambda = size(lambda,2);
end

if nargin < 6 % then verbose wasn't included
    verbose = 20;
else
    verbose = varargin{2+doQP};
end

if nargin < 5 || isempty(varargin{1})
    opts = spgSetParms('verbosity', 0);
else
    opts = varargin{1+doQP};
end


% residual = TwoNorm(x_c - x0),
% where x_c is the solution the minimization problem spg_program
n = size(x0,1); % dimension of the solution x
nC = size(constraint, 2); % length of constraint vector tau (or sigma)
nInversions = size(b,2); % number of b's for which to look for a solution x

residual = zeros(nC, nInversions); % rows will be plotted as independent variable; columns stratified as individual lines.
primal_min = zeros(nC, nInversions);

if doQP
    qp_objective = zeros(nC, nInversions, nlambda);
    residual_qp = zeros(nlambda, nInversions);
    primal_min_qp = zeros(nlambda, nInversions);
    tau_qp = zeros(nlambda, nInversions);
end

x_c = zeros(n,nC);

for ni = 1:nInversions
    
    display(sprintf('Computing for b(:, %d)...', ni));
    
    for t = 1:nC
        
        x_c(:,t) = spg_program(A,b(:,ni),constraint(t),opts);
        residual(t, ni) = norm(x_c(:,t) - x0);
                
        switch program
        
            case {'ls', 'lasso'}
                
                primal_min(t,ni) = norm(A*x_c(:,t) - b(:,ni));
                
            case {'bpdn', 'basis pursuit'}
                
                primal_min(t,ni) = norm(x_c(:,t), 1);
                
            case {'unconstrained', 'qp'}
                % compute objective function of QP_lambda
                
                % fast:
                sigma_empir = norm(A*x_c(:,t) - b(:,ni));
                tau_empir = norm(x_c, 1);
                qp_objective(t,ni,:) = sigma_empir.^2 + lambda.*tau_empir;
    
                % slow, more general:
                % qp_objective(t, ni, :) = QPApprox(A,b(:,ni),x_c(:,t), lambda);
                
        end
        
    end
        
    if doQP
        % Fix lambda and ni and look over all t to find the one that
        % minimizes the values of the objective function stored in the
        % vector qp_lambda(\cdot, ni, lambda)
        for ell = 1:nlambda

            [qp_min, idx] = getFirstMin(constraint, qp_objective(:, ni, ell));
            primal_min_qp(ell, ni) = qp_min(2);
            tau_qp(ell, ni) = qp_min(1);
            residual_qp(ell, ni) = residual(idx, ni);
            
            if debug_qp
                display(qp_min);
                plot(constraint, qp_objective(:,ni,ell));
                hold on; 
                plot(qp_min(1), qp_min(2), 'r.');
                %hold off;
                
                ylabel('$\|Ax_\tau-b\|_2^2 + \lambda\|x_\tau\|_1$', 'Interpreter', 'latex', 'FontSize', 18);
                xlabel('$\tau$', 'Interpreter', 'latex', 'FontSize', 18);
            end

        end
        if debug_qp
            hold off;
            pause;
        end
    end
        
        
        
end
    
if doQP
    primal_min = primal_min_qp;
    varargout{1} = tau_qp;
    residual = residual_qp;
end

end
