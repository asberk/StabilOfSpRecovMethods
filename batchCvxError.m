function [residual, primal_min, varargout] = batchCvxError(A,b,x0, program, constraint, varargin)
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

debug_qp = 0;
doQP = 0;
doBPDN=0;

switch program
    
    case { 'lasso', 'ls' }
        
        spg_program = @spg_lasso;
        
    case {'basis pursuit', 'bpdn'}
        
        spg_program = @cvx_bpdn;
        doBPDN = 1;
        
    case {'unconstrained', 'qp'}

        spg_program = @cvx_ucLasso;
        doQP = 1;
        
end

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

% residual = TwoNorm(x_c - x0),
% where x_c is the solution the minimization problem spg_program
n = size(x0,1); % dimension of the solution x
nC = size(constraint, 2); % length of constraint vector tau (or sigma)
nInversions = size(b,2); % number of b's for which to look for a solution x

if nInversions == 1
    error('Expected b to be a matrix');
end

residual = zeros(nC, nInversions); % rows will be plotted as independent variable; columns stratified as individual lines.
primal_min = zeros(nC, nInversions);
x_nnz = zeros(nC, nInversions);

x_c = zeros(n,nC);


for ni = 1:nInversions
    
    display(sprintf('Computing for b(:, %d)...', ni));
    
    for t = 1:nC
        
        % display(constraint(t));
        if doQP || doBPDN
            [x_c(:,t), primal_min(t,ni)] = spg_program(A, b(:,ni), constraint(t));
        else % doLS
            x_c(:,t) = spg_program(A, b(:,ni), constraint(t), opts);
            primal_min(t,ni) = norm(A*x_c(:,t) - b(:,ni));
        end
        
        x_nnz(t,ni) = nnz(x_c(:,t));
        residual(t, ni) = norm(x_c(:,t) - x0);
                
    end

end

if nargout > 2
    varargout{1} = x_nnz;
end