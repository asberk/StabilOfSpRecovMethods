%% Computing optimal sigma and lambda values given a dense grid of tau values
%  This code will solve a sequence of optimization problems depending on a
%  parameter tau and use the results to compute optimal parameter values
%  for other kinds of optimization frameworks. 

%% Variable and Parameter definitions
% For now, dimension parameters will be constant. 
% Later: might wish to vary k to see how sparsity affects ability of (LS),
% (QP) and (BPDN) to recover original vector x0. 
% Later: may also wish to vary the noise level to determine
% efficacy/robustness of (LS), Basis Pursuit De-Noise (BPDN(BPDN), (QP), too. 

verbose = 20;

% dimensions of the space / the matrix A; sparsity level of x0
m = 50; n = 400; k = 1; 
kappa = 1 - k/n;
A = randn(m,n);
% A = RanMat(n,m,'StdNormal');
x0 = RanSpVec(n,k,'StdNormal');

% noise level(s): set to run nIter-many iterations of nIter-many different
% noise levels
nIter = 4;
epsilon = [logspace(-1,-3, nIter-1),0]; 
epsilon = repmat(epsilon, nIter, 1); epsilon = epsilon(:).';
z = randn(m,nIter^2); % generate nIter-many m-dimensional noise vectors z

% generate noisy 'measured data' (each column is the desired m-vector)
b0 = A*x0; 
% use multiple variances, but also repeat each variance nIter times. 
b = bsxfun(@plus, b0,bsxfun(@times, epsilon, z)); 

noiseVariance = var(bsxfun(@times, z, epsilon));
legendLabels = cell(size(noiseVariance));
for j = 1:length(noiseVariance)
    legendLabels{j} = sprintf('%.2e', noiseVariance(j));
end

display(sprintf(['Create a %d-by-%d matrix A with standard-normal entries,\n',...
    'and a %d-sparse vector x0 with standard-normal entries.\n\n', ...
    'Define vector b0 = A*x0 and create matrix b by adding \n',...
    'epsilon-scaled standard-normal noise matrix epsilon*z.\n',...
    '\t b0(j) = (A*x0)(j);\n\tb(j,k) = b0(j) + epsilon(k)*z(j,k)\n'], m,n,k));

%% [Constrained] LASSO (LS_tau)
%
% Given b = A*x0 + epsilon*z, we want to recover x0 by solving the
% minimization problem 
% 
%      min{ TwoNorm(Ax-b) : OneNorm(x) <= tau }

% generate a dense grid for tau > 0
nTau = 201; % inverse density of tau grid. 
tau_vec = linspace(.01, 2*norm(x0,1), nTau); 

display(sprintf(['\n[Constrained] LASSO (LS_tau)\n',...
    'Given b(j,k) = (A*x0)(j) + epsilon(k)*z(j,k),\n',...
    'and vector tau = (0, ..., 2*TwoNorm(x0)),\nwe solve the minimization problem\n',...
    '    min{ TwoNorm(Ax-b) : OneNorm(x) <= tau }\n',...
    'for %d-many values of tau using batchCvxError \nlasso method\n'],nTau));

% define options and run solver
opts = spgSetParms('verbosity', 0);
[lasso_error, sigma_approx] = batchCvxError(A,b,x0, 'lasso', tau_vec, opts, verbose);

display(sprintf(['\nThe results of the method are stored in\n',...
    'lasso_error and sigma_approx, respectively.\n']));

%% Compute normalized tau's (normalized according to tau_*)
% Note that this therefore does not readily allow one to visualize how
% changing the noise level changes the size of tau_*
[lasso_optima, idx_lasso_optima] = min(lasso_error, [], 1);
invtaustar = 1./tau_vec(idx_lasso_optima);
tauoverstar = tau_vec.'*invtaustar;


%% Basis Pursuit De-noise [BPDN] (BP_sigma)
%
% Given b = A*x0 + epsilon*z, we want to recover x0 by solving
% the minimization problem 
%
%      min{ OneNorm(x) : TwoNorm(Ax - b) <= sigma }
% Note that largest possible value for sigma is TwoNorm(b), so we simply 
% take the largest of the norms of the vectors in b. 


nSigma = 201;

display(sprintf(['\nBasis Pursuit Denoise [BPDN_sigma]\n',...
    'Given b(j,k) = (A*x0)(j) + epsilon(k)*z(j,k),\n',...
    'and vector sigma = (1e-6, ..., 20*epsilon),\n',...
    'we solve the minimization problem\n',...
    '    min{ OneNorm(x) : TwoNorm(Ax-b) <= sigma }\n',...
    'for %d-many values of sigma using batchCvxError \nbpdn method\n'],nSigma));

residuals = bsxfun(@minus, A*x0, b);
%sigma_max = max(100*sqrt(sum(residuals.^2, 1)));
sigma_vec = logspace(-6, log(20*max(epsilon)), nSigma);

% define options and run solver 
opts = spgSetParms('verbosity', 0);
[bpdn_error, tau_approx] = batchCvxError(A,b,x0, 'bpdn', sigma_vec, opts, verbose);

%% Compute normalized sigma's (normalized according to sigma_*)
[bpdn_optima, idx_bpdn_optima] = min(bpdn_error, [], 1);
invsigmastar = 1./sigma_vec(idx_bpdn_optima);
sigmaoverstar = sigma_vec.'*invsigmastar;


%% [Unconstrained] LASSO (QP_lambda)
%
% Given b = A*x0 + epsilon*z, we want to recover x0 by solving the
% minimization problem 
%
%      min{ TwoNorm(Ax-b)^2 + lambda*OneNorm(x) }

nLambda = 201;

display(sprintf(['\n[Unconstrained] LASSO (QP_lambda)\n',...
    'Given b(j,k) = (A*x0)(j) + epsilon(k)*z(j,k),\n',...
    'and the vector lambda = 1e-6, ..., 1e.5,\n',...
    'we solve the minimization problem for QP_lambda,\n',...
    '    min{ .5*TwoNorm(Ax-b) + lambda*OneNorm(x) }\n',...
    'for %d-many value of sigma using batchCvxerror\n',...
    'qp method.\n'], nLambda))

% create values for lambda
lambda_vec = logspace(-6,.5, nLambda);
% use the same vector tau
% And compute the solutions for the QP minimization problem,
% where qp_error is the residueal TwoNorm(x_lambda - x0), qp_objective is
% the value of the QP_lambda objective function and tau_lambda(j) is the
% tau that corresponds to lambda(j). 
[qp_error, qp_objective] = batchCvxError(A,b,x0, 'qp', lambda_vec, [], verbose);


%% Compute normalized lambda's (normalized according to lambda_*)
[qp_optima, idx_qp_optima] = min(qp_error, [], 1);
invlamstar = 1./lambda_vec(idx_qp_optima);
lamoverstar = lambda_vec.'*invlamstar;


%% Save results if desired

if doSave
    filename = strcat('ls-bpdn-qp_error_', strrep(datestr(clock), ' ', '_'), '.mat');
    lassoError_struct.A = A;
    lassoError_struct.b0 = b0;
    lassoError_struct.b = b;
    lassoError_struct.epsilon = epsilon;
    lassoError_struct.k = k;
    lassoError_struct.kappa = kappa;
    lassoError_struct.lasso_error = lasso_error;
    lassoError_struct.bpdn_error = bpdn_error;
    lassoError_struct.qp_error = qp_error;
    lassoError_struct.m = m;
    lassoError_struct.n = n;
    lassoError_struct.tau_vec = tau_vec;
    lassoError_struct.sigma_vec = sigma_vec;
    lassoError_struct.lambda_vec = lambda_vec;
    lassoError_struct.x0 = x0;
    lassoError_struct.z = z;
    lassoError_struct.legendLabels = legendLabels;
    
    
    save(strcat('./saves/',filename), '-struct', 'lassoError_struct');
end

