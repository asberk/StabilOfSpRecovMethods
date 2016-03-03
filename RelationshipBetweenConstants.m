%% Computing optimal sigma and lambda values given a dense grid of tau values
%  This code will solve a sequence of optimization problems depending on a
%  parameter tau and use the results to compute optimal parameter values
%  for other kinds of optimization frameworks. 

%% Set-up
% clear workspace
close all; clear all; clc; 

% Set directory and add path to SPGL1 library (for my system)
cd ~/Documents/MATLAB/StabilOfSpRecovMethods/
addpath ~/Documents/MATLAB/spgl1-1.9/

% Set random number generator parameters for reproduceability
rand('twister', 0); randn('state', 0); % set types of rngs that MATLAB uses


%% Variable and Parameter definitions
% For now, dimension parameters will be constant. 
% Later: might wish to vary k to see how sparsity affects ability of (LS),
% (QP) and (BPDN) to recover original vector x0. 
% Later: may also wish to vary the noise level to determine
% efficacy/robustness of (LS), Basis Pursuit De-Noise (BPDN(BPDN), (QP), too. 

verbose = 20;

% dimensions of the space / the matrix A; sparsity level of x0
m = 50; n = 128; k = 3; 
A = RanMat(n,m,'StdNormal');
x0 = RanSpVec(n,k,'StdNormal');

% noise level(s): set to run nIter-many iterations of nIter-many different
% noise levels
nIter = 1;
epsilon = logspace(0, -2, nIter); 
epsilon = repmat(epsilon, nIter, 1); epsilon = epsilon(:).';
z = rand(m,nIter^2); % generate nIter-many m-dimensional noise vectors z

% generate noisy 'measured data' (each column is the desired m-vector)
b0 = A*x0; 
% use multiple variances, but also repeat each variance nIter times. 
b = bsxfun(@plus, b0,bsxfun(@times, epsilon, z)); 

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
nTau = 101; % inverse density of tau grid. 
tau_vec = linspace(.01, 2*norm(x0), nTau); 

display(sprintf(['\n[Constrained] LASSO (LS_tau)\n',...
    'Given b(j,k) = (A*x0)(j) + epsilon(k)*z(j,k),\n',...
    'and vector tau = (0, ..., 2*TwoNorm(x0)),\nwe solve the minimization problem\n',...
    '\tmin{ TwoNorm(Ax-b) : OneNorm(x) <= tau }\n',...
    'for %d-many values of tau using batchCvxError \nlasso method\n'],nTau));

% define options and run solver
opts = spgSetParms('verbosity', 0);
[lasso_error, sigma_approx] = batchCvxError(A,b,x0, 'lasso', tau_vec, opts, verbose);

display(sprintf(['\nThe results of the method are stored in\n',...
    'lasso_error and sigma_approx, respectively.\n']));

%% Normalized plot of |x_tau - x0|_2 vs. tau
[lasso_optima, idx_lasso_optima] = getFirstMin(tau_vec, lasso_error);
invtaustar = 1./lasso_optima(:,1);

plot((invtaustar*tau_vec).', lasso_error);

%% [Unconstrained] LASSO (QP_lambda)
%
% Given b = A*x0 + epsilon*z, we want to recover x0 by solving the
% minimization problem 
%
%      min{ TwoNorm(Ax-b)^2 + lambda*OneNorm(x) }

display(sprintf(['\n[Unconstrained] LASSO (QP_lambda)\n',...
    'Given b(j,k) = (A*x0)(j) + epsilon(k)*z(j,k),\n',...
    'and the vectors tau = [0, ..., 2*TwoNorm(x0)],\n',...
    'and lambda = lambda* / (1-lambda*),\n',...
    'we solve the minimization problem for QP_lambda.\n']))
% create values for lambda
lambda = logspace(-3,3, 51);
% use the same vector tau
% And compute the solutions for the QP minimization problem,
% where qp_error is the residueal TwoNorm(x_lambda - x0), qp_objective is
% the value of the QP_lambda objective function and tau_lambda(j) is the
% tau that corresponds to lambda(j). 
[qp_error, qp_objective, tau_lambda] = batchCvxError(A,b,x0, 'qp', tau_vec, lambda, opts, verbose);

%% Normalized plot of |x_tau - x0|_2 vs. lambda
[qp_optima, idx_qp_optima] = getFirstMin(lambda, qp_error);
invlamstar = 1./qp_optima(:,1);
plot((invlamstar*lambda).', qp_error, '.');

%% Basis Pursuit De-noise [BPDN] (BP_sigma)
%
% Given b = A*x0 + epsilon*z, we want to recover x0 by solving
% the minimization problem 
%
%      min{ OneNorm(x) : TwoNorm(Ax - b) <= sigma }
% Note that largest possible value for sigma is TwoNorm(b), so we simply 
% take the largest of the norms of the vectors in b. 

nSigma = 101;
sigma_vec = linspace(0, sqrt(max(sum(b.*b))), nSigma);

% define options and run solver 
opts = spgSetParms('verbosity', 0);
[bpdn_error, tau_approx] = batchCvxError(A,b,x0, 'bpdn', sigma_vec, opts, verbose);


%% Plot results
%
%% First plot is for LASSO

figure(1);
h = plot(tau_vec, lasso_error);
xlabel('$\tau$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\|\hat x_\tau - x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 16);
title(strcat('LASSO error vs. OneNorm(x); $\|x_0\|_2 =$ ', num2str(norm(x0))), ...
      'Interpreter', 'latex', 'FontSize', 16);

firstMin_LE = getFirstMin( tau_vec, lasso_error ); 

hold on;
plot(firstMin_LE(:,1), firstMin_LE(:,2), 'k*');
hold off;

% epslegend = cell(1, nIter^2);
% for ni = 1:nIter^2
%     epslegend{ni} = num2str(epsilon(ni));
% end
% legend(epslegend);

%% Second plot is for BPDN

figure(2);
h = plot(sigma_vec, bpdn_error);
xlabel('$\sigma$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\|\hat x_\sigma - x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 16);
title(strcat('BPDN error vs. $\|Ax-b\|_2$; $\|x_0\|_2 =$ ', num2str(norm(x0))), ...
      'Interpreter', 'latex', 'FontSize', 16);

firstMin_BPDN = getFirstMin(sigma_vec, bpdn_error);

hold on;
plot(firstMin_BPDN(:,1), firstMin_BPDN(:,2), 'k*');
hold off;


%% Plot sigma_approx
lambda_star = linspace(0,1, 25).^(1/4);
lambda = lambda_star./(1-lambda_star);

for ni = 1:nIter^2
    [X,Y] = meshgrid(lambda, tau_vec);
    surf(X,Y,reshape(sigma_approx(:, ni, :), size(tau_vec,2), 25));
    title(ni);
    pause;
    plot(tau_vec, reshape(sigma_approx(:,ni,:), size(tau_vec,2),25));
    title(ni);
    pause;
end

%% Use sigma_approx to obtain tau-lambda plot
firstMins = zeros(nIter^2, 25); 
figure;
hold on;
for ni = 1:nIter^2
    for lam = 1:25
        firstMins(ni, lam) = tau_vec(findFirstMin(sigma_approx(:, ni, lam)));
        
    end
    plot(firstMins, lambda);
    xlabel('minimizing tau');
    ylabel('optimal lambda wrt tau');
    title(ni);
    pause;

end
hold off;

%% Computation Round 2
%


%% Save results
filename = strcat('lasso_error_', strrep(datestr(clock), ' ', '_'), '.mat');
lassoError_struct.A = A;
lassoError_struct.b0 = b0;
lassoError_struct.b = b;
lassoError_struct.epsilon = epsilon;
lassoError_struct.k = k;
lassoError_struct.lasso_error = lasso_error;
lassoError_struct.m = m;
lassoError_struct.n = n;
lassoError_struct.tau_vec = tau_vec;
lassoError_struct.x0 = x0;
lassoError_struct.z = z;
lassoError_struct.h = h; % plot

save(strcat('./saves/',filename), '-struct', 'lassoError_struct');

%% Some analysis.
% hold on; 
% abs_min_lasso_error = bsxfun(@eq, lasso_error, min(lasso_error));
% [II, JJ] = find(abs_min_lasso_error > 0 );
% abs_min_error_loc = tau_vec(II);
% stem(abs_min_error_loc, 10*ones(size(abs_min_error_loc)));
