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
%rand('twister', 0); randn('state', 0); % set types of rngs that MATLAB uses
rng('default');
rng(0);


%% Variable and Parameter definitions
% For now, dimension parameters will be constant. 
% Later: might wish to vary k to see how sparsity affects ability of (LS),
% (QP) and (BPDN) to recover original vector x0. 
% Later: may also wish to vary the noise level to determine
% efficacy/robustness of (LS), Basis Pursuit De-Noise (BPDN(BPDN), (QP), too. 

verbose = 20;

% dimensions of the space / the matrix A; sparsity level of x0
m = 50; n = 150; k = 9; 
A = randn(m,n);
% A = RanMat(n,m,'StdNormal');
x0 = RanSpVec(n,k,'StdNormal');

% noise level(s): set to run nIter-many iterations of nIter-many different
% noise levels
nIter = 3;
epsilon = [logspace(-1,-3, nIter-1),0]; 
epsilon = repmat(epsilon, nIter, 1); epsilon = epsilon(:).';
z = randn(m,nIter^2); % generate nIter-many m-dimensional noise vectors z

% generate noisy 'measured data' (each column is the desired m-vector)
b0 = A*x0; 
% use multiple variances, but also repeat each variance nIter times. 
b = bsxfun(@plus, b0,bsxfun(@times, epsilon, z)); 

meanNoise = mean(bsxfun(@times, z, epsilon), 1);
legendLabels = cell(size(meanNoise));
for j = 1:length(meanNoise)
    legendLabels{j} = sprintf('%.2e', meanNoise(j));
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
nTau = 101; % inverse density of tau grid. 
tau_vec = linspace(.01, 2*norm(x0,1), nTau); 

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


%% Basis Pursuit De-noise [BPDN] (BP_sigma)
%
% Given b = A*x0 + epsilon*z, we want to recover x0 by solving
% the minimization problem 
%
%      min{ OneNorm(x) : TwoNorm(Ax - b) <= sigma }
% Note that largest possible value for sigma is TwoNorm(b), so we simply 
% take the largest of the norms of the vectors in b. 

nSigma = 51;
residuals = bsxfun(@minus, A*x0, b);
%sigma_max = max(100*sqrt(sum(residuals.^2, 1)));
sigma_vec = logspace(-5, log(25*max(epsilon)), nSigma);

% define options and run solver 
opts = spgSetParms('verbosity', 0);
[bpdn_error, tau_approx] = batchCvxError(A,b,x0, 'bpdn', sigma_vec, opts, verbose);


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

fig_qp = figure(3);
set(fig_qp, 'Color', [1,1,1]);

% create values for lambda
lambda = logspace(-.5,2, 51);
% use the same vector tau
% And compute the solutions for the QP minimization problem,
% where qp_error is the residueal TwoNorm(x_lambda - x0), qp_objective is
% the value of the QP_lambda objective function and tau_lambda(j) is the
% tau that corresponds to lambda(j). 
[qp_error, qp_objective, tau_lambda] = batchCvxError(A,b,x0, 'qp', tau_vec, lambda, opts, verbose);


%% QP_lambda #2
% Given $\tau$, we can find $x_\tau$ then using the duality relation, we
% can find that $\lambda_\tau = \| A^T r_\tau \|_\infty / \|r_\tau\|_2$

% Note: requires tau_vec, as defined above. 
[qp2_error, qp2_objective, lambda_qp2] = batchCvxError(A,b,x0, 'qp2', tau_vec, opts, verbose);



%% Plots
%% Algorithm Comparison, stratified by noise level. 
%

% We will plot values according to the re-scaled parameters: tau/tau*,
% sigma/sigma* and lambda/lambda* where (.)* denotes the value of (.) at
% which the method achieves its minimum. 
[lasso_optima, idx_lasso_optima] = min(lasso_error, [], 1);
invtaustar = 1./tau_vec(idx_lasso_optima);
tauoverstar = tau_vec.'*invtaustar;

[bpdn_optima, idx_bpdn_optima] = min(bpdn_error, [], 1);
invsigmastar = 1./sigma_vec(idx_bpdn_optima);
sigmaoverstar = sigma_vec.'*invsigmastar;

[qp_optima, idx_qp_optima] = min(qp_error, [], 1);
invlamstar = 1./lambda(idx_qp_optima);
lamoverstar = lambda.'*invlamstar;

% parametrize the values to be passed to sub-plot; we want something that
% resembles a square. 
szb = size(b);
n_cols = round(sqrt(szb(2)));
n_rows = ceil(szb(2)/n_cols);

% an attempt to automatically decide what the upper x-limit should.
xlim_hi = min([max(tauoverstar); max(sigmaoverstar); max(lamoverstar)]);
ylim_hi = 0.2;

figure(90);
for j = 1:szb(2)
    subplot(n_rows, n_cols, j)
    plot(tauoverstar(:, j), lasso_error(:, j));
    hold on;
    plot(sigmaoverstar(:, j), bpdn_error(:,j), 'r');
    plot(lamoverstar(:,j), qp_error(:,j), 'g');
    hold off;
    legend('C-LASSO', 'BPDN', 'UC-LASSO', 'Location', 'NorthEast');
    title(sprintf('Sparsity: %2d; noise: %s', k, legendLabels{j}));
    xlim([0, xlim_hi(j)]);
    ylim([0, .2]);
end
set(gcf, 'Color', [1,1,1]);

%% Plot LS_tau : Normalized plot of |x_tau - x0|_2 vs. tau
fig_ls = figure(1);
plot(tau_vec.' * invtaustar, lasso_error);
xlabel('$\tau/\tau^*$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|x_\tau - x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 18);
legend(legendLabels);

title(strcat('LASSO error vs. $\|x\|_1 / \|x^*\|_1$; $\|x_0\|_2 =$ ', num2str(norm(x0))), ...
    'Interpreter', 'latex', 'FontSize', 16);

% firstMin_LE = getFirstMin( tau_vec, lasso_error );
% hold on;
% plot((invtaustar*firstMin_LE(:,1).').', firstMin_LE(:,2), 'k*');
% hold off;

set(fig_ls, 'Color', [1,1,1]);


%% Plot BPDN: Normalized plot of |x_sigma - x0|_2 vs. sigma
fig_bpdn = figure(2);
subplot(2,1,1);
semilogx(sigma_vec.' * invsigmastar, bpdn_error);
xlabel('$\sigma/\sigma^*$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\|x_\sigma - x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 16);
legend(legendLabels);
title(strcat('BPDN error vs. $\|Ax-b\|_2/\sigma^*$; $\|x_0\|_2 =$ ', num2str(norm(x0))), ...
    'Interpreter', 'latex', 'FontSize', 16);

subplot(2,1,2);
plot(sigma_vec, bpdn_error);
hold on;
plot(sigma_vec(idx_bpdn_optima), bpdn_optima, 'k*');
hold off;
xlabel('$\sigma$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\|x_\sigma - x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 16);
legend(legendLabels);
title(strcat('BPDN error vs. $\|Ax-b\|_2$; $\|x_0\|_2 =$ ', num2str(norm(x0))), ...
    'Interpreter', 'latex', 'FontSize', 14);

% firstMin_BPDN = getFirstMin(sigma_vec, bpdn_error);
% 
% hold on;
% plot(firstMin_BPDN(:,1), firstMin_BPDN(:,2), 'k*');
% hold off;

set(fig_bpdn, 'Color', [1,1,1]);


%%% QUESTIONS
%   
% This normalization seems really weird; it doens't seem like it's
% reasonable to interpret the first by its slope. But if we want something
% unitless against which to compare both BPDN and LS, then maybe this is
% the only approach? 


%% Plot QP: Normalized plot of |x_tau - x0|_2 vs. lambda
figure(fig_qp);
semilogx(lambda.'*invlamstar, qp_error, '.-');
xlabel('$\lambda/\lambda^*$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|x_{\tau(\lambda)} - x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 18);
legend(legendLabels);
title(strcat('QP error vs. $\lambda/\lambda^*$; $\|x_0\|_2 =$ ', num2str(norm(x0))), ...
    'Interpreter', 'latex', 'FontSize', 14);

set(fig_qp, 'Color', [1,1,1]);



% figure;
% plot((lambda).', qp_error, '.-');
% xlabel('$\lambda$', 'Interpreter', 'latex', 'FontSize', 18);
% ylabel('$\|x_{\tau(\lambda)} - x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 18);


%% Plotting QP2_lambda

[qp2_minima, idx_qp2_minima] = min(qp2_error, [], 1);
lambda_qp2_nmz = bsxfun(@rdivide, lambda_qp2, lambda_qp2(idx_qp2_minima));

figure(10);
subplot(211);
plot(lambda_qp2, qp2_error, '.');
legend(legendLabels);
xlabel('$\lambda$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|x_{\lambda(\tau)}\|$', 'Interpreter', 'latex', 'FontSize', 18);
subplot(212);
plot(lambda_qp2_nmz, qp2_error, '.');
legend(legendLabels);
xlabel('$\lambda/\lambda^*$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|x_{\lambda(\tau)}\|$', 'Interpreter', 'latex', 'FontSize', 18);
set(gcf, 'Color', [1,1,1]);

figure(11);
for j = 1:nIter
    group_j = (1:nIter) + nIter*(j-1)
    subplot(nIter, 2, 2*j-1);
    lambda_qp2(:,group_j)
    plot(lambda_qp2(:,group_j), qp2_error(:,group_j), '.');
    legend(legendLabels(group_j));
    xlabel('$\lambda$', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel('$\|x_{\lambda(\tau)}\|$', 'Interpreter', 'latex', 'FontSize', 18);
    subplot(nIter,2,2*j);
    plot(lambda_qp2_nmz(:,group_j), qp2_error(:,group_j), '.');
    legend(legendLabels(group_j));
    xlabel('$\lambda/\lambda^*$', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel('$\|x_{\lambda(\tau)}\|$', 'Interpreter', 'latex', 'FontSize', 18);

end
set(gcf, 'Color', [1,1,1]);

%% If BPDN gives wonky results, we can use CVX BPDN to check (slow)
%% CVX bpdn 
nSigma = 51;
sigma_vec = logspace(-5, log(25*max(epsilon)), nSigma);
nIterates = size(b,2);
cvx_bpdn_error = zeros(nSigma, nIterates);
for j = 1:nIterates
    
    for s = 1:nSigma
    
        cvx_begin quiet
            variable x(n)
            minimize( norm(x, 1) );
            subject to
                norm(A*x - b(:, j)) <= sigma_vec(s);
        cvx_end
        
        cvx_bpdn_error(s,j) = norm(x-x0);
        clear x;
        
    end
    display(sprintf('\nRound %2d of %2d now complete.\n', j, nIterates));

end
%% Compare CVX bpdn and SPGL1 bpdn
subplot(211);
plot(sigma_vec, bpdn_error);
xlabel('$\sigma$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|x-x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 18);
title('SPGL1 BPDN', 'Interpreter', 'latex', 'FontSize', 18);
legend(legendLabels);


[cvxbpdn_optima, idx_cvx] = min(cvx_bpdn_error, [], 1);
sigma_opt = sigma_vec(idx_cvx);
sigmastar = bsxfun(@rdivide, sigma_vec.', sigma_opt);

subplot(212);
plot(sigma_vec, cvx_bpdn_error);
hold on;
plot(sigma_opt, cvxbpdn_optima, '.');
xlabel('$\sigma/\sigma^*$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|x-x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 18);
title('CVX BPDN', 'Interpreter', 'latex', 'FontSize', 18);
legend(legendLabels);



%% Save results
filename = strcat('ls-bpdn-qp_error_', strrep(datestr(clock), ' ', '_'), '.mat');
lassoError_struct.A = A;
lassoError_struct.b0 = b0;
lassoError_struct.b = b;
lassoError_struct.epsilon = epsilon;
lassoError_struct.k = k;
lassoError_struct.lasso_error = lasso_error;
lassoError_struct.bpdn_error = bpdn_error;
lassoError_struct.qp_error = qp_error;
lassoError_struct.m = m;
lassoError_struct.n = n;
lassoError_struct.tau_vec = tau_vec;
lassoError_struct.sigma_vec = sigma_vec;
lassoError_struct.lambda = lambda;
lassoError_struct.x0 = x0;
lassoError_struct.z = z;
lassoError_struct.h = figure(90); % plot
lassoError_struct.legendLabels = legendLabels;


save(strcat('./saves/',filename), '-struct', 'lassoError_struct');

%% Some analysis.
% hold on; 
% abs_min_lasso_error = bsxfun(@eq, lasso_error, min(lasso_error));
% [II, JJ] = find(abs_min_lasso_error > 0 );
% abs_min_error_loc = tau_vec(II);
% stem(abs_min_error_loc, 10*ones(size(abs_min_error_loc)));
