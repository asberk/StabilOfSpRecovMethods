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
m = 50; n = 150; k = 2; 
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
    legendLabels{j} = num2str(meanNoise(j));
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

%% Normalized plot of |x_tau - x0|_2 vs. tau
[lasso_optima, idx_lasso_optima] = getFirstMin(tau_vec, lasso_error);
invtaustar = 1./lasso_optima(:,1);

fig_ls = figure(1);
plot((invtaustar*tau_vec).', lasso_error);
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

%% Normalized plot of |x_sigma - x0|_2 vs. sigma
[bpdn_optima, idx_bpdn_optima] = min(bpdn_error, [], 1);
invsigmastar = 1./sigma_vec(idx_bpdn_optima).';

fig_bpdn = figure(2);
subplot(2,1,1);
semilogx((invsigmastar*sigma_vec).', bpdn_error);
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

%% CVX bpdn 
nSigma = 101;
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

%% Normalized plot of |x_tau - x0|_2 vs. lambda
[qp_optima, idx_qp_optima] = min(qp_error, [], 1);
lamstar = lambda(idx_qp_optima);
invlamstar = 1./lamstar;
% [qp_optima, idx_qp_optima] = getFirstMin(lambda, qp_error);
% invlamstar = 1./qp_optima(:,1);

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


%% QP_lambda #2
% Given $\tau$, we can find $x_\tau$ then using the duality relation, we
% can find that $\lambda_\tau = \| A^T r_\tau \|_\infty / \|r_\tau\|_2$

% Note: requires tau_vec, as defined above. 
[qp2_error, qp2_objective, lambda_qp2] = batchCvxError(A,b,x0, 'qp2', tau_vec, opts, verbose);


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


%%
%%%%%% BELOW THIS LINE IS OLD CODE THAT I'M KEEPING FOR NOW
 

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

%% Plot them all together
%

tauoverstar = (invtaustar*tau_vec).';
sigmaoverstar = (invsigmastar*sigma_vec).';
lamoverstar = lambda.'*invlamstar;
figure(90);
plot(tauoverstar(:, 3), lasso_error(:, 3));
hold on; 
plot(sigmaoverstar(:, 3), bpdn_error(:,3), 'r');
plot(lamoverstar(:,3), qp_error(:,3), 'g');
hold off;



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
