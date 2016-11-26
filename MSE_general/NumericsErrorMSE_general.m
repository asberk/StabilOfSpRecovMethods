%% Assumptions
%
% -- Model --
%
% y = Ax + w, w ~ N(0, sigma^2)
% xhat = eta(y, alpha*tau) where alpha and tau are defined below.
%


%% Clear workspace
close all; clear all; clc;
cd('~/Documents/MATLAB/StabilOfSpRecovMethods/MSE_general/');

%% Parameters and Set-Up
xData = 'SparseLargePositiveTwoVal';

n = 50;
N = 1000;
s = 3;
sigma = .1;

gamma = s./N;
delta = n./N;

% I think that A has to be normalized.
A = randn(n, N)./sqrt(n) ;
w = randn(n, 1) * sigma;


switch xData
    case 'SparseLargePositiveTwoVal'
        I = randi(N, s,1);
        x_I = 10;
        x = zeros(N, 1);
        x(I) = x_I;
    otherwise
        xData = 'SparseLargePositiveTwoVal';
        I = randi(N, s,1);
        x_I = 10;
        x = zeros(N, 1);
        x(I) = x_I;
end

xhat = SolveAMP(A, A*x + w, 1000, 1e-4, x);

display(sprintf('Mean Square Error: %5.4e\n', mean((xhat-x).^2)));

%% Analytic Computations

%% Finding ? minimizer of MSE

% Parameter set-up
gamma = .25; % "relative complexity of signal"
delta = .71; % "aspect ratio of matrix"
sigma = .32; % standard deviation of noise; ?^2 = .1024

% Method to compute the minimizing value alpha for the MSE
dMSE_numerator = @(a) DMSE_Numerator(a, gamma);
[alpha_star, MSE_star, exitflag] = fzero(dMSE_numerator,0);
if exitflag < 0
    warning('something went wrong with the minimization; check ''exitflag'' for more details');
end
% calculate corresponding value of lambda and MSE
lambda_star = lambda(alpha_star, gamma, delta, sigma);
MSE_star = AnalyticMSE(alpha_star, gamma, delta, sigma);

%% Visualizing MSE as a function of ?
% Compute ? and MSE given alpha. 
alpha_vec = linspace(0, 20, 1001); 
lambda_vec = lambda(alpha_vec, gamma, delta, sigma); 
MSE_vec = AnalyticMSE(alpha_vec, gamma, delta, sigma); 

% Output
display(alpha_star);
display(lambda_star);
display(MSE_star);
figure(1);
plot(lambda_vec, MSE_vec, '.-');
title('$\mathrm{MSE}_\lambda$', 'Interpreter', 'latex', 'FontSize', 18);
xlabel('$\lambda$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\mathrm{MSE}_{~\lambda}$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([-.1,1.2]);
ylim([-.1, 5]);

%% 
% Now how do I take this plot above and get a corresponding plot for the
% Constrained LASSO and for Basis Pursuit? 
% The existing map relies on there being data. 




%% Test 1
% we square these because we care less about what happens near 1, and more
% about what happens around zero.
gamma_vec = linspace(0,1,51).^2;
delta_vec = linspace(0,1,51).^2;
sigma_vec = linspace(0, 1, 51).^2;

alpha_res = zeros(51,1,1);
lambda_res = zeros(51,51,51);
MSE_res = zeros(51,51,51);

for gg = 1:length(gamma_vec)
    dMSE_num = @(a) DMSE_Numerator(a,gamma_vec(gg));
    alpha_res(gg) = fzero(dMSE_numerator, 0);
    display(gg);
    for dd = 1:length(delta_vec)
        for ss = 1:length(sigma_vec)
            MSE_res(gg, dd, ss) = AnalyticMSE(alpha_res(gg), gamma_vec(gg), delta_vec(dd), sigma_vec(ss));
            lambda_res(gg,dd,ss) = lambda(alpha_res(gg), gamma_vec(gg), delta_vec(dd), sigma_vec(ss));
        end
    end
end

%% Results 1

% ddidx = 81; % delta = 0.64
ssidx = 48; % sigma = 0.2209
for ddidx = 1:2:51
   
    figure(2);
    subplot(311); plot(gamma_vec, lambda_res(:, ddidx, ssidx), '.-');
    title('$\lambda_*(\gamma);~\gamma\in(0,1)$', 'Interpreter', 'latex', 'FontSize', 18);
    xlabel('$\gamma$', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('$\lambda_*(\gamma)$', 'Interpreter', 'latex', 'FontSize', 16);

    subplot(312); plot(gamma_vec, MSE_res(:, ddidx, ssidx), '.-');
    title('$\mathrm{MSE}_\lambda(\gamma);~\gamma\in(0,1)$', 'Interpreter', 'latex', 'FontSize', 18);
    xlabel('$\gamma$', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('$\mathrm{MSE}_\lambda(\gamma)$', 'Interpreter', 'latex', 'FontSize', 16);

    subplot(313); plot(lambda_res(:, ddidx, ssidx), MSE_res(:,ddidx,ssidx),'.-');
    title('$\mathrm{MSE}(\lambda_*); \lambda_* = \lambda_*(\gamma);~\gamma\in(0,1)$', 'Interpreter', 'latex', 'FontSize', 18);
    xlabel('$\lambda_*(\gamma)$', 'Interpreter', 'latex', 'FontSize', 16);
    ylabel('$\mathrm{MSE}_{~\lambda_*}$', 'Interpreter', 'latex', 'FontSize', 16);
    
    pause;
end