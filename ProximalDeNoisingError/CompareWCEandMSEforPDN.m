%% Clear workspace
clear all; close all; clc;

%% Parameter set-up
n = 128; % dimension of signal x
m = n; % dimension of measurement vector y
k = 4; % sparsity

L = 101; % length of lambda vector

x0 = zeros(n, 1);
I = randi(n, [k,1]);
x0_I = randn(k,1);
x0(I) = x0_I;

epsvec = [0, .01, .1, 1]; % vector of noise scalings
lamvec = linspace(0, 5, L); % soft-threshold parameter vector 

szeps = size(epsvec);

niter = 100;


%% Theoretical Computation of MSE and WHP WCE
% In this section, we calculate and plot the worst-case error value that
% holds with high probability for a signal x0 with the model 
% y = x0 + epsilon*z where is standard normal

wce_th = zeros( szeps(2), L );
mse_th = zeros( szeps(2), L );

for k = 1:szeps(2)
    for ell = 1:L
        
        wce_th(k,ell) = worstCaseErrorPDN(x0, epsvec(k), lamvec(ell));
        mse_th(k,ell) = MSEPDN(x0, epsvec(k), lamvec(ell));
        
    end
end



%% Empirical validation of WCE and MSE

y_mat = zeros(m, niter, szeps(2)); % each column is a measurement vector
r_mat = zeros( niter, szeps(2), L ); % residual / error

for j = 1:niter
    
    for k = 1:szeps(2)
        y_mat(:,j, k) = x0 + epsvec(k)*randn(m,1);

        for ell = 1:L
            r_mat(j,k,ell) = norm(shrinkage(y_mat(:,j,k), lamvec(ell)) - x0, 2);
            
        end
    end
end

wce_empir = reshape(max(r_mat, [], 1), [szeps(2), L]);
mse_empir = reshape(mean(r_mat, 1), [szeps(2), L]);

%% Plotting results
for j = 1:szeps(2)
    plot(lamvec, mse_th(j,:));
    hold on;
    plot(lamvec, wce_th(j,:), 'r');
    plot(lamvec, mse_empir(j,:), 'b-.');
    plot(lamvec, wce_empir(j,:), 'r-.');
    hold off;
    set(gcf, 'Color', [1,1,1]);
    pause;
end
