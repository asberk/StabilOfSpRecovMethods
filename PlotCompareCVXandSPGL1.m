%% Compare CVX bpdn and SPGL1 bpdn
% requires running
%  - ComputeMSEforLassoParameterVariation
%  - PlotCompareMSEforLassoParameterVariation
%  - ComputeBPDNwithCVX
% first, before this code will work. 

subplot(211);
plot(sigma_vec, bpdn_error);
xlabel('$\sigma$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|x-x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 18);
title('SPGL1 BPDN', 'Interpreter', 'latex', 'FontSize', 18);
legend(legendLabels);


[cvxbpdn_optima, idx_cvx] = min(bpdn_error, [], 1);
sigma_opt = sigma_vec(idx_cvx);
sigmastar = bsxfun(@rdivide, sigma_vec.', sigma_opt);

subplot(212);
plot(sigma_vec, bpdn_error);
hold on;
plot(sigma_opt, cvxbpdn_optima, '.');
xlabel('$\sigma/\sigma^*$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|x-x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 18);
title('CVX BPDN', 'Interpreter', 'latex', 'FontSize', 18);
legend(legendLabels);
