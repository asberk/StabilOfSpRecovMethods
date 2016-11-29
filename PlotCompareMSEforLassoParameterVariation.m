%% Plots



%% Plot Lasso error as a function of tau
% Want to use lasso_error and tau_vec
% Create a custom color matrix according to noise level
cMat_1 = hsv(nIter);
cMat_idx = repmat(1:nIter, [nIter,1]);
cMat = cMat_1(cMat_idx(:).', :);

figure(80); hold off;
subplot(311);
CL_p = semilogy(tauoverstar, lasso_error, '.-');
for j = 1:size(CL_p)
    set(CL_p(j), 'Color', cMat(j,:));
end
title(['Constrained Lasso; $\kappa = ', sprintf('%.3g', kappa), '$'], ...
    'Interpreter', 'latex', 'FontSize', 16);
xlabel('$\tau / \tau_*$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\|x^\sharp - x\|_2$', 'FontSize', 16, 'Interpreter', 'latex');
xlim([0,2]);
ylim([0,1]);


subplot(312);
BP_p = semilogy(sigmaoverstar, bpdn_error, '.-');
for j = 1:size(BP_p)
    set(BP_p(j), 'Color', cMat(j,:));
end
title(['Basis Pursuit Denoise; $\kappa = ', sprintf('%.3g', kappa), '$'], ...
    'Interpreter', 'latex', 'FontSize', 16);
xlabel('$\sigma/ \sigma_*$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\|x^\sharp - x\|_2$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0,2]);
ylim([0,1]);
columnlegend(2,legendLabels);


subplot(313);
UC_p = semilogy(lamoverstar, qp_error, '.-');
for j = 1:size(UC_p)
    set(UC_p(j), 'Color', cMat(j,:));
end
title(['Unconstrained Lasso; $\kappa = ', sprintf('%.3g', kappa), '$'], ...
    'Interpreter', 'latex', 'FontSize', 16);
xlabel('$\lambda/\lambda_*$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\|x^\sharp - x\|_2$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0,2]);
ylim([0,1]);
columnlegend(2,legendLabels);


set(gcf, 'Color', [1,1,1]);

%% Plots for algorithm comparison, stratified by noise level. 


% parametrize the values to be passed to sub-plot; we want something that
% resembles a square. 
szb = size(b);
n_cols = round(sqrt(szb(2)));
n_rows = ceil(szb(2)/n_cols);

% an attempt to automatically decide what the upper x-limit should be.
xlim_hi = min([max(tauoverstar); max(sigmaoverstar); max(lamoverstar)]);
ylim_hi = 0.2;

figure(90); clf;
for j = 1:szb(2)
    clf;
    set(gcf, 'Color', [1,1,1]);
    %subplot(n_rows, n_cols, j)
    h = semilogy(tauoverstar(:, j), lasso_error(:, j), 'b.-');
    hold on;
    semilogy(sigmaoverstar(:, j), bpdn_error(:,j), 'r.-');
    semilogy(lamoverstar(:,j), qp_error(:,j), 'g.-');
    hold off;
    %if j == round(.75*szb(2))
    legend('C-LASSO', 'BPDN', 'UC-LASSO', 'Location', 'NorthEast');
    %end
    title(sprintf('Sparsity: %.2f; noise: %s', kappa, legendLabels{j}), 'Interpreter', 'latex', 'FontSize', 20);
    xlim([0, xlim_hi(j)]);
    ylim([0, .2]);
    xlabel('Normalized Convex Parameter', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel('$\log_{10}\,(MSE)$','Interpreter', 'latex', 'FontSize', 18);
    hparent = get(h, 'Parent');
    set(hparent, 'yticklabel', log10(get(hparent, 'ytick')));
    set(hparent, 'FontSize', 16);
    pause;
end


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
figure();
semilogx(lambda_vec.'*invlamstar, qp_error, '.-');
xlabel('$\lambda/\lambda^*$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|x_{\tau(\lambda)} - x_0\|_2$', 'Interpreter', 'latex', 'FontSize', 18);
legend(legendLabels);
title(strcat('QP error vs. $\lambda/\lambda^*$; $\|x_0\|_2 =$ ', num2str(norm(x0))), ...
    'Interpreter', 'latex', 'FontSize', 14);

set(fig_qp, 'Color', [1,1,1]);
