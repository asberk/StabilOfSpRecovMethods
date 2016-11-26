function out = DMSE_Numerator(a,g)
% DMSE_NUMERATOR is (roughly) the numerator of the Analytic expression for the derivative of the MSE map with respect to alpha. To find the optimal MSE w/r/t alpha, one finds the zero of its derivative, which is equivalent to finding the zero of this map.
% See AnalyticDMSE for more information. 
% Depends on alpha and gamma only. 

%out = sqrt(2/pi)*(g-1)*exp(-a.^2/2) + g*a + (1-g)*a.*erfc(a/sqrt(2));
out = g*a - 2*(1-g)*PDF_Normal(a) + 2*(1-g)*a.*CDF_Normal(-a);


end