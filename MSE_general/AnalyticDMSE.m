function out = AnalyticDMSE ( a, g, d, s ) 
% ANALYTIC_DMSE returns the derivative of the analytic expression for the mean squared error (as defined in AnalyticMSE). 
% Given the underdetermined linear system y = Ax + w, w ~ N(0, sigma^2),
% a = alpha = mulitplier for the soft-threshold value (***can be a vector***)
% g = gamma = s/N = proportion of non-zero entries
% d = delta = n/N = aspect ratio of the matrix A
% s = sigma = std(w)

num_stable = 2*d^2*s^2*(g*a - sqrt(2/pi)*(1-g)*exp(-a.^2/2) + (1-g)*a.*erfc(a/sqrt(2)));
denom_stable = ( 2*(1-g)*DMM(a) + g*(s^2+a.^2)-d ).^2;

out = num_stable ./ denom_stable;

end