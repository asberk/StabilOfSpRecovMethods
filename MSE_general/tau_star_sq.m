function out = tau_star_sq ( a, g, d, s ) 
% TAU_STAR_SQ computes \tau_*^2 from the Bayati Montanari paper
% given the underdetermined linear system y = Ax + w, w ~ N(0, sigma^2),
% a = alpha = mulitplier for the soft-threshold value (***can be a vector***)
% g = gamma = s/N = proportion of non-zero entries
% d = delta = n/N = aspect ratio of the matrix A
% s = sigma = std(w)

kappa = g/d;

out = s^2 ./ ( 1 - kappa*(1+a.^2) - (2/d)*(1-g)*DMM(a) );


end