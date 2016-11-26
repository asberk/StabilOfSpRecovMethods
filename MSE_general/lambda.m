function out = lambda( a, g, d, s ) 
% LAMBDA computes the scaling value for the convex optimization problem \| y - A*x \|_2 ^2 + lambda * \| x \|_1 for a particular value of alpha, which is the multiplier of tau for the soft-thresholding in the AMP algorithm. 
% Given the underdetermined linear system y = Ax + w, w ~ N(0, sigma^2),
% a = alpha = mulitplier for the soft-threshold value (***can be a vector***)
% g = gamma = s/N = proportion of non-zero entries
% d = delta = n/N = aspect ratio of the matrix A
% s = sigma = std(w)
% t = tau_star

tsq = tau_star_sq(a,g,d,s);
tsq(tsq < 0) = nan;
t = sqrt(tsq);
out = a .* t .* ( 1 - g/d - (2/d)*CDF_Normal(-a) ); 

end