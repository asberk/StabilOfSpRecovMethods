function x_lambda = shrinkage( y, lambda ) 
% SHRINKAGE returns the lambda-soft-thresholded version of y according to 
%    x_lambda = sign(y) * (abs(y) - lambda) * (abs(y) - lambda > 0)

x_lambda = sign(y) .* (abs(y) - lambda) .* (abs(y) - lambda > 0);


end