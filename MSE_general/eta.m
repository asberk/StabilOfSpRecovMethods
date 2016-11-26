function out = eta(x, theta)
% ETA is the vector-valued soft-thresholding function eta(x;theta) = sign(x)*max(0, |x|-theta)
% 
out = sign(x) .* max(0, abs(x) - theta);

end