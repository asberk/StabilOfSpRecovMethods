function out = etaprime ( x, theta)
% ETAPRIME is the derivative of the soft-thresholding function
%     eta(x; theta) = sign(x) * max(0, abs(x) - theta),
%     which is equal to the indicator function on the region |x| > theta. 
out = (abs(x) > theta); 

end