function out = DMM(a) 
% DMM is the function on the LHS of 1.14 in Bayati-Montanari, given by (1+?^2)*?(-?) - ?*?(?)
out = (1+a.^2).*CDF_Normal(-a) - a.*PDF_Normal(a);
end