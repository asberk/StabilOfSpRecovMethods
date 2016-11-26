function out = CDF_Normal(z) 
out = .5 * ( 1 + erf(z./sqrt(2)) );
end