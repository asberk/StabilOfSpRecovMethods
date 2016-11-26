function out = PDF_Normal( z ) 
 out = exp(-z.^2/2)./sqrt(2*pi);
end