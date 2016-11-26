function mse = MSEPDN(x, epsilon, lambda) 
% MSEPDN computes the mean squared error for the proximal de-noising algorithm. 
%    according to the relation 

if ~iscolumn(x)
    x = x.';
    
    if ~ iscolumn(x)
        error('x must be a vector!');
    end
end

sq2pi = sqrt(2*pi); 
n = size(x,1);

mse_vec = zeros(n, 1);

for j = 1:n
    xx = x(j);
    mse_vec(j) = ...
          epsilon.*(xx-lambda).*exp(-.5*(xx+lambda).^2./epsilon.^2)./sq2pi ...
        - epsilon.*(xx+lambda).*exp(-.5*(xx-lambda).^2./epsilon.^2)./sq2pi ...
        + .5*(epsilon.^2 + lambda.^2).*(1 + erf((xx-lambda)./(sqrt(2)*epsilon))) ...
        + .5*xx.^2.*(erf((xx+lambda)./(sqrt(2)*epsilon)) ...
                   - erf((xx-lambda)./(sqrt(2).*epsilon))) ...
        + .5*(epsilon.^2 + lambda.^2).*erfc((xx+lambda)./(sqrt(2).*epsilon));
    
end

mse = sum(mse_vec);

end