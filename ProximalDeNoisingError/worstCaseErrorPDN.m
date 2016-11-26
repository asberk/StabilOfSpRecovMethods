function wcErrorPDN = worstCaseErrorPDN(x, epsilon, lambda)
% WORSTCASEERRORPDN computes the worst case error a signal x0 with noise
% level epsilon 

if ~ iscolumn(x)
    x = x.';
    
    if ~ iscolumn(x)
        error( ' x must be a vector! ' );
    end
        
end

n = size(x, 1);
t_n = sqrt(2*log(n))*(epsilon + lambda);

wcErrorPDN = sum(min([x.^2, t_n.^2*ones(size(x))].'));

end