function xhat = SolveAMP ( A, y, maxIter, tol, x ) 
% SOLVEAMP finds a solution xhat to the underdetermined linear system 
% y = A*x + w, w ~ N(0, sigma^2), using the Approximate Message Passing
% algorithm: 
%   x(t+1) = eta( A' * z(t) + x(t); theta(t) )
%     z(t) = y - A*x(t) + 1/delta * z(t-1) * mean( etaprime( A' * z(t-1) + x(t-1); theta(t-1) ) )
% 

if nargin < 3
    maxIter = 500;
end

if nargin < 4
    tol = 1e-6;
end

if nargin < 5
    x = 0; 
end

n = size(A, 1); % rows
N = size(A, 2); % columns

x0 = zeros(N, 1); 
z0 = y;
zhat = z0;

t = 0; 
while t <= maxIter

    t = t + 1; 
    
    gamma = A.' * zhat + x0;
    theta = largestElement(abs(gamma), n); % probably a better way of choosing this. ideally, follow bayati montanari. 
    
    xhat = eta(gamma, theta); 
    x0 = xhat; % set x(t-1) to x(t)
    
    if norm(y - A*xhat)/norm(y) < tol 
        display('convergence reached');
        break;
    end
    
    if x ~= 0 
        sprintf('MSE = %5.3f', mean((xhat-x).^2));
    end
    
    zhat = y - A*xhat + zhat./n * sum( etaprime( gamma, theta ) );
    
    if t>= maxIter
        display('max # of iterations reached.');
        break;
    end
    
    
end




end