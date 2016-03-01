function magVec = vecNorm(A, varargin) 
%VECNORM Computes the norm of each column vector of A and returns the magnitudes as a vector
%   This function uses the Matlab function norm to compute the norm of each
%   column vector of the matrix A, using the standard option P that can be
%   provided to norm. 

iter = size(A, 2);
magVec = zeros(1, iter);
for j = 1:iter
    if nargin > 1
        magVec(j) = norm(A(:, j), varargin{:});
    else
        magVec(j) = norm(A(:,j));
    end
end


end
