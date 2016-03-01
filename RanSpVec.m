function x0 = RanSpVec(n,k,dist)
%RANSPVEC generates a random n-dimensional vector with sparsity level k whose elements are distributed according to dist  
%    dist feature has yet to be implemented. 
  
  p = randperm(n); p = p(1:k); % Location of k non-zeros (nnzs) in x
  x0 = zeros(n,1); x0(p) = randn(k,1); % The n-dimensional k-sparse solution x0
end