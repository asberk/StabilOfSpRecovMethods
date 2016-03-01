function A = RandomMatrix(n,m, dist) 
%RANDOMMATRIX generates a random matrix of size m-by-n whose elements are distributed according to dist. 
%   Note: presently, dist has yet to be implemented; elements
%   of A are normally-distributed. Moreover, columns of A are
%   orthogonal. 
  
  [At, Rtmp] = qr(randn(n,m),0); % random encoding matrix with orthogonal rows
  A = At'; % orthogonal columns instead of rows. 
  
end