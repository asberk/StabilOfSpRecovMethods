function idxvec = batchFindFirstMin(mat)
% BATCHFINDFIRSTMIN iterates over a matrix, returning the result of findFirstMin for each column vector 
%    
  sz_mat = size(mat,2);
  idxvec = zeros(1, sz_mat);

  for k = 1:sz_mat
    idxvec(k) = findFirstMin( mat(:,k));
  end  
  
  end