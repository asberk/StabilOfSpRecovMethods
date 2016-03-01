function A = RanMat(n,m, dist) 
% RANMAT generates a random m-by-n matrix with orthogonal columns, whose elements have come from distribution dist. 
%    Note that dist is not yet functional, and thus does nothing. 

At = qr(randn(n,m), 0); % random encoding matrix with orthogonal rows
A = At'; % orthogonal columns 

end