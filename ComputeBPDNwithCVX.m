%% If BPDN gives wonky results, we can use CVX BPDN to check (slow)
%% CVX bpdn 
nSigma = 51;
sigma_vec = logspace(-5, log(25*max(epsilon)), nSigma);
nIterates = size(b,2);
cvx_bpdn_error = zeros(nSigma, nIterates);
for j = 1:nIterates
    
    for s = 1:nSigma
    
        cvx_begin quiet
            variable x(n)
            minimize( norm(x, 1) );
            subject to
                norm(A*x - b(:, j)) <= sigma_vec(s);
        cvx_end
        
        cvx_bpdn_error(s,j) = norm(x-x0);
        clear x;
        
    end
    display(sprintf('\nRound %2d of %2d now complete.\n', j, nIterates));

end
