function DD = firstVariation(N)

DD = zeros(N*(N-1)/2 + 1, N);
total = 0;
for j = 1:N-1
    for k = j+1:N
        total = total + 1;
        DD(total, j) = 1; DD(total, k) = -1;
    end
end

DD(end,end) = 1;

end
