%%
clear all; close all; clc;

%%
n = 16;
N = n^2; 

Image = zeros(n,n);
[XX, YY] = meshgrid(linspace(-1,1,n));
Image(XX.^2 + YY.^2 < .5) = 1;

x = Image(:);

%% compute the matrix takes the first variation of the data
DD = firstVariation(N);

%% compute pseudo-inverse of the first variation
piDD = (DD.' * DD) \ DD.';

%%
A1 = randn(N./2, N)./sqrt(N/2);
A = A1 * piDD;

%%
y = A1*x;

%% SPGL1
xhat = spg_bp(A, y);

% It really doesn't work (why?) =(
