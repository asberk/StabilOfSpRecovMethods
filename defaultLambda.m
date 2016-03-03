function lambda = defaultLambda(nlambda)
% DEFAULTLAMBDA computes a default sequence of nlambda-many points. 

lamstar = linspace(0, 1, nlambda);
lambda = lamstar ./ (1 - lamstar);

end