function out = AnalyticMSE ( a, g, d, s ) 
% ANALYTICMSE computes, analytically the mean-squared error of the soft-threshold approximation to the underdetermined linear system y = A*x + w, w ~ N(0, ?^2).
% delta = n/N is the aspect ratio of A; tau_star_sq is ?_*^2, as defined Bayati-Montanari. 

out = d * (tau_star_sq(a,g,d,s) - s.^2);

end