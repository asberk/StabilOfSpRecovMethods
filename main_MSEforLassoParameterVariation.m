%% Set-up
% clear workspace
close all; clear all; clc; 

% Set directory and add path to SPGL1 library (for my system)
cd ~/Documents/MATLAB/StabilOfSpRecovMethods/
addpath ~/Documents/MATLAB/spgl1-1.9/

% Set random number generator parameters for reproduceability
%rand('twister', 0); randn('state', 0); % set types of rngs that MATLAB uses
rng('default');
rng(0);

%%
doSave = 1;
ComputeMSEforLassoParameterVariation

%%
PlotCompareMSEforLassoParameterVariation

%% No need to run these:
% ComputeBPDNwithCVX
% PlotCompareCVXandSPGL1