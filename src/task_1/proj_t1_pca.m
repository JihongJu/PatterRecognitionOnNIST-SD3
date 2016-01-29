%% Initialise workspace
clear ; close all; clc;
% N = prmemory(2^26);
%% Setup the parameters
% Parzen
h = 0.5;
% KNN
k = 3;
% ANN
input_layer_size  = 625;  % 25x25 Input Images of Digits
hidden_layer_size = [50 50 50];   % one layer, each with 50 hidden units
num_iter = 500;          % 500 iterations
num_labels = 10;          % 10 labels, from 0 to 9
%% Load data
fprintf('Loading Data ...\n');
raw_data = prnist(0:9, 1:2:1000);
%% Preprocess
a = my_rep(raw_data);
%% split the whole dataset by 80 %
[trData, tstData] = gendat(a,0.8);

%% Classifier_Non-parametric
% Parzen classifier
% u = scalem([], 'variance') * pcam([],0.6) * parzenc([], h);
% KNN classifier
% u = scalem([], 'variance') * pcam([],0.6) * knnc([], k);
%% Classifier_Parametric
% Fisher classifier
% u = scalem([], 'variance') * pcam([],0.6) * fisherc;
% Quadratic classifier
% u = scalem([], 'variance') * pcam([],0.6) * qdc;
%% Classifier_Advanced
% ANN
%[u_bpxnc, h] = bpxnc([], hidden_layer_size, num_iter);
% u = scalem([], 'variance') * pcam([],0.6) * u_bpxnc;
% Support vector classifier, proxm denotes the choose of kernel
% u = scalem([], 'variance') * pcam([],0.6) * svc([], proxm('r'));
% Random forest classifier
% u = scalem([], 'variance') * pcam([],0.6) * randomforestc;
%% Cross-Validation
e1 = prcrossval(trData,u,10,1);
%% Train Classifier
w = trData * u;
%% Evaluation
e2 = nist_eval('my_rep',w,100);

prwaitbar off;