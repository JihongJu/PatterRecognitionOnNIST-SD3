%% Initialise workspace
clear ; close all; clc;

%% Setup the parameters
% ANN
input_layer_size  = 625;  % 20x20 Input Images of Digits
hidden_layer_size = [50 50 50];   % one layer, each with 50 hidden units
num_iter = 500;          % 500 iterations
num_labels = 10;          % 10 labels, from 0 to 9

%% Load data
fprintf('Loading Data ...\n');
raw_data = prnist(0:9, 1:2:1000);

%% Preprocess
a = my_rep(raw_data);

%% Split the whole dataset by 80 %
[trData, tstData] = gendat(a,0.8);

%% Classifiers
% ANN
% [u_bpxnc, h] = bpxnc([], hidden_layer_size, num_iter);
% u = scalem([], 'variance') * pcam([],0.6) * u_bpxnc;
% KNN
% u = scalem([], 'variance') * pcam([],0.6) * knnc;
% parzenc with PCA
u = scalem([], 'variance') * pcam([],0.6) * parzenc;
% qdc
% fisherc
% u = scalem([], 'variance') * pcam([],0.6) * fisherc;
% RandomForest
% u = scalem([], 'variance') * pcam([],0.6) * randomforestc;

%% Cross-Validation
e1 = prcrossval(trData,u,10,1);

%% Training
% simple train
% w = trData * u;

% split, train and combine
% trData1
for i = 1:10
    objects{i,1} = 101:400;
end
trData1 = seldat(trData,[],[],objects);
% trData2
for i = 1:10
    objects{i,1} = cat(2,1:100,201:400);
end
trData2 = seldat(trData,[],[],objects);
% trData3
for i = 1:10
    objects{i,1} = cat(2,1:200,301:400);
end
trData3 = seldat(trData,[],[],objects);
% trData4
for i = 1:10
    objects{i,1} = 1:300;
end
trData4 = seldat(trData,[],[],objects);

% train
w1 = trData1 * u;
w2 = trData2 * u;
w3 = trData3 * u;
w4 = trData4 * u;

% combine
w = [w1 w2 w3 w4] * maxc;

%% Testing
cls = tstData * w * labeld;


%% Calculate the error
tstLabs = tstData.nlab - 1;
corr = cls==tstLabs;
err = sum(corr(:) == 0)/numel(corr);

prwaitbar off;
