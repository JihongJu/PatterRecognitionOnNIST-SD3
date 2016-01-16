% Initialization
clear ; close all; clc;

% Setup the parameters
input_layer_size  = 625;  % 20x20 Input Images of Digits
hidden_layer_size = 50;   % 50 hidden units
num_iter = 500;          % 500 iterations
num_labels = 10;          % 10 labels, from 0 to 9 

% Load and Visualize Data
% Load the images
fprintf('Loading Data ...\n');
raw_data = prnist(0:9, 1:10:100);
raw_data2 = prnist(0:9, 2:10:100);

figure(1);
show(raw_data);

% x1 = im_features(raw_data,raw_data,'all');
% x2 = im_features(raw_data2,raw_data2,'all');

% x3 = im_mean(raw_data)*datasetm;
% x4 = im_mean(raw_data2)*datasetm;


x1 = im_features(raw_data,raw_data,'all');
x2 = im_features(raw_data2,raw_data2,'all');

% labels
labs = [];
for i = 1:10
    labs = [labs; (i-1)*ones(10,1)];
end
a1 = prdataset(+x1,labs);
a2 = prdataset(+x2,labs);

% split the whole dataset by 80 %
[trData, tstData] = gendat(a1,0.8);
[trData1, tstData1] = gendat(a1,0.8);
[trData2, tstData2] = gendat(a1,0.8);
[trData3, tstData3] = gendat(a1,0.8);
[trData4, tstData4] = gendat(a1,0.8);
[trData5, tstData5] = gendat(a1,0.8);

%Apply PCA into the trainning set
u = scalem([], 'variance') * pcam([],8)*parzenc;
w1 = trData1 * u;
w2 = trData2 * u;
w3 = trData3 * u;
w4 = trData4 * u;
w5 = trData5 * u;
w_all = [w1 w2 w3 w4 w5] * maxc;

% test on tstData
cls = a2*w_all*labeld;
tstLabs = a2.nlab-1;

% calculate the error
corr = cls==tstLabs;
err = sum(corr(:) == 0)/numel(corr);

prwaitbar off;