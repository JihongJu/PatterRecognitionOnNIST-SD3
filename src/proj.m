% Initialization
clear ; close all; clc;

% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 50;   % 50 hidden units
num_iter = 10000;          % 10000 iterations
num_labels = 10;          % 10 labels, from 0 to 9 

% Load and Visualize Data
% Load the images
fprintf('Loading Data ...\n');
raw_data = prnist(0:9, 1:10:1000);

% Preprocess
% preproc = im_box([],0,1)*im_rotate*im_resize([],[20 20])*im_box([],1,0); % aligned
preproc = im_box([],0,1)*im_resize([],[20 20])*im_box([],1,0);  % not aligned
im = raw_data * preproc *datasetm;

% Show
figure(1);
show(im);

% labels
labs = [];
for i = 1:10
    labs = [labs; i*ones(100,1)];
end
a = prdataset(+im,labs);

% split the whole dataset by 50 %
[trData, tstData] = gendat(a,0.5);

% train the bpxnc classifier on trData
[w, h] = bpxnc(trData, hidden_layer_size, num_iter);

% test on tstData
cls = tstData*w*labeld;
tstLabs = tstData.nlab;

% calculate the error
corr = cls==tstLabs;
err = sum(corr(:) == 0)/numel(corr);


prwaitbar off;