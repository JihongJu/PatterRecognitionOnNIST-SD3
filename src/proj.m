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
raw_data = prnist(0:9, 1:2:1000);
raw_data2 = prnist(0:9, 2:2:1000);

% Preprocess
% preproc = im_box([],0,1)*im_rotate*im_resize([],[20 20])*im_box([],1,0); % aligned
preproc = im_box([],0,1)*im_resize([],[25 25])*im_box([],1,0);  % not aligned
im = raw_data * preproc *datasetm;
im2 = raw_data2 * preproc *datasetm;

% Show
figure(1);
show(im);

% labels
labs = [];
for i = 1:10
    labs = [labs; (i-1)*ones(500,1)];
end
a = prdataset(+im,labs);
a2 = prdataset(+im2,labs);


% split the whole dataset by 80 %
[trData, tstData] = gendat(a,0.8);
[trData1, tstData1] = gendat(a,0.8);
[trData2, tstData2] = gendat(a,0.8);
[trData3, tstData3] = gendat(a,0.8);
[trData4, tstData4] = gendat(a,0.8);
[trData5, tstData5] = gendat(a,0.8);

% train the bpxnc classifier on trData
% [w, h] = bpxnc(trData, hidden_layer_size, num_iter);

% Testing combining result
% [w2,J] = trData * svc(proxm('p', 20));

%Apply PCA into the trainning set
u = scalem([], 'variance') * pcam([],0.6)*parzenc;
w1 = trData1 * u;
w2 = trData2 * u;
w3 = trData3 * u;
w4 = trData4 * u;
w5 = trData5 * u;
w_all = [w1 w2 w3 w4 w5] * maxc;

% test on tstData
cls = a2*w_all*labeld;
tstLabs = a2.nlab - 1;

% cls1 = a2*w1*labeld;
% cls2 = a2*w2*labeld;
% cls3 = a2*w3*labeld;
% cls4 = a2*w4*labeld;
% cls5 = a2*w5*labeld;

% for i = 1:5000
%     
%     X = zeros(10);
%     
%     X(cls1(i)+1) = X(cls1(i)+1) + 1;
%     X(cls2(i)+1) = X(cls2(i)+1) + 1;
%     X(cls3(i)+1) = X(cls3(i)+1) + 1;
%     X(cls4(i)+1) = X(cls4(i)+1) + 1;
%     X(cls5(i)+1) = X(cls5(i)+1) + 1;
%     
%     [M,I] = max(X(:));
%     
%     cls(i) = I-1;
%     
% end

% calculate the error
corr = cls==tstLabs;
err = sum(corr(:) == 0)/numel(corr);

prwaitbar off;