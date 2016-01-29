function [e1,e,e2,e3,c] = single_classifier(trData,tstData,exData,clr,frac)
% trData: training data
% tstData: test data
% clr:classifier
% frac: Fraction of cumulative variance (> 0 & < 1) to retain

%% Setup the parameters
% ANN
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = [15 15 15];   % one layer, each with 50 hidden units
num_iter = 500;          % 500 iterations
num_labels = 10;          % 10 labels, from 0 to 9

%% Classifiers
% Parametric
% Nearest Mean
if strcmp(clr,'nmc')
    u = pcam([],frac) * nmc;
end
% Linear Bayes Normal Classifier 
if strcmp(clr,'ldc')
    u = pcam([],frac) * ldc;
end
% qdc
if strcmp(clr,'qdc')
    u = pcam([],frac) * qdc;
end
% fisherc
if strcmp(clr,'fisherc')
    u = pcam([],frac) * fisherc;
end
% loglc
if strcmp(clr,'loglc')
    u = pcam([],frac) * loglc;
end

% Non-parametric
% KNN
if strcmp(clr,'knnc')
    u = pcam([],frac) * knnc;
end
% parzenc with PCA
if strcmp(clr,'parzenc')
    % u = parzenc;
    u = pcam([],frac) * parzenc;
end

% Advanced
% ANN
if strcmp(clr,'bpxnc')
    [u_bpxnc, h] = bpxnc([], hidden_layer_size, num_iter);
    u = pcam([],frac) * u_bpxnc;
end

% SVM
if strcmp(clr,'svc')
%     kernel = proxm('p',1); % default linear kernel
    kernel = proxm('r');
%     kernel = proxm('d');
    u = svc(kernel);
end

% RandomForest
if strcmp(clr,'randomforestc')
    u = scalem([], 'variance') * pcam([],frac) * randomforestc;
end

% NaiveBayes
if strcmp(clr,'naivebc')
    u = scalem([], 'variance') * pcam([],frac) * naivebc;
end

%% Cross-Validation
% e1 = 0;
% e1 = prcrossval(trData,u,10,1);   % for n =1000
e1 = prcrossval(trData,u,[],3); % for n = 10

%% Training
% simple train
w = trData * u;
% use log-densities for more accuracy (ldc, parzenc)
% w = logdens(w);


%% Evaluation
% e = 0;
e = tstData * w * testc;
% e2 = 0;
e2 = nist_eval('my_rep',w,100);

%% Live Test
% e3 = 0;
[e3, c] = exData * w * testc;




end