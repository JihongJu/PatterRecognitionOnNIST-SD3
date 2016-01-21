function [e1,e] = diss_classifier(trData,tstData,clr)
% trData: training data
% tstData: test data
% clr:classifier

%% Setup the parameters
% ANN
input_layer_size  = 625;  % 20x20 Input Images of Digits
hidden_layer_size = [50 50 50];   % one layer, each with 50 hidden units
num_iter = 500;          % 500 iterations
num_labels = 10;          % 10 labels, from 0 to 9

%% Classifiers
% dissimilarities only
if strcmp(clr,'')
    D = tstData * (trData * proxm('m',1));% Euclidean dissimilarity matrix
    [e,c] = (1-D)*testc;
    e1 = 1;
    return
end
% Parametric
% Nearest Mean
if strcmp(clr,'nmc')
    u = nmc;
end
% Linear Bayes Normal Classifier 
if strcmp(clr,'ldc')
    u = ldc;
end
% qdc
if strcmp(clr,'qdc')
    u = qdc;
end
% fisherc
if strcmp(clr,'fisherc')
    u = fisherc;
end
% loglc
if strcmp(clr,'loglc')
    u = loglc;
end

% Non-parametric
% KNN
if strcmp(clr,'knnc')
    u = knnc;
end
% parzenc with PCA
if strcmp(clr,'parzenc')
    % u = parzenc;
    u = parzenc;
end

% Advanced
% ANN
if strcmp(clr,'bpxnc')
    [u_bpxnc, h] = bpxnc([], hidden_layer_size, num_iter);
    u = u_bpxnc;
end

% SVM
if strcmp(clr,'svc')
    kernel = proxm('p',1);
    u = svc(kernel);
end

% RandomForest
if strcmp(clr,'randomforestc')
    u = randomforestc;
end



%% Cross-Validation
% e1 = prcrossval(trData,u,10,1);   % for n =1000
% e1 = prcrossval(trData,u,10,10); % for n = 10

%% Training
% random 25% for representation
[repData, trData] = gendat(trData, 0.25);
% train
disSpace = (repData * proxm('d',1));
w = trData * disSpace * u;

%% Cross-Validation
e1 = prcrossval(trData * disSpace,u,10,1);   % for n =1000
% e1 = prcrossval(trData * disSpace,u,10,10); % for n = 10

%% Evaluation
D = tstData * disSpace;
e = D*w*testc;
e2 = 1;
% e2 = nist_eval('my_rep',w,100);

end