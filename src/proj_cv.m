%% Initialise workspace
clear ; close all; clc;
N = prmemory(2^26);

%% Setup the parameters
% PCA
frac = 0.7; % Fraction of cumulative variance (> 0 & < 1) to retain 
% 0.7 has best performance

%% Load data
fprintf('Loading Data ...\n');
raw_data = prnist(0:9, 1:1:1000);

%% Preprocess
a = my_rep(raw_data);

%% Split the whole dataset by 80 %
[trData, tstData] = gendat(a,0.8); % for n = 1000
% [trData, tstData] = gendat(a,0.5); % for n =10
%% Train, evaluate and test with pca
% [e1_nmc, e_nmc, e2_nmc] = single_classifier(trData, tstData,'nmc',frac);
% [e1_ldc, e_ldc, e2_ldc] = single_classifier(trData, tstData,'ldc',frac);
% [e1_qdc, e_qdc, e2_qdc] = single_classifier(trData, tstData,'qdc',frac);
% [e1_fisherc, e_fisherc, e2_fisherc] = single_classifier(trData, tstData,'fisherc',frac);
% [e1_loglc, e_loglc, e2_loglc] = single_classifier(trData, tstData,'loglc',frac);
% 
% [e1_knnc, e_knnc, e2_knnc] = single_classifier(trData, tstData,'knnc',frac);
% [e1_parzenc, e_parzenc, e2_parzenc] = single_classifier(trData, tstData,'parzenc',frac);
% [e1_bpxnc, e_bpxnc, e2_bpxnc] = single_classifier(trData, tstData,'bpxnc',frac);

[e1_svc, e_svc, e2_svc] = single_classifier(trData, tstData,'svc',frac);
[e1_randomforestc, e_randomforestc, e2_randomforestc] = single_classifier(trData, tstData,'randomforestc',frac);

%% Train, evaluate and test with disimilarities
% [e1_dis, e_dis] = diss_classifier(trData,tstData,'');
% [e1_dis_nmc, e_dis_nmc] = diss_classifier(trData,tstData,'nmc');
% [e1_dis_ldc, e_dis_ldc] = diss_classifier(trData,tstData,'ldc');
% [e1_dis_qdc, e_dis_qdc] = diss_classifier(trData,tstData,'qdc');
% [e1_dis_fisherc, e_dis_fisherc] = diss_classifier(trData,tstData,'fisherc');
% [e1_dis_loglc, e_dis_loglc] = diss_classifier(trData,tstData,'loglc');
% 
% [e1_dis_knnc, e_dis_knnc] = diss_classifier(trData,tstData,'knnc');
% [e1_dis_parzenc, e_dis_parzenc] = diss_classifier(trData,tstData,'parzenc');
% [e1_dis_bpxnc, e_dis_bpxnc] = diss_classifier(trData,tstData,'bpxnc');
% 
% [e1_dis_svc, e_dis_svc] = diss_classifier(trData,tstData,'svc');


%% find best frac for PCA
% e1 =[]; e = []; e2 = [];
% for i = 0.05:0.05:1
%     [e1_knnc, e_knnc, e2_knnc] = single_classifier(trData, tstData,'knnc',i);
%     e1 = [e1,e1_knnc]; e = [e,e_knnc]; e2 = [e2,e2_knnc];
%     if e1_knnc < e1
%         frac = i;
%     end
% end

%% Multiple Classifier Training
% 
% % split, train and combine
% % trData1
% for i = 1:10
%     objects{i,1} = 101:400;
% end
% trData1 = seldat(trData,[],[],objects);
% % trData2
% for i = 1:10
%     objects{i,1} = cat(2,1:100,201:400);
% end
% trData2 = seldat(trData,[],[],objects);
% % trData3
% for i = 1:10
%     objects{i,1} = cat(2,1:200,301:400);
% end
% trData3 = seldat(trData,[],[],objects);
% % trData4
% for i = 1:10
%     objects{i,1} = 1:300;
% end
% trData4 = seldat(trData,[],[],objects);

% %Random split
% [trData1,rest] = gendat(trData, 0.8);
% [trData2,rest] = gendat(trData, 0.8);
% [trData3,rest] = gendat(trData, 0.8);
% [trData4,rest] = gendat(trData, 0.8);
% 
% % train
% w1 = trData1 * u;
% w2 = trData2 * u;
% w3 = trData3 * u;
% w4 = trData4 * u;
% 
% % combine
% w = [w1 w2 w3 w4] * maxc;





prwaitbar off;
