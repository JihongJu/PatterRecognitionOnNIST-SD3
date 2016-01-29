%% Initialise workspace
clear ; close all; clc;
N = prmemory(2^26);

%% Setup the parameters
% PCA
frac = 0.7; % Fraction of cumulative variance (> 0 & < 1) to retain 
% 0.7 has best performance
% frac = 25;

%% Load data
fprintf('Loading Data ...\n');
raw_data = prnist(0:9, 1:50:1000);
labels = getlabels(raw_data);

% Extra data
[extra_data,nlabls,labls] = loadImage('example_digits.png');
extra_data.nlab = nlabls;
extra_data = setlabels(extra_data,labls);



%% Preprocess
a = my_rep(raw_data);
exData = my_rep(extra_data);

%% Show
% figure(1);
% show(raw_data);
% figure(2);
% show(extra_data)

%% Split the whole dataset by 80 %
% [trData, tstData] = gendat(a,0.8); % for n = 1000
[trData, tstData] = gendat(a,0.5); % for n =10
% [trData, tstData] = gendat(a,0.25); % for n = 10 combined classifier

%% Train, evaluate and test with pca
% [e1_nmc, e_nmc, e2_nmc, e3_nmc] = single_classifier(trData, tstData, exData, 'nmc',frac);
% [e1_ldc, e_ldc, e2_ldc, e3_ldc] = single_classifier(trData, tstData, exData, 'ldc',frac);
% [e1_qdc, e_qdc, e2_qdc, e3_qdc] = single_classifier(trData, tstData, exData, 'qdc',frac);
% [e1_fisherc, e_fisherc, e2_fisherc,  e3_fisherc] = single_classifier(trData, exData,  tstData,'fisherc',frac);
% [e1_loglc, e_loglc, e2_loglc, e3_loglc] = single_classifier(trData, tstData, exData, 'loglc',frac);
% 
% [e1_knnc, e_knnc, e2_knnc, e3_knnc] = single_classifier(trData, tstData, exData, 'knnc',frac);
% [e1_parzenc, e_parzenc, e2_parzenc, e3_parzenc] = single_classifier(trData, tstData, exData, 'parzenc',frac);
% [e1_bpxnc, e_bpxnc, e2_bpxnc, e3_bpxnc] = single_classifier(trData, tstData, exData, 'bpxnc',frac);
% 
[e1_svc, e_svc, e2_svc,  e3_svc] = single_classifier(trData, tstData, exData, 'svc',frac);
% [e1_randomforestc, e_randomforestc, e2_randomforestc, e3_randomforestc] = single_classifier(trData, tstData, exData, 'randomforestc',frac);


% [e1_naivebc, e_naivebc, e2_naivebc, e3_naivebc] = single_classifier(trData, tstData, exData, 'naivebc',frac);

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
% % 
[e1_dis_svc, e_dis_svc] = diss_classifier(trData,tstData,'svc');

%% k-means
% w = trData* pcam([],0.4);
% [l,d] =  kmeans(tstData*w,10);
% tstLabs = tstData.nlab - 1;
% cls = d.nlab - 1;
% corr = cls==tstLabs;
% e_kmeans = sum(corr(:) == 0)/numel(corr);
% % 83%


%% Combined Classifier Training
% % Sampling with replacement
% [trData1,~] = gendat(trData, 0.8);
% [trData2,~] = gendat(trData, 0.8);
% [trData3,~] = gendat(trData, 0.8);
% [trData4,~] = gendat(trData, 0.8);
% [trData5,~] = gendat(trData, 0.8);
% % train
% w1 = trData1 * (scalem([], 'variance') * pcam([],frac) * ldc);
% w2 = trData2 * (scalem([], 'variance') * pcam([],frac) * ldc);
% w3 = trData3 * (scalem([], 'variance') * pcam([],frac) * ldc);
% w4 = trData4 * (scalem([], 'variance') * pcam([],frac) * ldc);
% w5 = trData5 * (scalem([], 'variance') * pcam([],frac) * ldc);
% 
% % combine
% w = [w1 w2 w3 w4 w5] * maxc;
% 
% % evaluation
% e_combined = tstData * w * testc;
% e2_combined = nist_eval('my_rep',w,100);
% e3_combined = exData * w * testc;
% 
% % compare with single classifier
% [e1_base_ldc, e_base_ldc, e2_base_ldc, e3_base_ldc] = single_classifier(trData1, tstData, exData, 'ldc',frac);
% % [e1_base_qdc, e_base_qdc, e2_base_qdc, e3_base_qdc] = single_classifier(trData1, tstData, exData, 'qdc',frac);
% % [e1_base_knnc, e_base_knnc, e2_base_knnc, e3_base_knnc] = single_classifier(trData1, tstData, exData, 'knnc',frac);


%% Learning Curve
% % Replace the following untrained classifier if you want to check the
% % learning curve for another classifier (BE CAREFUL when n = 1000)

% u = pcam([],frac) * ldc;
% e = cleval(a,{u},[],50);
% plote(e);

%% find best frac for PCA
% disable e3 computing before starting test
% e1 =[]; e = []; e2 = [];
% for i = 0.05:0.05:1
%     [e1_ldc, e_ldc, e2_ldc] = single_classifier(trData, tstData,[],'ldc', i);
%     e1 = [e1,e1_ldc]; e = [e,e_ldc]; e2 = [e2,e2_ldc];
%     if e1_ldc < e1
%         frac = i;
%     end
% %     [e1_knnc, e_knnc, e2_knnc] = single_classifier(trData, tstData,'knnc',i);
% %     e1 = [e1,e1_knnc]; e = [e,e_knnc]; e2 = [e2,e2_knnc];
% %     if e1_knn < e1
% %         frac = i;
% %     end
% end


%% n-fold training
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



prwaitbar off;
