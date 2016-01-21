function a = my_rep(m)

% measurement m

% Preprocess
% preproc = im_box([],0,1)*im_resize([],[20 20])*im_box([],1,0);
preproc = im_rotate([], 220)*im_box([],0,1)*im_resize([],[20 20]);
im = m * preproc *datasetm;

% labels
labs = getlabels(im);

% dataset
a = prdataset(+im,labs);

end
