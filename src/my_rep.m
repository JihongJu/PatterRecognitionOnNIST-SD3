function a = my_rep(m)

% measurement m

% Preprocess
preproc = im_box([],0,1)*im_resize([],[20 20])*im_box([],1,0);
im = m * preproc *datasetm;

% labels
labs = [];
for i = 1:10
    labs = [labs; (i-1)*ones(500,1)];
end

% dataset
a = prdataset(+im,labs);

end
