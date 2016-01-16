function a = my_rep(m)

preproc = im_box([],0,1)*im_resize([],[25 25])*im_box([],1,0);  % not aligned
im = m * preproc *datasetm;

labs = [];
for i = 1:10
    labs = [labs; (i-1)*ones(100,1)];
end
a = prdataset(+im,100);