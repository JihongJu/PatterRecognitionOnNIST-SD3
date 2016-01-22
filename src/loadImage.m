function [image_prdatafile,nlabels,labls] = loadImage( filename )
    RGB = imread(filename);
    image_gray = rgb2gray(RGB);
    image_gray = imcomplement(image_gray);
    for i = 5 : 127 : 1148
        x = i + 126;
        for j = 25 : 127 : 2438
            y = j + 126;
            % filter edges
            th1 = 64;
            th2 = 150;
            l=j;r=y;t=i;b=x;
            l_switch_off=false;
            r_switch_off=false;
            t_switch_off=false;
            b_switch_off=false;
            for m = i:x
                if t_switch_off == false && max(image_gray(m , j:y)) >= th1 && max(image_gray(m-1 , j:y)) < th1 && m < x -30
                    t = m;
                    t_switch_off = true;
                end
                if b_switch_off == false && max(image_gray(m , j:y)) < th1 && max(image_gray(m-1 , j:y)) >= th1 && m > i + 30
                    b = m;
                    b_switch_off = true;
                end
            end
            for n = j:y
                if l_switch_off == false && max(image_gray(i:x , n)) >= th2 && max(image_gray(i:x , n-1)) < th2 && n < y - 30
                    l = n;
                    l_switch_off = true;
                end
                if r_switch_off == false && max(image_gray(i:x , n)) < th2 && max(image_gray(i:x , n-1)) >= th2 && n > j + 30
                    r = n;
                    r_switch_off = true;
                end
            end
%             fprintf('l=%s, r=%s, t=%s, b=%s\n',num2str(l),num2str(r),num2str(t),num2str(b));
            imwrite(image_gray(t:b , l:r), ['.\extra_data\',num2str(fix(i / 126)), '_', num2str(fix(j / 126)), '.png']);
        end
    end
    image_prdatafile = prdatafile('extra_data');
    nlabels = [ones(1,20),2*ones(1,20),3*ones(1,20),4*ones(1,20),5*ones(1,20),6*ones(1,20),7*ones(1,20),8*ones(1,20),9*ones(1,20),10*ones(1,20)]';
    labls = [];
    for i = 1 : 1 : size(nlabels,1)
        labls = [labls ; ['digit_' , num2str(nlabels(i) - 1)]];
    end
end