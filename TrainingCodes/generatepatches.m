function imdb = generatepatches(scale)
%function [imdb, imdb2, imdb3] = generatepatches(scale)

folder     = 'testsets/Set5';
%scale      = 2;

size_label = 32*scale; % size of the HR patch
stride     = 40*scale; % 1) control the total number of patches
stride_low = stride/scale;
batchSize  = 256;
nchannels  = 3;
kernelsize = 15;

size_input = size_label;
padding    = abs(size_input - size_label)/2;

ext               =  {'*.jpg','*.png','*.bmp'};
filepaths         =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folder, ext{i})));
end
%imdb1 = [];imdb2=[];imdb3=[];
%rdis       = randperm(length(filepaths));
%rdis       = sort(rdis);
nimages    = round(length(filepaths));  % 2) control the total number of patches
scalesc    = min(1,0.5 + 0.05*randi(15,[1,nimages]));
nn         = randi(8,[1,nimages]);

count      = 0;
for i = 1 : nimages
    im = imread(fullfile(folder,filepaths((i)).name));%im = imread(fullfile(folder,filepaths(rdis(i)).name));
    im = data_augmentation(im, nn(i));
    disp([i,nimages,round(count/256)])
    im = imresize(im,scalesc(i),'bicubic');
    
    im = modcrop(im, scale);
    
    LR = ones([size(im,1)/scale,size(im,2)/scale]);
    [hei,wid,~] = size(im);
    for x = 1 : stride : (hei-size_input+1)
        for y = 1 : stride : (wid-size_input+1)
            x_l = stride_low*(x-1)/stride + 1;
            y_l = stride_low*(y-1)/stride + 1;
            if x_l+size_input/scale-1 > size(LR,1) || y_l+size_input/scale-1 > size(LR,2)
                continue;
            end
            count=count+1;
        end
    end
end


numPatches = ceil(count/batchSize)*batchSize;
diffPatches = numPatches - count;
disp([numPatches,numPatches/batchSize,diffPatches]);

disp('---------------PAUSE------------');
%pause

count = 0;
imdb.LRlabels  = zeros(size_label/scale, size_label/scale, nchannels, numPatches,'single');
imdb.HRlabels  = zeros(size_label, size_label, nchannels, numPatches,'single');
imdb.kernels   = zeros(kernelsize, kernelsize, numPatches,'single');
imdb.sigmas    = zeros(1,numPatches,'single');
imdb2.kernels   = zeros(kernelsize, kernelsize, numPatches,'single');
imdb2.sigmas    = zeros(1,numPatches,'single');
imdb3 = imdb2;imdb4=imdb2;
for i = 1 : nimages
    im = imread(fullfile(folder,filepaths((i)).name));%imread(fullfile(folder,filepaths(rdis(i)).name));
    im = data_augmentation(im, nn(i));
    disp([i,nimages,round(count/256)])
    im = imresize(im,scalesc(i),'bicubic');
    
    im = im2double(im);
    
    [LR, HR, kernel, sigma] = degradation_model(im, scale);
    %%%%
     [~, ~, kernel2, sigma2] = degradation_model(im, scale);
     [~, ~, kernel3, sigma3] = degradation_model(im, scale);
     [~, ~, kernel4, sigma4] = degradation_model(im, scale);
     %kernel2(:,:,i) = k2; kernel3(:,:,i) = k3;
     %sigma2(:,i) = s2; sigma3(:,i) = s3;
    %%%%
    
    [hei,wid,~] = size(HR);
%     for nim = 1: 3
%          if nim == 1
%             LR = LR; kernel = kernel; sigma = sigma; imdb = imdb1;
%         elseif nim==2
%             LR = LR2; kernel = kernel2; sigma = sigma2; imdb = imdb2;
%         elseif nim==3
%             LR=LR3; kernel = kernel3; sigma = sigma3; imdb = imdb3;
%         end
        
        for x = 1 : stride : (hei-size_input+1)
            for y = 1 : stride : (wid-size_input+1)
                x_l = stride_low*(x-1)/stride + 1;
                y_l = stride_low*(y-1)/stride + 1;
                if x_l+size_input/scale-1 > size(LR,1) || y_l+size_input/scale-1 > size(LR,2)
                    continue;
                end
                subim_LR = LR(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:nchannels);
                
%                 subim_LR2 = LR2(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:nchannels);
%                 subim_LR3 = LR3(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:nchannels);
                
                subim_HR = HR(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:nchannels);
%                 subim_HR2 = HR2(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:nchannels);
%                 subim_HR3 = HR3(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:nchannels);
%                 
                count=count+1;
                imdb.HRlabels(:, :, :, count) = subim_HR;
%                 imdb2.HRlabels(:, :, :, count) = subim_HR2;
%                 imdb3.HRlabels(:, :, :, count) = subim_HR3;
                
                imdb.LRlabels(:, :, :, count) = subim_LR;
                
%                 imdb2.LRlabels(:, :, :, count) = subim_LR2;
%                 imdb3.LRlabels(:, :, :, count) = subim_LR3;
                
                imdb.kernels(:,:,count)       = single(kernel);
                imdb2.kernels(:,:,count)       = single(kernel2);
                imdb3.kernels(:,:,count)       = single(kernel3);
                imdb4.kernels(:,:,count)       = single(kernel4);
                
                imdb.sigmas(count)            = single(sigma);
                imdb2.sigmas(count)            = single(sigma2);
                imdb3.sigmas(count)            = single(sigma3);
                imdb4.sigmas(count)            = single(sigma4);

                if count<=diffPatches
                    imdb.LRlabels(:, :, :, end-count+1)   = LR(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:nchannels);
                    
%                     imdb2.LRlabels(:, :, :, end-count+1)   = LR2(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:nchannels);
%                     imdb3.LRlabels(:, :, :, end-count+1)   = LR3(x_l : x_l+size_input/scale-1, y_l : y_l+size_input/scale-1,1:nchannels);
%                     
                    
                    imdb.HRlabels(:, :, :, end-count+1)   = HR(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:nchannels);
%                     imdb2.HRlabels(:, :, :, end-count+1)   = HR2(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:nchannels);
%                     imdb3.HRlabels(:, :, :, end-count+1)   = HR3(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,1:nchannels);
                    
                    imdb.kernels(:,:,end-count+1)         = single(kernel);
                    imdb2.kernels(:,:,end-count+1)         = single(kernel2);
                    imdb3.kernels(:,:,end-count+1)         = single(kernel3);
                    imdb4.kernels(:,:,end-count+1)         = single(kernel4);
                    
                    imdb.sigmas(end-count+1)              = single(sigma);
                     imdb2.sigmas(end-count+1)              = single(sigma2);
                      imdb3.sigmas(end-count+1)              = single(sigma3);
                      imdb4.sigmas(end-count+1)              = single(sigma4);
                end
            end
        end
        
%         if nim == 1
%             imdb1 = imdb;
%         elseif nim==2
%             imdb2 = imdb;
%         elseif nim==3
%             imdb3 = imdb;
%         end
%     end
end 
imdb.set    = uint8(ones(1,size(imdb.LRlabels,4)));
save( 'imdb2.mat','imdb2', '-V7.3');
save( 'imdb3.mat','imdb3','-V7.3');
save( 'imdb4.mat','imdb4','-V7.3');

