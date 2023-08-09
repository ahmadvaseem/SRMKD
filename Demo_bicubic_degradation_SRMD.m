%==========================================================================
% This is the testing code of SRMD for the widely-used <bicubic degradation>.
% For bicubic degradation, the basic setting is:
%   1. the blur kernel is delta kernel; "kernel = fspecial('gaussian',15,0.2)".
%   2. the noise level is zero; "nlevel = 0".
%   3. the downsampler is fixed to bicubic downsampler.
%      For direct downsampler, you can either train a new model with
%      direct downsamper or use the estimated kernel k_b under bicubic
%      downsampler. The former is preferred.
%   4. there are three SRMD models, "SRMDx2.mat" for scale factor 2, "SRMDx3.mat"
%      for scale factor 3, and "SRMDx4.mat" for scale factor 4.
%==========================================================================
% The basic idea of SRMD is to learn a CNN to infer the MAP of general SISR, i.e.,
% solve x^ = arg min_x 1/(2 sigma^2) ||(kx)\downarrow_s - y||^2 + lamda \Phi(x)
% via x^ = CNN(y,k,sigma;\Theta) or HR^ = CNN(LR,kernel,noiselevel;\Theta).
%
% There involves two important factors, i.e., blur kernel (k; kernel) and noise
% level (sigma; nlevel).
%
% For more information, please refer to the following paper.
%    @article{zhang2017learningsrmd,
%    title={Learning a Single Convolutional Super-Resolution Network for Multiple Degradations},
%    author={Kai, Zhang and Wangmeng, Zuo and Lei, Zhang},
%    year={2017},
%    }
%
% If you have any question, please feel free to contact with <Kai Zhang (cskaizhang@gmail.com)>.
%
% This code is for research purpose only.
%
% by Kai Zhang (Nov, 2017)
%==========================================================================
% clear; clc;
format compact;

addpath('utilities');
imageSets    = {'Set1','Set5','Set14','BSD100','Urban100'}; % testing dataset

%% select testing dataset, use GPU or not, ...
setTest      = imageSets([1,2]); % select the datasets for each tasks
showResult   = 0; % save restored images
pauseTime    = 0;
useGPU       = 0; % 1 or 0, true or false
method       = 'SRMD';
folderTest   = 'testsets';
folderResult = 'results';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

%% scale factor (2, 3, 4)

sf          = 4;

%% load model with scale factor sf
folderModel = 'models';
load(fullfile(folderModel,['SRMDx',int2str(sf),'.mat']));



%% degradation parameter (kernel & noise level) setting
global degpar;
global kernel;
nlevel = 0; % noise level is zero.
kernelwidth = 2.53;
kernel = fspecial('gaussian',15,kernelwidth); % kernel is delta kernel. Note: the kernel size is fixed to 15X15.

%kernel = randn(15 ,15 ,'single');
%kernel = imresize(kernel, [size(kernel,1)+12  size(kernel,2)+12]);
degpar = single([net.meta.P*kernel(:); nlevel(:)/255]);
tag    = ['_',method,'_x',num2str(sf),'_bicubic'];

% %++++++++++++++++++++++++++++++++++++
nCh = 128;
dim_PCA  = 15;
C   = 3;   % C = 3 for color image, C = 1 for gray-scale image
lr10  = [1 1];
weightDecay = [1 1];
padd = [7 7 7 7];

%kernel =  net.meta.AtrpGaussianKernel; %net.meta.directKernel;

net.meta.scale = sf;
%kernel = imresize(kernel, [size(kernel,1)+12  size(kernel,2)+12]);

kernel = kernel(:,:,:,1);
kernel = cat(3,kernel,kernel,kernel);
kernel = cat(4,kernel,kernel,kernel);

%kernel = imresize(kernel, [size(kernel,1)+12  size(kernel,2)+12]);
kernel = single(kernel);
%+++++++

%////////////////////////////////////////////////////////////////////////////////

% net.layers{end+1} = struct('type', 'conv', ...
%      'weights', {{kernel,  zeros(3, 1, 'single')}}, ...
%     'stride', 1, ...
%     'learningRate',lr10, ...
%     'dilate',1, ...
%     'weightDecay',weightDecay, ...
%     'precious', 1,...
%     'pad', padd, 'opts', {{}}) ;

%///////////////////////////////////////
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {{orthrize(sqrt(2/(9*nCh))*randn(3,3,nCh,sf^2*C,'single')), zeros(sf^2*C,1,'single')}}, ...
%     'stride', 1, ...
%     'learningRate',lr10, ...
%     'dilate',1, ...
%     'weightDecay',weightDecay, ...
%     'pad', 1, 'opts', {{}}) ;


net.layers{end+1} = struct('type', 'loss',...
                    'precious', 1 ) ;

%net.layers = net.layers(1:end-1);


%+++++++++++++
%opts = init_opts_SRMD(sf, useGPU);

%net = re_init_SRMD_model(opts);
%net.vars(end-2).precious = 1;
%---------------

%net = vl_simplenn_tidy(net);
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end



for n_set = 1 : numel(setTest)
    
    %% search images
    setTestCur = cell2mat(setTest(n_set));
    disp('--------------------------------------------');
    disp(['    ----',setTestCur,'-----Super-Resolution-----']);
    disp('--------------------------------------------');
    folderTestCur = fullfile(folderTest,setTestCur);
    ext                 =  {'*.jpg','*.png','*.bmp'};
    filepaths           =  [];
    for i = 1 : length(ext)
        filepaths = cat(1,filepaths,dir(fullfile(folderTestCur, ext{i})));
    end
    
    %% prepare results
    eval(['PSNR_',setTestCur,'_x',num2str(sf),' = zeros(length(filepaths),1);']);
    eval(['SSIM_',setTestCur,'_x',num2str(sf),' = zeros(length(filepaths),1);']);
    folderResultCur = fullfile(folderResult, [setTestCur,tag]);
    if ~exist(folderResultCur,'file')
        mkdir(folderResultCur);
    end
    pp = length(filepaths);
    %% perform SISR
    for i = 1 : length(filepaths)
        
        HR  = imread(fullfile(folderTestCur,filepaths(i).name));
        C   = size(HR,3);
        if C == 1
            HR = cat(3,HR,HR,HR);
        end
        [~,imageName,ext] = fileparts(filepaths(i).name);
        HR  = modcrop(HR, sf);
        label_RGB = HR;
        
       %% bicubic degradation
        %blury_HR = imfilter(im2double(HR),double(kernel),'replicate'); % blur
        LR       = imresize(im2double(HR),1/sf,'bicubic'); % bicubic downsampling
        
        input    = im2single(LR);
        %input    = im2single(im2uint8(LR)); % another widely-used setting
        %++++
        %---
        %tic
        if useGPU
            input = gpuArray(input);
        end
        
        %res = vl_srmd(net, input,[],[],'conserveMemory',true,'mode','test','cudnn',true);
        %res = vl_srmd_concise(net, input); % a concise version of "vl_srmd".
        
        %res = vl_srmd_matlab(net, input); %  you should also set "useGPU = 0;" and comment "net = vl_simplenn_tidy(net);"
        
        %++++++
        net.layers{end}.class = input;
        dzdy = single(1);
        res = [];
        evalMode = 'normal';
        opts.conserveMemory = true;
        opts.backPropDepth  = +inf ;
        opts.cudnn          = false ;
        
        net.layers{1,1}.precious = 1;
        res = vl_simplenn(net, input, dzdy, res, ...
            'accumulate', 1, ...
            'mode', evalMode, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'cudnn', opts.cudnn) ;
        
       
        %original_kernel = net.meta.AtrpGaussianKernel;
        
        pca = net.meta.P;
        newKernel = GetNewKernel(res, kernel, pca);
        degpar = single([net.meta.P*newKernel(:); nlevel(:)/255]);
        %degpar = newKernel;
        %res = vl_srmd(net, input,[],[],'conserveMemory',true,'mode','test','cudnn',true);
        res = vl_srmd_matlab(net, input);
        
        %output_RGB = gather(res(end-2).x);
        %-----------------
        output_RGB = gather(res(end).x);
        %toc;
        
        if C == 1
            label  = mean(im2double(HR),3);
            output = mean(output_RGB,3);
        else
            label  = rgb2ycbcr(im2double(HR));
            output = rgb2ycbcr(double(output_RGB));
            label  = label(:,:,1);
            output = output(:,:,1);
        end
        
       %% calculate PSNR and SSIM
        [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(label*255,output*255,sf,sf); %%% single
        disp([setTestCur,'    ',int2str(i),'    ',num2str(PSNR_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
        eval(['PSNR_',setTestCur,'_x',num2str(sf),'(',num2str(i),') = PSNR_Cur;']);
        eval(['SSIM_',setTestCur,'_x',num2str(sf),'(',num2str(i),') = SSIM_Cur;']);
        if showResult
            imshow(cat(2,label_RGB,imresize(im2uint8(LR),sf),im2uint8(output_RGB)));
            drawnow;
            title(['SISR     ',filepaths(i).name,'    ',num2str(PSNR_Cur,'%2.2f'),'dB'],'FontSize',12)
            pause(pauseTime)
            imwrite(output_RGB,fullfile(folderResultCur,[imageName,'_x',int2str(sf),'_',int2str(PSNR_Cur*100),'.png']));% save results
        end
        
    end
    disp(['Average PSNR is ',num2str(mean(eval(['PSNR_',setTestCur,'_x',num2str(sf)])),'%2.2f'),'dB']);
    disp(['Average SSIM is ',num2str(mean(eval(['SSIM_',setTestCur,'_x',num2str(sf)])),'%2.4f')]);
    
   %% save PSNR and SSIM results
    save(fullfile(folderResultCur,['PSNR_',setTestCur,'_x',num2str(sf),'.mat']),['PSNR_',setTestCur,'_x',num2str(sf)]);
    save(fullfile(folderResultCur,['SSIM_',setTestCur,'_x',num2str(sf),'.mat']),['SSIM_',setTestCur,'_x',num2str(sf)]);
    
end
function W = orthrize(a)

s_ = size(a);
a = reshape(a,[size(a,1)*size(a,2)*size(a,3),size(a,4),1,1]);
[u,d,v] = svd(a, 'econ');
if(size(a,1) < size(a, 2))
    u = v';
end
%W = sqrt(2).*reshape(u, s_);
W = reshape(u, s_);

end

function result = GetNewKernel(res,kernel,pca)
    global degpar; 

    lmbda = 1e-9;
     
     
  %-der2 = res(26).dzdw{1};
  
  %der2 = res(2).dzdw{1};
  
  der1 = res(2).dzdx;
%   der1 = der1(:,:,[1:end-3]);
  
  %der1 = mean(der1(:,:,1),3);
  der1 = der1(:,:,[1:end-4]);
  %-der2 = mean(der2(:,:,1),3);
  
 % out = [];
  %[m,n,p] = size(der2);

  out = mean(mean(der1,1),2);
  out = reshape(out,numel(out),1,1);
  
  k = pinv(pca);
  k = k * out;
  k1 = reshape(k,15,15,1);
  
  %-k1 = lmbda*((k1 + der2)/2);
   k1 = lmbda*(k1);

  
  %k2 = net.meta.directKernel;
  
  k2 = kernel;
  k2 = mean(k2(:,:,1),3);
  
  result = k1 + k2;
%   result = lmbda * ((degpar + out)/2);
  
  %k = k + net.meta.directKernel;
      %out = out + mean(der2(:,:,ch),3);
    %im = squeeze(der2(:,:,ch));
    % out = mean(der2,3);
    %out = squeeze(nanmean([nanmean(der2,1)],2)); 
  %%%%%%%%%%%%%%%%%%%
%-------------------------------
end