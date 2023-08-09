%==========================================================================
% This is the testing code of SRMD for the <general degradation> of SISR.
% For general degradation, the basic setting is:
%   1. there are tree types of kernels, including isotropic Gaussian,
%      anisotropic Gaussian, and estimated kernel k_b for isotropic
%      Gaussian k_d under direct downsampler (x2 and x3 only).
%   2. the noise level range is [0, 75].
%   3. the downsampler is fixed to bicubic downsampler. 
%      For direct downsampler, you can either train a new model with 
%      direct downsamper or use the estimated kernel k_b under direct 
%      downsampler. The former is preferred.
%   4. there are three models, "SRMDx2.mat" for scale factor 2, "SRMDx3.mat"
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
%global degpar;
addpath('utilities');
imageSets    = {'Set1','Set5','Set14','BSD100','BSD200','T91','General100','SunHays80'}; % testing dataset

global kernels;
Ntop_values = 3;
ssim_results = [];
iqaLR_results =[];
iqaSR_results = [];
ssim_Set = [];
iqaLR_Set = [];
iqaSR_Set = [];
iqaSR_results_temp = [];
imgOut = [];

%% select testing dataset, use GPU or not, ...
setTest      = imageSets([2]); %
showResult   = 1; % 1, show ground-truth, bicubicly interpolated LR image, and restored HR images by SRMD; 2, save restored images
pauseTime    = 1;
useGPU       = 1; % 1 or 0, true or false
%method       = 'SRMD';
folderTest   = 'testsets';
folderResult = 'results';
if ~exist(folderResult,'file')
    mkdir(folderResult);
end

%% scale factor (2, 3, 4)

sf          = 4; %{2, 3, 4}

%% load model with scale factor sf
folderModel = 'models';
load(fullfile(folderModel,['SRMDx',int2str(sf),'.mat']));
%net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end


%gKernel = compareKernels(net,sf,15);


%net0 = PreModel_init();
%[net0, info] = PreModel_train(net0);

%net0 = AlexNet();



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
    
    
    %% perform SISR
    for i = 1 : length(filepaths)
        
        %% degradation parameter (noise level and kernel) setting
        %############################# noise level ################################
        % noise level, from a range of [0, 75]
        
        nlevel     = 15;  % [0, 75]
        
        
        
        %kerneltype = 1;  % {1, 2, 3}
        if (sf == 2 ) || (sf == 3)
            kernel_size = 1;
        else
            kernel_size = 1;
        end
        %############################### kernel ###################################
        % there are tree types of kernels, including isotropic Gaussian,
        % anisotropic Gaussian, and estimated kernel k_b for isotropic Gaussian k_d
        % under direct downsampler (x2 and x3 only).
        
        %+++++++++

           

        %for ky=1:2
        %[kernel, tag] = SelectRandKernel(net,sf,nlevel);
        %++++++++++++++++++
        %% for degradation maps
        %if(ky==1)
        %k1 =kernel;
        for kx=1:kernel_size
            [kernel, tag] = SelectKernel(net,sf,nlevel,kx);
            %else
            degpar = single([net.meta.P*kernel(:); nlevel(:)/255]);
            %degpar = single([kernel(:); nlevel(:)/255]);
            %end
            
            %kerneltype = 3;
            %end
            %----------------------
            %##########################################################################
            folderResultCur = fullfile(folderResult, [setTestCur,tag]);
            if ~exist(folderResultCur,'file')
                mkdir(folderResultCur);
            end

            %     surf(kernel) % show kernel
            %     view(45,55);
            %     title('Assumed kernel');
            %     xlim([1 15]);
            %     ylim([1 15]);
            %     pause(2)
            %     close;
            
            HR  = imread(fullfile(folderTestCur,filepaths(i).name));
            C   = size(HR,3);
            if C == 1
                HR = cat(3,HR,HR,HR);
            end
            [~,imageName,ext] = fileparts(filepaths(i).name);
            HR  = modcrop(HR, sf);
            
            label_RGB = HR;
            blury_HR = imfilter(im2double(HR),double(kernel),'replicate'); % blur%imfilter(im2double(HR),double(kernel),'replicate'); % blur
            LR       = imresize(blury_HR,1/sf,'bicubic'); % bicubic downsampling
            randn('seed',0);
            LR_noisy = LR + nlevel/255.*randn(size(LR)); % add random noise (AWGN)
            input    = single(LR_noisy);
            
            
            
            %tic
            if useGPU
                input = gpuArray(input);
                degpar = gpuArray(degpar);
            end
            %res = vl_srmd(net, input,[],[],'conserveMemory',true,'mode','test','cudnn',true);
            %res = vl_srmd_concise(net, input); % a concise version of "vl_srmd".
            %res = vl_srmd_matlab(net, input); % When use this, you should also set "useGPU = 0;" and comment "net = vl_simplenn_tidy(net);"
            inputs{kx} = input;
            kernels{kx} = kernel;
            degpars{kx} = degpar;
        end
        
         res = modelTest(inputs, degpars, sf);
          
          
            %output_RGB = gather(res(end).x);
            
            output_RGB = gather(res);
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
            
            imgOut(kx).x = output_RGB;
            
            %% calculate PSNR and SSIM
            
             [PSNR_Cur,SSIM_Cur] = Cal_PSNRSSIM(label*255,output*255,sf,sf); %%% single
%             [PSNR_CurHR,SSIM_CurHR] = Cal_PSNRSSIM(label*255,fHR*255,sf,sf); %%% single
%             [PSNR_CurLR,SSIM_CurLR] = Cal_PSNRSSIM(label*255,fLR*255,sf,sf); %%% single
            
            disp([setTestCur,'    ',int2str(i),'    ',num2str(PSNR_Cur,'%2.2f'),'dB','    ',filepaths(i).name]);
            eval(['PSNR_',setTestCur,'_x',num2str(sf),'(',num2str(i),') = PSNR_Cur;']);
            eval(['SSIM_',setTestCur,'_x',num2str(sf),'(',num2str(i),') = SSIM_Cur;']);
            if showResult
                imshow(cat(2,label_RGB,imresize(im2uint8(LR_noisy),sf),im2uint8(output_RGB)));
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


%%% Building an extra Conv Layer for Combination 
% function lnet = lastLayerNet ()
% 
% net.layers = {};
% 
% %parameters
% lr  = [1 0];
% weightDecay = [1 1];
% 
% a = -1;
% b = 1;
% %r = (b-a).*rand(100,1) + a;
% 
% %first Fully Connected Layer
%     net.layers{end+1} = struct('type', 'conv', ...
%         'weights', {{((b-a).*rand(1,1,kernel_size,1024, 'single') + a), zeros(1,1024,'single')}},... %{{rand(1,1,342,1024,'single'), zeros(1,1024,'single')}},...
%         'stride', 1,...
%          'pad', 0, ...
%          'dilate',1,...
%          'learningRate',lr, ...
%          'weightDecay',weightDecay, ... 
%          'opts',{{}}) ;
%      
%    net.layers{end+1} = struct('type', 'relu',...
%                                'leak', 0.2) ;  
%                            
%       lnet = net;                     
% end

%%% Select Random Kernel for First kernel to use with imfilter
function [kernel,tag] = SelectRandKernel(net,sf,nlevel)

method       = 'SRMD';
%kerneltype = randi([1 3],1);
kernel = [];
    while(isempty(kernel))
%%%        kerneltype = randi([1 3],1);
        kerneltype = randi([1 2],1);
        %---------
        if kerneltype == 1
            % type 1, isotropic Gaussian---although it is a special case of anisotropic Gaussian.
            kernelwidth = 2.6; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
            kernel = fspecial('gaussian',15, kernelwidth); % Note: the kernel size is fixed to 15X15.
            tag    = ['_',method,'_x',num2str(sf),'_itrG_',int2str(kernelwidth*10),'_nlevel_',int2str(nlevel)];

        elseif kerneltype == 2
            % type 2, anisotropic Gaussian
%%%            nk     = randi(size(net.meta.AtrpGaussianKernel,4)); % randomly select one
            nk     = randi(20); % randomly select one
            %if(ky==1)
            kernel = net.meta.AtrpGaussianKernel(:,:,:,nk);
            %else
            %kernel = net.meta.AtrpGaussianKernel(:,:,:,kx*25);
            % end

            tag    = ['_',method,'_x',num2str(sf),'_atrG_',int2str(nk),'_nlevel_',int2str(nlevel)];

        elseif kerneltype == 3 && ( sf==2 || sf==3 )
            % type 3, estimated kernel k_b (x2 and x3 only)
            nk     = randi(size(net.meta.directKernel,4)); % randomly select one
            % if (ky==1)
            kernel = net.meta.directKernel(:,:,:,nk);
            %else
            %kernel = net.meta.directKernel(:,:,:,kx*2);
            %end
            tag    = ['_',method,'_x',num2str(sf),'_dirG_',int2str(nk),'_nlevel_',int2str(nlevel)];
        end
    end
end

%%% Select the required kernel from All kernels
function [kernel,tag] = SelectKernel(net,sf,nlevel,kNumber)

    method       = 'SRMD';
    %---------
    if kNumber == 1
        % type 1, isotropic Gaussian---although it is a special case of anisotropic Gaussian.
        kernelwidth = 1.6; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
        kernel = fspecial('gaussian',15, kernelwidth); % Note: the kernel size is fixed to 15X15.
        tag    = ['_',method,'_x',num2str(sf),'_itrG_',int2str(kernelwidth*10),'_nlevel_',int2str(nlevel)];

    elseif kNumber < 344
        % type 2, anisotropic Gaussian
        %nk     = randi(size(net.meta.AtrpGaussianKernel,4)); % randomly select one
        %if(ky==1)
        kernel = net.meta.AtrpGaussianKernel(:,:,:,kNumber-1);
        %else
        %kernel = net.meta.AtrpGaussianKernel(:,:,:,kx*25);
        % end

        tag    = ['_',method,'_x',num2str(sf),'_atrG_',int2str(kNumber),'_nlevel_',int2str(nlevel)];

    elseif kNumber > 343 && ( sf==2 || sf==3 )
        % type 3, estimated kernel k_b (x2 and x3 only)
        %nk     = randi(size(net.meta.directKernel,4)); % randomly select one
        % if (ky==1)
        kernel = net.meta.directKernel(:,:,:,kNumber-343);
        %else
        %kernel = net.meta.directKernel(:,:,:,kx*2);
        %end
        tag    = ['_',method,'_x',num2str(sf),'_dirG_',int2str(kNumber),'_nlevel_',int2str(nlevel)];
    end

end


% function results = compareKernels(net,sf,nlevel)
% 
%     bigg = [];
%     smalll =[];
%     equ = [];
%     remm = [];
%     diff = [];
%     %diff1 = zeros(15,15);
%     for i=1:359
%         kernel1 = SelectKernel(net,sf,nlevel,i);
%        for j=i+1:359
%            
%           kernel2 = SelectKernel(net,sf,nlevel,j);
%           
%           
%           if kernel1 ~= kernel2 
%               diff1 = kernel1 - kernel2;              
%           elseif kernel1 == kernel2 
%               equ{end + 1} = kernel1;
%           end        
%           
%           if exist('diff2','var')
%                if nnz(diff2) <= nnz(diff1)
%                    diff2 = diff1;
%                end
%           else
%                diff2 = diff1;
%           end
%  
%        end
% %             bigg{end + 1} = kernel1;
% %             bigg{end + 1} = kernel2;
%             diff{end + 1} = diff2;
%     end
%         results= bigg;
% end