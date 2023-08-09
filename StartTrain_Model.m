% This is the training demo of SRMD for scale factor 3.
%
% To run the code, you should install Matconvnet (http://www.vlfeat.org/matconvnet/) first.
%
% For more information, please refer to the following paper.
%
% @inproceedings{zhang2018learning,
%   title={Learning a single convolutional super-resolution network for multiple degradations},
%   author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
%   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
%   pages={3262-3271},
%   year={2018}
% }
%
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

%% xxxxxxxxxxxxxxx  Note!  xxxxxxxxxxxxxxx
%
% run 'Demo_Get_PCA_matrix.m' first to calculate the PCA matrix of vectorized
% blur kernels.
%
% ** You should set the training images folders from "generatepatches.m" first. Then you can run "Demo_Train_SRMD_x3.m" directly.
% **
% ** folders    = {'path_of_your_training_dataset'};% set this from "generatepatches.m" first!
% ** stride     = 40*scale;                         % control the number of image patches, from "generatepatches.m"
% ** nimages    = round(length(filepaths));         % control the number of image patches, from "generatepatches.m"
% **
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
clear;
format compact;
addpath('utilities');
addpath('kernels');

global P;
load('PCA_P.mat');

%-------------------------------------------------------------------------
% Configuration
%-------------------------------------------------------------------------
scale    = 4;
gpu     = 1;

opts = opts_init(scale, gpu);
%opts.model_name        = sprintf('SRMD_x%s', scale); % model name
opts.learningRate     = [logspace(-4,-4,100),logspace(-4,-4,100)/3,logspace(-4,-4,100)/(3^2),logspace(-4,-4,100)/(3^3),logspace(-4,-4,100)/(3^4)];% you can change the learning rate
%opts.learningRate2    = [logspace(-4,-4,100),logspace(-4,-4,100)/3,logspace(-4,-4,100)/(3^2),logspace(-4,-4,100)/(3^3),logspace(-4,-4,100)/(3^4)];% you can change the learning rate
opts.batchSize        = 128; % default  
opts.gpus             = [1]; % this code can only support one GPU!
opts.numSubBatches    = 2;
opts.weightDecay      = 0.0005;
opts.expDir           = fullfile('data', opts.model_name);

%-------------------------------------------------------------------------
%  Initialize model
%-------------------------------------------------------------------------
%netWS  =   Model_initDagNN(opts);
net  =   model_initnew(opts.scales);

%netWS.conserveMemory = 0;
%netSRMD = SRMD_reinit(opts);
%-------------------------------------------------------------------------
%   Train
%-------------------------------------------------------------------------
imdb = [];    
%[imdb,imdb2,imdb3] = generatepatches(opts.scales);
imdb = generatepatches(opts.scales);
%imdbs = {imdb, imdb2, imdb3};
%save( 'imdb2.mat','imdb2', '-V7.3');
%save( 'imdb3.mat','imdb3','-V7.3');
 %   imdb.images.set = 1 ; 
 
    get_batch = @(x,y,mode) getBatch(opts,x,y,mode);

    [net, info] = sw_cnn_train_dag(net, imdb, get_batch, opts.train, ...
                                      'train', find(imdb.set == 1));
                                  
     myCNN = net.saveobj();                                                                                                      % Save the DagNN trained CNN
     save('SWNet0.mat', '-struct', 'myCNN');                              
                                  
                                  
    
function inputs = getBatch(opts, imdb, batch , mode)                                                                               % This fucntion is what generates new mini batch
	
    
    imdb2 = load('imdb2.mat');
    imdb3 = load('imdb3.mat');
    imdb4 = load('imdb4.mat');
    imdb2 = imdb2.imdb2;
    imdb3 = imdb3.imdb3;
    imdb4 = imdb4.imdb4;
%     imdb2 = imdbs{2};
%     imdb3 = imdbs{3};
    

    
%     images2 = imdb2.LRlabels(:,:,:,batch) ;
%     images3 = imdb3.LRlabels(:,:,:,batch) ;
    
    kernel2 = imdb2.kernels(:,:,batch);
    kernel3 = imdb3.kernels(:,:,batch);
    kernel4 = imdb4.kernels(:,:,batch);
    
    sigma2 = imdb2.sigmas(batch);
    sigma3 = imdb3.sigmas(batch);
    sigma4 = imdb4.sigmas(batch);

    images = imdb.LRlabels(:,:,:,batch) ;                                      % Generates specified number of images 
    labels = imdb.HRlabels(:,:,:,batch);                                      % Gets the associated labels
    kernel1 = imdb.kernels(:,:,batch);
    sigma = imdb.sigmas(batch);
   
    
    
         if opts.gpu > 0
            images = gpuArray(images) ;
            labels = gpuArray(labels);
         end  
     
        
    folderModel = 'models';
    net2 = load(fullfile(folderModel,['SRMDx',int2str(opts.scales),'.mat']));
    net2 = net2.net;
    net2 = vl_simplenn_tidy(net2);
    if opts.gpu > 0 
        net2 = vl_simplenn_move(net2, 'gpu') ;
    end
    
     dm1 = getDegMap(net2, kernel1,sigma);
    res = vl_srmd(net2, images, dm1, [],[],'conserveMemory',true,'mode','test','cudnn',true);
    %res = vl_srmd_concise(net, input); % a concise version of "vl_srmd".
    %res = vl_srmd_matlab(net, input); % When use this, you should also set "useGPU = 0;" and comment "net = vl_simplenn_tidy(net);"

    inputs = gather(res(end).x);


        
        %Random Isotropic Kernels
%         if opts.scales == 2
%                 a = 0.80; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
%                 b = 1.60;
%         elseif opts.scales == 3
%                 a = 1.35; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
%                 b = 2.40;
%         elseif opts.scales == 4
%                 a = 1.80; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
%                 b = 3.20;
%         end
%         kernelwidth = (b-a).*rand(1,1) + a;
%         kernelwidth2 = (b-a).*rand(1,1) + a;
        %kernelwidth = 2.6; 
%           for i=1:size(batch,2)
% %             k2(:,:,i) = single(fspecial('gaussian',15, kernelwidth)); % Note: the kernel size is fixed to 15X15.
% %             k3(:,:,i) = single(fspecial('gaussian',15, kernelwidth2)); 
% % 
% %                     %NoiseLevel
% %             sigma_max = 1;
% %             sigma2(:,i) = single(sigma_max*rand(1)/255); % single
% %             sigma3(:,i) = single(sigma_max*rand(1)/255); % single
% %         
%                [~, ~, kk2, ss2] =  degradation_model(inputs, opts.scales);
%                [~, ~, kk3, ss3] =  degradation_model(inputs, opts.scales);
%                k2(:,:,i) = kk2;sigma2(:,i) = ss2;
%                k3(:,:,i) = kk3;sigma3(:,i) = ss3;
%                
%          end   
%        
%         k1  = kernel;
% 
%       
% 
%     
%         % Select one of them at random.
%         r2 = randi(6);
%         switch r2
%           case 1
%             kernel1 = k1;
%             kernel2 = k2;
%             kernel3 = k3;
%           case 2
%             kernel1 = k2;
%             kernel2 = k1;
%             kernel3 = k3;
%           case 3
%             kernel1 = k3;
%             kernel2 = k2;
%             kernel3 = k1;
%           case 4
%             kernel1 = k1;
%             kernel2 = k3;
%             kernel3 = k2;
%           case 5
%             kernel1 = k2;
%             kernel2 = k3;
%             kernel3 = k1;  
%           otherwise
%             kernel1 = k3;
%             kernel2 = k1;
%             kernel3 = k2;  
%         end
%         
        
     %global kernels;
     kernels = {kernel1, kernel2, kernel3, kernel4};
    %global degvectors; % degradation vectors ---> degradation maps
    dm1 = getDegMap(net2,kernel1,sigma);
    dm2 = getDegMap(net2,kernel2,sigma2);
    dm3 = getDegMap(net2,kernel3,sigma3);
    dm4 = getDegMap(net2,kernel4,sigma4);

    images = getnewData(inputs, kernels, opts.scales);
        
            degvectors = dm1;
            degvectors2 = dm2;
            degvectors3 = dm3;
            degvectors4 = dm4;
    %inputs = {'input', images, 'bcinput',imagesBC, 'label', labels} ;    % Assigns images and label to inputs
    inputs = {'input1', images{1}, 'dm1',degvectors, 'input2', images{2},'dm2',degvectors2, 'input3', images{3},'dm3',degvectors3,'scale',opts.scales, 'label', labels} ;    
%     %inputs{2} = {'bicubic', imagesBC};
%     
%     net2.mode = 'test' ;
%     output_var = 'SubP';
%     output_index = net2.getVarIndex(output_var);
%     net2.vars(output_index).precious = 1;
%     
%     if opts.gpu > 0 
%            net2.move('gpu') ;
%     end 
%     net2.eval(inputs);
%     outputs = gather(net2.vars(output_index).value);

    
    
    
end

function degMap = getDegMap(net2, kernel,sigma)

    
    global P;          % PCA projection matrix  
    
        P = net2.meta.P;      

    kk = P*reshape(kernel,size(P,2),size(kernel,3));
    ss = sigma;%ss = imdb.sigmas(batch);
    degMap  = [kk;ss];
    
      %if opts.gpu > 0
        degMap = gpuArray(degMap);
      %end
    
%         n = randi(8);
%         images = data_augmentation(images,n);
%         labels = data_augmentation(labels,n);
%         kernel = data_augmentation(kernel,n);

 

end

function outputs = getnewData(sr, kernels, scale)

    %global kernels;
    
        for i=1:size(sr,4)
              bSr1(:,:,:,i) = imfilter(sr(:,:,:,i),double(kernels{1}(:,:,i)),'replicate'); 
              bSr2(:,:,:,i) = imfilter(sr(:,:,:,i),double(kernels{2}(:,:,i)),'replicate'); 
              bSr3(:,:,:,i) = imfilter(sr(:,:,:,i),double(kernels{3}(:,:,i)),'replicate'); 
              bSr4(:,:,:,i) = imfilter(sr(:,:,:,i),double(kernels{4}(:,:,i)),'replicate'); 
        end
        lr1 = imresize(bSr1,1/scale,'bicubic');
        bSr2 = gather(bSr2); bSr3 = gather(bSr3);bSr4 = gather(bSr4);
        
        lr2 = imresize(bSr2,1/scale, 'nearest');
        lr3 = imresize(bSr3,1/scale, 'bilinear');
        lr4 = imresize(bSr4,1/scale, 'lanczos2');
        lr2 = gpuArray(lr2); lr3 = gpuArray(lr3);lr4 = gpuArray(lr4);
        
        outputs{1} = lr1;
        outputs{2} = lr2;
        outputs{3} = lr3;
        outputs{4} = lr4;
        %outputs  = inputs;
end
% [net, info] = model_train(net,  ...
%     'expDir', opts.expDir, ...
%     'learningRate',opts.learningRate, ...
%     'numSubBatches',opts.numSubBatches, ...
%     'weightDecay',opts.weightDecay, ...
%     'batchSize', opts.batchSize, ...
%     'modelname', opts.modelName, ...
%     'gpus',opts.gpus) ;





