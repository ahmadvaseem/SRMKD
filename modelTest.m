function  outputs  = modelTest( images, degvectors, scales )
%WEIGHTSUMTEST Summary of this function goes here
%   Detailed explanation goes here
numGpus = 1;
% netStruct = load('SWNet0.mat');                                      % Load in Trained CNN                                                       
netStruct1 = load('models/SRMDv2_x4/net-epoch-40.mat'); 
netStruct = netStruct1.net;
net = dagnn.DagNN.loadobj(netStruct);  

    net.mode = 'test' ;
    output_var = 'SubP';
    output_index = net.getVarIndex(output_var);
    net.vars(output_index).precious = 1;
    
    if numGpus >= 1
           net.move('gpu') ;
    end 
    
    
    folderModel = 'models';
    net2 = load(fullfile(folderModel,['SRMDx',int2str(scales),'.mat']));
    net2 = net2.net;
    net2 = vl_simplenn_tidy(net2);
    if numGpus > 0 
        net2 = vl_simplenn_move(net2, 'gpu') ;
    end
    res = vl_srmd(net2, images{1}, degvectors{1}, [],[],'conserveMemory',true,'mode','test','cudnn',true);
    %res = vl_srmd_concise(net, input); % a concise version of "vl_srmd".
    %res = vl_srmd_matlab(net, input); % When use this, you should also set "useGPU = 0;" and comment "net = vl_simplenn_tidy(net);"

    inputs = gather(res(end).x);

    
           %Random Isotropic Kernels
        if scales == 2
                a = 0.80; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
                b = 1.60;
        elseif scales == 3
                a = 1.35; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
                b = 2.40;
        elseif scales == 4
                a = 1.80; % from a range of [0.2, 2] for sf = 2, [0.2, 3] for sf = 3, and [0.2, 4] for sf = 4.
                b = 3.20;
        end
        kernelwidth = 0.2;%(b-a).*rand(1,1) + a;
        kernelwidth2 = 1.4;%(b-a).*rand(1,1) + a;
        kernelwidth3 = 2.0;%(b-a).*rand(1,1) + a;
        kernelwidth4 = 2.6;
        %kernelwidth = 2.6; 
         %for i=1:size(batch,2)
            k1(:,:) = single(fspecial('gaussian',15, kernelwidth));  
            k2(:,:) = single(fspecial('gaussian',15, kernelwidth2)); % Note: the kernel size is fixed to 15X15.
            k3(:,:) = single(fspecial('gaussian',15, kernelwidth3)); 
            k4(:,:) = single(fspecial('gaussian',15, kernelwidth4));

                    %NoiseLevel
            sigma_max = 0;
            sigma1(:) = single(1/255);%sigma1(:) = single(sigma_max*rand(1)/255);
            sigma2(:) = single(1/255);%sigma2(:) = single(sigma_max*rand(1)/255); % single
            sigma3(:) = single(1/255);%sigma3(:) = single(sigma_max*rand(1)/255); % single
            sigma4(:) = single(1/255);
        %end     

       %[~, ~, kernel2, sigma2] =  degradation_model(inputs, opts.scales);
       %[~, ~, kernel3, sigma3] =  degradation_model(inputs, opts.scales);
      

    
        % Select one of them at random.
%         r2 = randi(6);
%         switch r2
%           case 1
            kernel1 = k1;
            kernel2 = k2;
            kernel3 = k3;
            kernel4 = k4;
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
        
        
     %global kernels;
    kernels = {kernel1, kernel2, kernel3, kernel4};
    %global degvectors; % degradation vectors ---> degradation maps
    degvectors1 = getDegMap(net2,kernel1,sigma1);
    degvectors2 = getDegMap(net2,kernel2,sigma2);
    degvectors3 = getDegMap(net2,kernel3,sigma3);
    degvectors4 = getDegMap(net2,kernel4,sigma4);
    
    images = getnewData(inputs,kernels, scales);
    tic;
    inputs = {'input1', images{1}, 'dm1',degvectors1, 'input2', images{2},'dm2',degvectors2,...
                'input3', images{3},'dm3',degvectors3, 'scale', scales} ;    
    
    %inputs = {'input', inputs{1} ,'SRMDoutput',inputs{2}, 'BicubicHR', inputs{3}};
    %inputs = cat(3,inputs{1},inputs{2});
    %inputs = {'input', inputs};
    net.eval(inputs);
    t = toc;
    
    outputs = gather(net.vars(output_index).value);   
%     inputs = getnewData(outputs, inputs);
%     
%     net.eval(inputs);
%     t = toc;
%     
%     outputs = gather(net.vars(output_index).value); 

end

function degMap = getDegMap(net2, kernel,sigma)

    
    %global P;          % PCA projection matrix  
    
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


function outputs = getnewData(sr,kernels, scales)

    %global kernels;
    
    
        for i=1:size(sr,4)
              lr1(:,:,:,i) = imfilter(sr(:,:,:,i),double(kernels{1}(:,:,i)),'replicate'); 
              lr2(:,:,:,i) = imfilter(sr(:,:,:,i),double(kernels{2}(:,:,i)),'replicate'); 
              lr3(:,:,:,i) = imfilter(sr(:,:,:,i),double(kernels{3}(:,:,i)),'replicate'); 
              lr4(:,:,:,i) = imfilter(sr(:,:,:,i),double(kernels{4}(:,:,i)),'replicate'); 
              
        end
        lr1 = imresize(lr1,1/scales,'bicubic');
        lr2 = gather(lr2); lr3 = gather(lr3);lr4 = gather(lr4);
        
        lr2 = imresize(lr2,1/scales, 'nearest');
        lr3 = imresize(lr3,1/scales, 'bilinear');
        lr4 = imresize(lr4,1/scales, 'lanczos2');
%         lr1 = imresize(sr,1/scales,'bicubic');
%         lr2 = gather(sr); lr3 = gather(sr);
%         
%         lr2 = imresize(lr2,1/scales, 'nearest');
%         lr3 = imresize(lr3,1/scales, 'bilinear');
        
        lr2 = gpuArray(lr2); lr3 = gpuArray(lr3);lr4 = gpuArray(lr4);
        outputs{1} = lr1;
        outputs{2} = lr2;
        outputs{3} = lr3;
        outputs{4} = lr4;
end
