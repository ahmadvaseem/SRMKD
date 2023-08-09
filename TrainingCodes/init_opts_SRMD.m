function opts = init_opts_SRMD( scales, gpu)
%INIT_OPTS_SRMD Summary of this function goes here
%   Detailed explanation goes here


opts.scales             = scales;
opts.loss               = 'L2';

opts.beta1        = 0.9;
opts.beta2        = 0.999;
opts.alpha        = 0.01;
opts.epsilon      = 1e-8;
opts.weightDecay  = 0.0001;

%-------------------------------------------------------------------------
%  setting for simplenn
%-------------------------------------------------------------------------

opts.conserveMemory = true;
opts.mode           = 'normal';
opts.cudnn          = true ;
opts.backPropDepth  = +inf ;
opts.skipForward    = false;
opts.numSubBatches  = 1;

%-------------------------------------------------------------------------
%  setting for model
%-------------------------------------------------------------------------

opts.batchSize  = 128 ;
opts.gpus       = [];
opts.numEpochs  = 300 ;
opts.learningRate = 0.0001*ones(1,100,'single');
opts.modelName  = 'model';
opts.expDir     = fullfile('data',opts.modelName) ;

%+++++++++++++++
    opts.scales             = scales;   % training scales (use vector, e.g., [2, 3, 4])
    opts.weight_decay       = 0.0001;
    opts.init_sigma         = 0.001;
    opts.conv_f             = 3;
    opts.conv_n             = 64;
    opts.loss               = 'L2';

    %% training options
    opts.gpu                = gpu;
    opts.batch_size         = 1; %8;
    opts.num_train_batch    = 1;%100;     % number of training batch in one epoch
    opts.num_valid_batch    = 0;%100;      % number of validation batch in one epoch
    opts.lr                 = 1e-7;     % initial learning rate
    opts.lr_step            = 100;      % number of epochs to drop learning rate
    opts.lr_drop            = 0.5;      % learning rate drop ratio
    opts.lr_min             = 1e-9;     % minimum learning rate
    opts.patch_size         = 128;
    opts.data_augmentation  = 1;
    opts.scale_augmentation = 1;

    %% dataset options
    opts.train_dataset          = {};
%     opts.train_dataset{end+1}   = 'T91';
    opts.train_dataset{end+1}   = 'Set1';
%     opts.train_dataset{end+1}   = 'DIV2K_train_HR';
%     opts.train_dataset{end+1}   = 'WED';
    opts.valid_dataset          = {};
%     opts.valid_dataset{end+1}   = 'Set5';
    %opts.valid_dataset{end+1}   = 'Set14';
%     opts.valid_dataset{end+1}   = 'BSDS100';


    %% setup model name
    opts.data_name = 'train';
    for i = 1:length(opts.train_dataset)
        opts.data_name = sprintf('%s_%s', opts.data_name, opts.train_dataset{i});
    end
    
    scale_str = '';
    for s = 1:length(opts.scales)
        scale_str = sprintf('%s%d', scale_str, opts.scales(s));
    end

    opts.net_name = sprintf('SRMD_x%s', scale_str);

    opts.model_name = sprintf('SRMD_x248');


    %% setup dagnn training parameters
    if( opts.gpu == 0 )
        opts.train.gpus     = [];
    else
        opts.train.gpus     = [opts.gpu];
    end
    opts.train.batchSize    = opts.batch_size;
    opts.train.numEpochs    = 50;%1000;
    opts.train.continue     = true;
    opts.train.learningRate = learning_rate_policy(opts.lr, opts.lr_step, opts.lr_drop, ...
                                                   opts.lr_min, opts.train.numEpochs);

    opts.train.expDir = fullfile('models', opts.model_name) ; % model output dir
    if( ~exist(opts.train.expDir, 'dir') )
        mkdir(opts.train.expDir);
    end

    opts.train.model_name       = opts.model_name;
    opts.train.num_train_batch  = opts.num_train_batch;
    opts.train.num_valid_batch  = opts.num_valid_batch;
    
    %% setup loss
    opts.train.derOutputs = {};
    for k = 1:length(opts.scales)

        scale = opts.scales(k);
        level = ceil(log(scale) / log(2));

%         for l = 1 : level
%             opts.train.derOutputs{end + 1} = sprintf('x%dSR_%dx_HR_loss', scale, 2^l);
%             opts.train.derOutputs{end + 1} = 1;
%         end

       for n=1 : scale-1
            if scale-n ~=  2 && scale-n ~=  4 && scale-n ~=  8
              opts.train.derOutputs{end + 1} = sprintf('x%dSR_%dx_LR_loss', scale, scale-n);
              opts.train.derOutputs{end + 1} = 1;  
            end
       end
          
    end
%-------------------------------------------------------------------------
%  update settings
%-------------------------------------------------------------------------

% opts            = vl_argparse(opts, varargin);
% opts.numEpochs  = numel(opts.learningRate);
% 
% if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end


end

