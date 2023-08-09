function net = Model_initDagNN(opts)
      f   = opts.conv_f;
      n   = opts.conv_n;
      pad     = floor(f/2);
      if( f == 3 )
        crop = [0, 1, 0, 1];
      elseif( f == 5 )
        crop = [1, 2, 1, 2];
      else
        error('Need to specify crop in deconvolution for f = %d\n', f);
      end

%      %folderModel = 'models';
%      %model_filename = fullfile(folderModel,['SRMDx',int2str(opts.scales),'.mat']);
%       
%       if ~exist(model_filename)
%         fprintf('model do not exist') ;
%       end
%       load(model_filename);
      net  = model_init (opts.scales);
      net = vl_simplenn_tidy(net);     
%       concat = find(cellfun(@(a) strcmp(a.name, 'concat'), net.layers)==1);
        %kernel =  net.meta.directKernel;
      %net.layers = net.layers([2:end-1]);
      
      % Convert to DagNN.
      net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ; 
%      % /////
%       inputs = {'input'};
%       outputs = {sprintf('x2SR_LR_input') };  
%        net.addLayer(outputs{1}, ...
%         dagnn.Concat(), ...
%         inputs, outputs);
%       %/////
      % rename
     %net.renameVar('input', 'x2SR_LR_input');
     for i=2:2:numel(net.layers)
         oldlayernameconv = sprintf('layer%d',i);
         newlayernameconv = sprintf('x2SR_level1_conv%d',i/2);
         net.renameLayer(oldlayernameconv, newlayernameconv);
         oldlayernamerelu = sprintf('layer%d',i+1);
         newlayernamerelu = sprintf('x2SR_level1_relu%d',i/2);
         net.renameLayer(oldlayernamerelu, newlayernamerelu);
     end
     net.layers(end).name = sprintf('subP');
     for i=1:numel(net.layers)
         oldvarname = sprintf('x%d',i);
         newvarname = net.layers(i).name;
         net.renameVar(oldvarname, newvarname);
     end
     next_input = net.layers(end).outputs{1};
%      inputs = {next_input};
%      outputs = {sprintf('x%dSR_%dx_output_subP', 2, 2) };  
%      net.addLayer(outputs{1}, ...
%         dagnn.SubP(), ...
%         inputs, outputs);
%      next_input = outputs{1}; 
      %for s = 1:length(opts.scales)
          %scale = opts.scales(s);
%           scale = opts.scales;
%           level = ceil(log(scale) / log(2));
%           %for l = 1 : level     
%               %% conv 
%                inputs = {next_input};
%                outputs = {sprintf('x%dSR_%dx_output_conv', scale, 2^level) }; 
%                params  = { 'output_conv_f', 'output_conv_b' };
%                net.addLayer(outputs{1}, ...
%                dagnn.Conv('size', [15, 15, 3, 3], ...
%                            'pad', pad, ...
%                            'stride', level), ...
%                             inputs, outputs, params);
%                next_input = outputs{1}; 
%                              
% %             inputs = {next_input};
% %                outputs = {sprintf('x%dSR_%dx_output_conv', scale, 2^l) }; 
% %                params  = { 'output_conv_f', 'output_conv_b' };
% %                net.addLayer(outputs{1}, ...
% %                dagnn.Conv('size', [15, 15, 3, 3], ...
% %                            'pad', pad, ...
% %                            'stride', 1), ...
% %                             inputs, outputs, params);
% %                next_input = outputs{1}; 
% %                
%                
% %             %% HR Loss
% %                inputs = {next_input, ...
% %                         sprintf('x%dSR_%dx_HR', scale, 2^l)};
% %                outputs = {sprintf('x%dSR_%dx_HR_loss', scale, 2^l) };  
% %                net.addLayer(outputs{1}, ...
% %                      dagnn.vllab_dag_loss(...
% %                      'loss_type', opts.loss), ...
% %                        inputs, outputs);
% %                    

%             %%HR Conv
% % %                inputs = {next_input};
% % %                outputs = {sprintf('x2SR_output_lastConv') }; 
% % %                params  = { 'output_conv_f', 'output_conv_b' };
% % %                net.addLayer(outputs{1}, ...
% % %                dagnn.Conv('size', [f, f, 3, 3], ...
% % %                            'pad', pad, ...
% % %                            'stride', 1), ...
% % %                             inputs, outputs, params);
% % %                         
% % %                     filters = ((2).*rand(f,f,3,3, 'single') - 1);%sigma * randn(1, 1, 6, 3, 'single');
% % %                     biases  = zeros(1, 3, 'single');         
% % %                     idx = net.getParamIndex(params{1});
% % %                     net.params(idx).value         = filters;
% % %                     net.params(idx).learningRate  = 1;
% % %                     net.params(idx).weightDecay   = 1;
% % % 
% % %                     idx = net.getParamIndex(params{2});
% % %                     net.params(idx).value         = biases;
% % %                     net.params(idx).learningRate  = 0.1;
% % %                     net.params(idx).weightDecay   = 1;   
% % %                         
% % %                     next_input = outputs{1}; 
               
%                         outputs = { 'outputSum' };
%                         inputs  = { next_input, 'SRMDoutput','BicubicHR'};%,'HR_label'
%                         %params  = { 'weight1'};
%                         net.addLayer(outputs{1}, dagnn.SumW(), inputs, outputs{1});%, params);
% 
%                         %idx = net.getParamIndex(params{1});
%                         %net.params(idx).value         = 1;
% 
%                         next_input = outputs{1};
%              %% conv Loss
% %                inputs = {next_input};
% %                outputs = {sprintf('x%dSR_%dx_output_conv', scale, 2^l) }; 
% %                params  = { 'output_conv_f', 'output_conv_b' };
% %                net.addLayer(outputs{1}, ...
% %                dagnn.Conv('size', [f, f, 3, 3], ...
% %                            'pad', pad, ...
% %                            'stride', 1), ...
% %                             inputs, outputs, params);
% %                next_input = outputs{1}; 
%             %% LR Loss 
               inputs = {next_input, ...
                        sprintf('HR_label')};   
               outputs = {sprintf('x2SR_LR_loss') };  
               net.addLayer(outputs{1}, ...
                     dagnn.Loss(), ...
                       inputs, outputs);    
%           %end
%       %end  
%         params  = { 'output_conv_f', 'output_conv_b' };
%         filters = zeros(15, 15, 3, 3, 'single');
%         filters(2,2,1,1) = 1;
%         filters(2,2,2,2) = 1;
%         filters(2,2,3,3) = 1;
%         biases  = zeros(1, 3, 'single');
% %/////////////
% % ksize  = 15;
% % theta  = pi*rand(1);
% % l1     = 0.1+9.9*rand(1);
% % l2     = 0.1+(l1-0.1)*rand(1);
% % kernel = anisotropic_Gaussian(ksize,theta,l1,l2); % double
% %net.meta.newKernel = 0;
% kernel =  net.meta.AtrpGaussianKernel; %net.meta.directKernel;
% net.meta.scale = opts.scales;
% kernel = kernel(:,:,:,1);
% kernel = cat(3,kernel,kernel,kernel);
% kernel = cat(4,kernel,kernel,kernel);
% kernel = single(kernel);
% 
% filters = kernel;
% %/////////////
%         idx = net.getParamIndex(params{1});
%         net.params(idx).value         = filters;
%         net.params(idx).learningRate  = 1;
%         net.params(idx).weightDecay   = 1;
% 
%         idx = net.getParamIndex(params{2});
%         net.params(idx).value         = biases;
%         net.params(idx).learningRate  = 0.1;
%         net.params(idx).weightDecay   = 1;
% 
% end











% % 
% % sf    = 2;
% % 
% % lr11  = [1 1];
% % lr10  = [1 0];
% % weightDecay = [1 1];
% % nCh = 128; % number of channels
% % C   = 3;   % C = 3 for color image, C = 1 for gray-scale image
% % dim_PCA  = 15;
% % 
% % useBnorm  = 1; % if useBnorm  = 0, you should also use adam.
% % 
% % % Define network
% % net.layers = {} ;
% % 
% % %net.layers{end+1} = struct('type', 'SubP','scale',1/2) ;
% % 
% % net.layers{end+1} = struct('type', 'concat') ;
% % 
% % net.layers{end+1} = struct('type', 'conv', ...
% %     'weights', {{orthrize(sqrt(2/(9*nCh))*randn(3,3,C+dim_PCA+1,nCh,'single')), zeros(nCh,1,'single')}}, ...
% %     'stride', 1, ...
% %     'pad', 1, ...
% %     'dilate',1, ...
% %     'learningRate',lr11, ...
% %     'weightDecay',weightDecay, ...
% %     'opts',{{}}) ;
% % net.layers{end+1} = struct('type', 'relu','leak',0) ;
% % 
% % for i = 1:1:10
% %     
% %     if useBnorm ~= 0
% %         net.layers{end+1} = struct('type', 'conv', ...
% %             'weights', {{orthrize(sqrt(2/(9*nCh))*randn(3,3,nCh,nCh,'single')), zeros(nCh,1,'single')}}, ...
% %             'stride', 1, ...
% %             'learningRate',lr10, ...
% %             'dilate',1, ...
% %             'weightDecay',weightDecay, ...
% %             'pad', 1, 'opts', {{}}) ;
% %         net.layers{end+1} = struct('type', 'bnorm', ...
% %             'weights', {{clipping(sqrt(2/(9*nCh))*randn(nCh,1,'single'),0.01), zeros(nCh,1,'single'),[zeros(nCh,1,'single'), 0.01*ones(nCh,1,'single')]}}, ...
% %             'learningRate', [1 1 1], ...
% %             'weightDecay', [0 0]) ;
% %     else
% %         net.layers{end+1} = struct('type', 'conv', ...
% %             'weights', {{orthrize(sqrt(2/(9*nCh))*randn(3,3,nCh,nCh,'single')), zeros(nCh,1,'single')}}, ...
% %             'stride', 1, ...
% %             'learningRate',lr11, ...
% %             'dilate',1, ...
% %             'weightDecay',weightDecay, ...
% %             'pad', 1, 'opts', {{}}) ;
% %     end
% %     net.layers{end+1} = struct('type', 'relu','leak',0) ;
% %     
% % end
% % 
% % net.layers{end+1} = struct('type', 'conv', ...
% %     'weights', {{orthrize(sqrt(2/(9*nCh))*randn(3,3,nCh,sf^2*C,'single')), zeros(sf^2*C,1,'single')}}, ...
% %     'stride', 1, ...
% %     'learningRate',lr10, ...
% %     'dilate',1, ...
% %     'weightDecay',weightDecay, ...
% %     'pad', 1, 'opts', {{}}) ;
% % 
% % 
% % net.layers{end+1} = struct('type', 'SubP','scale',sf) ;
% % 
% % %%%%%%%%%%%%%%%%%%%
% % net.layers{end+1} = struct('type', 'conv', ...
% %     'weights', {{rand(3,3,3,3,'single'), zeros(3,1,'single')}}, ...
% %     'stride', 1, ...
% %     'learningRate',lr10, ...
% %     'dilate',1, ...
% %     'weightDecay',weightDecay, ...
% %     'pad', 1, 'opts', {{}}) ;
% % %%%%%%%%%%%%%%%%%%%%%%%
% % 
% % net.layers{end+1} = struct('type', 'loss') ; % make sure the new 'vl_nnloss.m' is in the same folder.
% % 
% % % Fill in default values
% % net = vl_simplenn_tidy(net);
% % 
% % end
% % 
% % 
% % function W = orthrize(a)
% % 
% % s_ = size(a);
% % a = reshape(a,[size(a,1)*size(a,2)*size(a,3),size(a,4),1,1]);
% % [u,d,v] = svd(a, 'econ');
% % if(size(a,1) < size(a, 2))
% %     u = v';
% % end
% % %W = sqrt(2).*reshape(u, s_);
% % W = reshape(u, s_);
% % 
% % end
% % 
% % 
% % function A = clipping2(A,b)
% % 
% % A(A<b(1)) = b(1);
% % A(A>b(2)) = b(2);
% % 
% % end
% % 
% % 
% % 
% % function A = clipping(A,b)
% % 
% % A(A>=0&A<b) = b;
% % A(A<0&A>-b) = -b;
% % 
% % end
% % 
