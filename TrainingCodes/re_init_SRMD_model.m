function net = re_init_SRMD_model(opts)
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
      %model_filename = 'models/SRMDx2.mat';
     folderModel = 'models';
     model_filename = fullfile(folderModel,['SRMDx',int2str(opts.scales),'.mat']);
      
      if ~exist(model_filename)
        fprintf('model do not exist') ;
      end
      load(model_filename);
      net = vl_simplenn_tidy(net);     
%       concat = find(cellfun(@(a) strcmp(a.name, 'concat'), net.layers)==1);
        %kernel =  net.meta.directKernel;
      %net.layers = net.layers([2:end-1]);
      net.layers{end-1}.precious = 1;
      net.layers = net.layers([1:end-1]);
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
     net.renameVar('input', 'x2SR_LR_input');
     for i=2:2:numel(net.layers)+1
         oldlayernameconv = sprintf('layer%d',i);
         newlayernameconv = sprintf('x2SR_level1_conv%d',i/2);
         net.renameLayer(oldlayernameconv, newlayernameconv);
         oldlayernamerelu = sprintf('layer%d',i+1);
         newlayernamerelu = sprintf('x2SR_level1_relu%d',i/2);
         net.renameLayer(oldlayernamerelu, newlayernamerelu);
     end
     for i=1:numel(net.layers)
         oldvarname = sprintf('x%d',i);
         newvarname = net.layers(i).name;
         net.renameVar(oldvarname, newvarname);
     end
     next_input = net.layers(end).outputs{1};
     inputs = {next_input};
     outputs = {sprintf('x%dSR_%dx_output_subP', 2, 2) };  
     net.addLayer(outputs{1}, ...
        dagnn.SubP(), ...
        inputs, outputs);
     next_input = outputs{1}; 
      %for s = 1:length(opts.scales)
          %scale = opts.scales(s);
          scale = opts.scales;
          level = ceil(log(scale) / log(2));
          %for l = 1 : level     
              %% conv 
                inputs = {next_input};
               outputs = {sprintf('x%dSR_%dx_output_conv', scale, 2^level) }; 
               params  = { 'output_conv_f', 'output_conv_b' };
               net.addLayer(outputs{1}, ...
               dagnn.Conv('size', [15, 15, 3, 3], ...
                           'pad', pad, ...
                           'stride', level), ...
                            inputs, outputs, params);
               next_input = outputs{1}; 
                             
%             inputs = {next_input};
%                outputs = {sprintf('x%dSR_%dx_output_conv', scale, 2^l) }; 
%                params  = { 'output_conv_f', 'output_conv_b' };
%                net.addLayer(outputs{1}, ...
%                dagnn.Conv('size', [15, 15, 3, 3], ...
%                            'pad', pad, ...
%                            'stride', 1), ...
%                             inputs, outputs, params);
%                next_input = outputs{1}; 
%                
               
%             %% HR Loss
%                inputs = {next_input, ...
%                         sprintf('x%dSR_%dx_HR', scale, 2^l)};
%                outputs = {sprintf('x%dSR_%dx_HR_loss', scale, 2^l) };  
%                net.addLayer(outputs{1}, ...
%                      dagnn.vllab_dag_loss(...
%                      'loss_type', opts.loss), ...
%                        inputs, outputs);
%                    
            %%HR Conv
%                inputs = {next_input};
%                outputs = {sprintf('x%dSR_%dx_output_BlurHR', scale, 2^l) }; 
%                params  = { 'output_conv_f', 'output_conv_b' };
%                net.addLayer(outputs{1}, ...
%                dagnn.Conv('size', [f, f, 3, 3], ...
%                            'pad', pad, ...
%                            'stride', 1), ...
%                             inputs, outputs, params);
%                next_input = outputs{1}; 
             %% conv Loss
%                inputs = {next_input};
%                outputs = {sprintf('x%dSR_%dx_output_conv', scale, 2^l) }; 
%                params  = { 'output_conv_f', 'output_conv_b' };
%                net.addLayer(outputs{1}, ...
%                dagnn.Conv('size', [f, f, 3, 3], ...
%                            'pad', pad, ...
%                            'stride', 1), ...
%                             inputs, outputs, params);
%                next_input = outputs{1}; 
            %% LR Loss 
               inputs = {next_input, ...
                        sprintf('x%dSR_LR', scale)};   
               outputs = {sprintf('x%dSR_%dx_LR_loss', scale, 1) };  
               net.addLayer(outputs{1}, ...
                     dagnn.vllab_dag_loss(...
                     'loss_type', opts.loss), ...
                       inputs, outputs);    
          %end
      %end  
        params  = { 'output_conv_f', 'output_conv_b' };
        filters = zeros(15, 15, 3, 3, 'single');
        filters(2,2,1,1) = 1;
        filters(2,2,2,2) = 1;
        filters(2,2,3,3) = 1;
        biases  = zeros(1, 3, 'single');
%/////////////
% ksize  = 15;
% theta  = pi*rand(1);
% l1     = 0.1+9.9*rand(1);
% l2     = 0.1+(l1-0.1)*rand(1);
% kernel = anisotropic_Gaussian(ksize,theta,l1,l2); % double
%net.meta.newKernel = 0;
kernel =  net.meta.AtrpGaussianKernel; %net.meta.directKernel;
net.meta.scale = opts.scales;
kernel = kernel(:,:,:,1);
kernel = cat(3,kernel,kernel,kernel);
kernel = cat(4,kernel,kernel,kernel);
kernel = single(kernel);

filters = kernel;
%/////////////
        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;

        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;

end