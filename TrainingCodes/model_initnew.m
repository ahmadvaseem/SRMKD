
function [ net ] = model_initnew(scale)
%MODEL_INITNEW Summary of this function goes here
%   Detailed explanation goes here

nCh1  = 19;
nCh2 = 64;
nCh3 = 128;
pad = [1 1 1 1];
net = dagnn.DagNN();
numfirstNets = 3;
numsecondNets = 8;
%next_input0 = [];

%%% Block of Concat-Conv_Relu Layers
    for i=1:numfirstNets
% %ConcatLayer 1
            inputs = {['input',num2str(i,'%01d')],['dm', num2str(i,'%01d')]};
            outputs = {['concatN', num2str(i,'%01d'),'_1']};
            net.addLayer(outputs{1}, dagnn.Concat(),inputs, outputs);
            %next_input = outputs{1};

% %ConvLayer1
%                inputs = {next_input};
%                outputs = {sprintf(['ConvN', num2str(i,'%01d'),'_1']) }; 
%                params  = { ['output_conv_f_N', num2str(i,'%01d'),'_1'], ['output_conv_b_N', num2str(i,'%01d'),'_1'] };
%                filters = {orthrize(sqrt(2/(9*nCh2))*randn(3,3,nCh1,nCh2,'single'))};
%                biases  = zeros(1, nCh2, 'single'); 
%                
%                net.addLayer(outputs{1}, ...
%                dagnn.Conv('size', size(filters{1}), ...
%                            'pad', pad, ...
%                            'stride', 1), ...
%                             inputs, outputs, params);
% 
%                     idx = net.getParamIndex(params{1});
%                     net.params(idx).value         = filters{1};
%                     net.params(idx).learningRate  = 1;
%                     net.params(idx).weightDecay   = 1;
% 
%                     idx = net.getParamIndex(params{2});
%                     net.params(idx).value         = biases;
%                     net.params(idx).learningRate  = 1;
%                     net.params(idx).weightDecay   = 1;   
%                         
%                     next_input = outputs{1}; 
%                     
%     % Relu Layer1
%         inputs = {next_input};
%         outputs = {['ReluN', num2str(i,'%01d'),'_1']};
%         net.addLayer(outputs{1}, dagnn.ReLU(),inputs, outputs);
         next_input0{i} = num2str(outputs{1});
    end     
    
    %%Concat Layer1 NetSecondPart
            next_input = next_input0;
            inputs = next_input;
            outputs = {['concatN', num2str(2,'%01d')]};
            net.addLayer(outputs{1}, dagnn.Concat(),inputs, outputs);
            next_input = outputs{1};
            
    %%%MiddleConvLayer
               inputs = {next_input};
               outputs = {sprintf('ConvMid') }; 
               params  = { 'output_conv_f_mid', 'output_conv_b_mid' };
               filters = {orthrize(sqrt(2/(9*(numfirstNets*nCh1)))*randn(3,3,(numfirstNets*nCh1),nCh3,'single'))};
               biases  = zeros(1, nCh3, 'single'); 
               
               net.addLayer(outputs{1}, ...
               dagnn.Conv('size', size(filters{1}), ...
                           'pad', pad, ...
                           'stride', 1), ...
                            inputs, outputs, params);

                    idx = net.getParamIndex(params{1});
                    net.params(idx).value         = filters{1};
                    net.params(idx).learningRate  = 1;
                    net.params(idx).weightDecay   = 1;

                    idx = net.getParamIndex(params{2});
                    net.params(idx).value         = biases;
                    net.params(idx).learningRate  = 1;
                    net.params(idx).weightDecay   = 1;   
                        
                    next_input = outputs{1};         
                             
    % Relu Layer
        inputs = {next_input};
        outputs = {['ReluNx1_', num2str(1,'%01d')]};
        net.addLayer(outputs{1}, dagnn.ReLU(),inputs, outputs);
        next_input = outputs{1};   
    
    for i=1:numsecondNets
        %%% ConvLayer
               inputs = {next_input};
               outputs = {sprintf(['ConvNx2_', num2str(i,'%01d')]) }; 
               params  = { ['output_conv_f_Nx2_', num2str(i,'%01d')], ['output_conv_b_Nx2_', num2str(i,'%01d')] };
               filters = {orthrize(sqrt(2/(9*nCh3))*randn(3,3,nCh3,nCh3,'single'))};
               biases  = zeros(1, nCh3, 'single'); 
               
               net.addLayer(outputs{1}, ...
               dagnn.Conv('size', size(filters{1}), ...
                           'pad', pad, ...
                           'stride', 1), ...
                            inputs, outputs, params);

                    idx = net.getParamIndex(params{1});
                    net.params(idx).value         = filters{1};
                    net.params(idx).learningRate  = 1;
                    net.params(idx).weightDecay   = 1;

                    idx = net.getParamIndex(params{2});
                    net.params(idx).value         = biases;
                    net.params(idx).learningRate  = 1;
                    net.params(idx).weightDecay   = 1;   
                        
                    next_input = outputs{1}; 
                    
    %BatchNormLayer
        
                    
    % Relu Layer
        inputs = {next_input};
        outputs = {['ReluNx2_', num2str(i,'%01d')]};
        net.addLayer(outputs{1}, dagnn.ReLU(),inputs, outputs);
        next_input = outputs{1};
    
    end
    
    %%%LastConvLayer
               inputs = {next_input};
               outputs = {sprintf('ConvLast') }; 
               params  = { 'output_conv_f_last', 'output_conv_b_last' };
               filters = {orthrize(sqrt(2/(9*nCh3))*randn(3,3,nCh3,(scale^2 * 3),'single'))};
               biases  = zeros(1, (scale^2 * 3), 'single'); 
               
               net.addLayer(outputs{1}, ...
               dagnn.Conv('size', size(filters{1}), ...
                           'pad', pad, ...
                           'stride', 1), ...
                            inputs, outputs, params);

                    idx = net.getParamIndex(params{1});
                    net.params(idx).value         = filters{1};
                    net.params(idx).learningRate  = 1;
                    net.params(idx).weightDecay   = 1;

                    idx = net.getParamIndex(params{2});
                    net.params(idx).value         = biases;
                    net.params(idx).learningRate  = 1;
                    net.params(idx).weightDecay   = 1;   
                        
                    next_input = outputs{1}; 
                  
               %%% SubPixelLayer
               inputs = {next_input, 'scale'};
               outputs = {sprintf('SubP') }; 
               net.addLayer(outputs{1}, dagnn.SubP(),inputs,outputs);
               next_input = outputs{1}; 
               
               %%%Loss Layer
               inputs = {next_input, 'label'};
               outputs = {sprintf('x2SR_LR_loss') };
               net.addLayer(outputs{1}, dagnn.Loss(),inputs,outputs);
               
    
    
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
