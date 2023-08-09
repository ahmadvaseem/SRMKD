function res = vl_srmd_matlab(net, input)

%% If you did not install the matconvnet package, you can use this for testing.

global degpar;
n = numel(net.layers);
%res = struct('x', cell(1,n+1));
res = struct('x', cell(1,n-1));
res(1).x = input;
net.vars(end-1).value = input;
net.vars(1).value = input;

for i = 1 : (n)
    l = net.layers(i); %l = net.layers{i};
    if strfind(l.name,'conv') %switch l.type
        weights = net.params(i-1).value;
        biass = net.params(i).value;
        %case 'conv'
            disp(['Processing ... ',int2str(i),'/',int2str(n)]);
            for noutmaps = 1 : size(weights,4) %for noutmaps = 1 : size(l.weights{1},4)
                z = zeros(size(res(i).x,1),size(res(i).x,2),'single');
                for ninmaps = 1 : size(res(i).x,3)
                    z = z + convn(res(i).x(:,:,ninmaps), rot90(weights(:,:,ninmaps,noutmaps),2),'same'); %z = z + convn(res(i).x(:,:,ninmaps), rot90(l.weights{1}(:,:,ninmaps,noutmaps),2),'same'); % 180 degree rotation for kernel
                end
                res(i+1).x(:,:,noutmaps) = z + biass(noutmaps); %res(i+1).x(:,:,noutmaps) = z + l.weights{2}(noutmaps);
                % net.vars(i+1).value = res(i+1).x;
            end
            
        elseif strfind(l.name,'relu') %case 'relu'
            res(i+1).x = max(res(i).x,0);
            net.vars(i+1).value = res(i+1).x;
            
        elseif strfind(l.name,'layer') %case 'concat'
            if size(degpar,1)~=size(res(i).x,1)
                sigmaMap   = bsxfun(@times,ones(size(res(i).x,1),size(res(i).x,2),1,size(res(i).x,4),'single'),permute(degpar,[3 4 1 2]));
                res(i+1).x = cat(3,res(i).x,sigmaMap);
                net.vars(i+1).value = res(i+1).x;
            else
                res(i+1).x = cat(3,res(i).x,sigmaMap);
                 net.vars(i+1).value = res(i+1).x;
            end
            
        elseif strfind(l.name,'subP') %case 'SubP'
            res(i+1).x = vl_nnSubP(res(i).x, [],'scale',net.meta.scale); %res(i+1).x = vl_nnSubP(res(i).x, [],'scale',l.scale);
            net.vars(i+1).value = res(i+1).x;
        elseif strfind(l.name,'loss')%+++++
           loss = vllab_nn_L2_loss(res(i).x, input); % ???????? which images we need to get loss
    end
    
    if(i<27)
    res(i).x = [];
    end
end

%%++++++++++++++++++++++++++++++++++++ Backward
net.vars(end).der = 1;
for i = n :-1:1 
    %backwardAdvanced(net, net.layers(i))
end
    

end


% function [derInputs, derParams] = backward(obj, inputs, params, derOutpus)
%     %BACKWARD  Bacwkard step
%     %  [DERINPUTS, DERPARAMS] = BACKWARD(OBJ, INPUTS, INPUTS, PARAMS,
%     %  DEROUTPUTS) takes the layer object OBJ and cell arrays of
%     %  inputs, parameters, and output derivatives and produces cell
%     %  arrays of input and parameter derivatives evaluating the layer
%     %  backward.
%       derInputs = {} ;
%       derOutputs = {} ;
% end
% 
% 
% function backwardAdvanced(net, layer)
%     %BACKWARDADVANCED Advanced driver for backward computation
%     %  BACKWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
%     %  the backward step of the layer.
%     %
%     %  The advanced interface can be changed in order to extend DagNN
%     %  non-trivially, or to optimise certain blocks.
%       in = layer.inputIndexes ;
%       out = layer.outputIndexes ;
%       par = layer.paramIndexes ;
%       %net = obj.net ;
% 
%       inputs = {net.vars(in).value} ;
%       derOutputs = {net.vars(out).der} ;
%       for i = 1:numel(derOutputs)
%         if isempty(derOutputs{i}), return ; end
%       end
% 
%       if net.conserveMemory
%         % clear output variables (value and derivative)
%         % unless precious
%         for i = out
%           if net.vars(i).precious, continue ; end
%           net.vars(i).der = [] ;
%           net.vars(i).value = [] ;
%         end
%       end
% 
%       % compute derivatives of inputs and paramerters
%       [derInputs, derParams] = backward ...
%         (inputs, {net.params(par).value}, derOutputs) ;
%       if ~iscell(derInputs) || numel(derInputs) ~= numel(in)
%         error('Invalid derivatives returned by layer "%s".', layer.name);
%       end
% 
%       % accumuate derivatives
%       for i = 1:numel(in)
%         v = in(i) ;
%         if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
%           net.vars(v).der = derInputs{i} ;
%         elseif ~isempty(derInputs{i})
%           net.vars(v).der = net.vars(v).der + derInputs{i} ;
%         end
%         net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1 ;
%       end
% 
%       for i = 1:numel(par)
%         p = par(i) ;
%         if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
%               || isempty(net.params(p).der)
%           net.params(p).der = derParams{i} ;
%         else
%           net.params(p).der = vl_taccum(...
%             1, net.params(p).der, ...
%             1, derParams{i}) ;
%           %net.params(p).der = net.params(p).der + derParams{i} ;
%         end
%         net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1 ;
%         if net.numPendingParamRefs(p) == net.params(p).fanout
%           if ~isempty(net.parameterServer) && ~net.holdOn
%             net.parameterServer.pushWithIndex(p, net.params(p).der) ;
%             net.params(p).der = [] ;
%           end
%         end
%       end
% end