classdef Concat < dagnn.ElementWise
  properties
    dim = 3
  end

  properties (Transient)
    inputSizes = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
        if(size(inputs)) < 3
            %global degvectors;
            sigmaMap   = bsxfun(@times,ones(size(inputs{1},1),size(inputs{1},2),1,size(inputs{1},4)),permute(inputs{2},[3 4 1 2]));
                outputs{1} = vl_nnconcat({inputs{1}, sigmaMap}) ;
        else 
                outputs{1} = vl_nnconcat(inputs, obj.dim) ;
        end 
      obj.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
     if(size(inputs)) < 3
         derInputs = {[],[]};
             %sigmaMap   = bsxfun(@times,ones(size(inputs{1},1),size(inputs{1},2),1,size(inputs{1},4)),permute(inputs{2},[3 4 1 2]));
            %derInputs = vl_nnconcat({inputs{1},sigmaMap}, obj.dim, derOutputs{1}, 'inputSizes', obj.inputSizes) ;
            derOutputs = gather(derOutputs{1});
            derInputs{1} = derOutputs(:,:,1,:);
            derInputs{1}(:,:,end+1,:) = derOutputs(:,:,2,:);
            derInputs{1}(:,:,3,:) = derOutputs(:,:,3,:);
            
            for i = 1: 16
             derInputs{2}(:,:,i,:) = derOutputs(:,:,(3+i),:);
            end
            
     else
         derInputs = vl_nnconcat(inputs, obj.dim, derOutputs{1}, 'inputSizes', obj.inputSizes) ;
     end
      derParams = {} ;
    end

    function reset(obj)
      obj.inputSizes = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      sz = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        sz(obj.dim) = sz(obj.dim) + inputSizes{k}(obj.dim) ;
      end
      outputSizes{1} = sz ;
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      if obj.dim == 3 || obj.dim == 4
        rfs = getReceptiveFields@dagnn.ElementWise(obj) ;
        rfs = repmat(rfs, numInputs, 1) ;
      else
        for i = 1:numInputs
          rfs(i,1).size = [NaN NaN] ;
          rfs(i,1).stride = [NaN NaN] ;
          rfs(i,1).offset = [NaN NaN] ;
        end
      end
    end

    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      % backward file compatibility
      if isfield(s, 'numInputs'), s = rmfield(s, 'numInputs') ; end
      load@dagnn.Layer(obj, s) ;
    end

    function obj = Concat(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
