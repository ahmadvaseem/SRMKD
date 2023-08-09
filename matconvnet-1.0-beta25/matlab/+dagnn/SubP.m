classdef SubP < dagnn.Layer
  methods
    function outputs = forward(obj, inputs, params)
      outputs = vl_nnSubP(inputs{1}, inputs{2});
      outputs = {outputs};
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      dX = vl_nnSubP(inputs{1},inputs{2}, derOutputs{1});
      derInputs{1} = dX;
      derInputs{2} = []; 
      derParams = {};
    end
  end
end
