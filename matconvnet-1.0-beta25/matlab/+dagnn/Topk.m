classdef Topk < dagnn.ElementWise
    %TOPK Summary of this class goes here
    %   Detailed explanation goes here 
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nntopk(inputs{1}) ;
        end
    end
    
end

