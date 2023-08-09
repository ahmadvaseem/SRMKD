function out = vl_nntopk(t_score, o_score, l_name,NMax, dzdy )
%VL_NNTOPK Summary of this function goes here
%   Detailed explanation goes here
% o_score -  is original score that comes from validation, it has to be
%                 converted to binary
% binaryVector - is binary form of of original score
% t_score -     result from Tanh layer
% l_name  -  layer name (which top layer is used)
% out -    is result of merged results of binaryVector and tanh result
        
        flag = false;
    if (l_name == 'topk1')
        flag = true;
        if ~(nargin <= 4 || isempty(dzdy))
           dzdy = dzdy(:,1,:,:); end
    end 
    
    if nargin <= 4 && flag
        %Nmax = 10; % get Nmax biggest entries
        [ ~, Ind ] = sort(o_score(:),1,'descend');
        %max_values = Avec(1:Nmax);
        [ ~, ind_col ] = ind2sub(size(o_score),Ind(1:NMax)); % fetch indices
        [~,~,z]=size(o_score(1,1,:));
        for i=1:z    
            if any(ind_col(:) == i)
                o_score(:,:,i) = 1;
            else
                o_score(:,:,i) = 0;
            end
        end
        out0 = [t_score o_score];
%             sorted_result = sort(o_score, 'descend');
%             top_values = sorted_result(1:2);
%             binaryVector = zeros(size(sorted_result));
% 
%             [p,q,r] = size(top_values);
%             binaryVector(1:p,1:q,1:r) = ones(p,q,r);
%             out0 = [t_score binaryVector];           
    else
        [ ~, Ind ] = sort(t_score(:),1,'descend');
        %max_values = Avec(1:Nmax);
        [ ~, ind_col ] = ind2sub(size(t_score),Ind(1:NMax)); % fetch indices
        [~,~,z]=size(t_score(1,1,:));
        for i=1:z    
            if any(ind_col(:) == i)
                t_score(:,:,i) = 1;
            else
                t_score(:,:,i) = 0;
            end
        end
        out0 = t_score;
%             sorted_result = sort(t_score, 'descend');
%             top_values = sorted_result(1:2);
%             binaryVector = zeros(size(sorted_result));
% 
%             [p,q,r] = size(top_values);
%             binaryVector(1:p,1:q,1:r) = ones(p,q,r);
%             out0 = binaryVector;    
    end

    if nargin <= 4 || isempty(dzdy)
        out = out0 ;
    else
        %dzdy = 1;
         out = dzdy .* (out0 .* (1 - out0)) ;
    end
end

