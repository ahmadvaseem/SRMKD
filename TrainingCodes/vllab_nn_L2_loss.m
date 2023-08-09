function Y = vllab_nn_L2_loss(X, Z, dzdy)
% -------------------------------------------------------------------------
%   Description:
%       L2 (MSE) loss function used in MatConvNet NN
%       forward : Y = vllab_nn_L2_loss(X, Z)
%       backward: Y = vllab_nn_L2_loss(X, Z, dzdy)
%
%   Input:
%       - X     : predicted data
%       - Z     : ground truth data
%       - dzdy  : the derivative of the output
%
%   Output:
%       - Y     : loss when forward, derivative of loss when backward
%
%   Citation: 
%       Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------
    
    global sum_kernel;
    a = size(X,1);
    Xo = X;
    if size(X,1)~=size(Z,1)
        for i=1:size(X,4)
               X(:,:,:,i) = imfilter(X(:,:,:,i),double(sum_kernel(:,:,:,i)),'replicate'); 
        end
        n = size(Z,1)/size(X,1);
        X = imresize(X, n, 'bicubic');
        %+++++++++++++++++++
        if(size(Z,1) ~= size(Z,2))
            if(size(Z,1) > size(X,1))
                X = imresize(X, [size(X,1)+1  size(X,2)]);
            elseif(size(Z,1) < size(X,1))
                X = X([1:end-1],:,:);
            elseif(size(Z,2) > size(X,2))
                X = imresize(X, [size(X,1)  size(X,2)+1]);
            elseif(size(Z,2) < size(X,2))
                X = X(:,[1:end-1],:);
            end 
        end
        %-----------------------
       
    end
    if nargin <= 2
        diff = (X - Z) .^ 2;
        Y = 0.5 * sum(diff(:));
    else
        Y = (X - Z) * dzdy;
        if size(Y,1)~= a
            Y = imresize(Y ,1/n, 'bicubic');
        end
    end
    %++++++++++++++++
    if(size(Xo,1) ~= size(Y,1) || size(Xo,2) ~= size(Y,2) )
        if(size(Y,1) > size(Xo,1))
            dif = size(Y,1) - size(Xo,1);
            Y = Y([1:end-dif],:,:);
        elseif(size(Y,1) < size(Xo,1))
            dif = size(Xo,1) - size(Y,1);
            Y = imresize(Y, [size(Y,1)+dif  size(Y,2)]);
        end
        if(size(Y,2) > size(Xo,2))
            dif = size(Y,2) - size(Xo,2);
            Y = Y(:,[1:end-dif],:);
        elseif(size(Z,2) < size(X,2))
            Y = imresize(Y, [size(Y,1)  size(Y,2)+dif]);
        end 
    end
    %----------------
    
end
