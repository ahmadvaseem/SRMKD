# SRMKD

Typically, bicubic interpolation is being used for downsampling/upsampling an image and by using different methods, that image is being enhanced. However,
when we apply the bicubic interpolation on a simple/plane image or simple/plane part of an image and compare the results with some state of the art deep
learning methods, bicubic interpolation results outperform. Based on this analysis, we propose a method that can jointly work with bicubic interpolation 
and a deep learning method. To do this, we added the results of bicubic interpolation with the results of a deep learning method named SRMD. As SRMD uses
degradation maps to provide better SR results, so the degradation maps are very important which can give rise to poor performance if wrong degradation maps
are used. Extensive experiments on various images while using random degradation maps show that we surpass the results of SRMD while using randomly 
enerated degradation maps.
