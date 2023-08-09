function out = vl_nntanh(x,dzdy)

%x(x==0) = [];%V = setdiff(x,0);
%r1 = zeros(1,1,342);
%r1(:) = x;
y = tanh(x);

if nargin <= 1 || isempty(dzdy)
  out = y ;
else
  out = dzdy .* (y .* (1 - y)) ; %out = 1 .* (y .* (1 - y)) ; %% 1 is not original Values dzdy is original
end