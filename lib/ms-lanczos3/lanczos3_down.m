% downscale by a factor of 2 using lanczos3 kernel
function down = lanczos3_down(im);

% -----x---x---x---x---x---x---x---x---x---x----- 2n high res signal
% -------o-------o-------o-------o-------o------- n low res

% input size
h = size(im,1);
w = size(im,2);
c = size(im,3);

k = 0.5 * lanczos3_kernel(0.5* (0.5 + [-6:5])); k = k/sum(k(:));
ph = [repmat(im(:,1,:),1,5), im , repmat(im(:,w,:),1,6)];

down = zeros(ceil(h/2), ceil(w/2), c);

for cc = 1:c,
	tmp = conv2(ph(:,:,cc), k, 'valid');
	down1 = tmp(:,1:2:end);

	pv = [repmat(down1(1,:),5,1); down1 ; repmat(down1(h,:),6,1)];
	tmp = conv2(pv, k', 'valid');
	down(:,:,cc) = tmp(1:2:end,:);
end

