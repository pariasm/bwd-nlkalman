% upscale by a factor of 2 using lanczos3 kernel
function up = lanczos3_up(im,sz);

% The output size can be 2n, 2n-1 or 2n+1. The first implementation follows
% Gabriele's pyramid. It first uses imsize(...,2,'lanczos3') to upscale the
% low resimage, resulting in an image with 2n. Then it duplicates or removes
% the last sample. Matlab's imresize resamples the input data at positions
% -0.25, 0.25, 0.75, 1.25, ... 

% -------o-------o-------o-------o-------o------- n low res samples
% -----x---x---x---x---x---x---x---x---x---x----- n --> 2n   (result of imresize)
% -----x---x---x---x---x---x---x---x---x--------- n --> 2n-1
% -----x---x---x---x---x---x---x---x---x---x---x- n --> 2n+1

% input size
h = size(im,1);
w = size(im,2);
c = size(im,3);

% output size
hup = sz(1);
wup = sz(2);

% upsampling kernels
k1 = lanczos3_kernel(.25 + [-3:2]); k1 = k1/sum(k1(:));
k2 = lanczos3_kernel(.75 + [-3:2]); k2 = k2/sum(k2(:));

% upsample horizontally
up1 = zeros(h,2*w,c);
p = [repmat(im (:,1,:),1,3), im , repmat(im (:,w,:),1,2)]; 
for cc = 1:c, up1(:,1:2:2*w,cc) = conv2(p(:,:,cc), k2 , 'valid'); end

p = [repmat(im (:,1,:),1,2), im , repmat(im (:,w,:),1,3)];
for cc = 1:c, up1(:,2:2:2*w,cc) = conv2(p(:,:,cc), k1 , 'valid'); end

% upsample vertically
up = zeros(2*h, 2*w, c);
p = [repmat(up1(1,:,:),3,1); up1; repmat(up1(h,:,:),2,1)];
for cc = 1:c, up(1:2:2*h,:,cc) = conv2(p(:,:,cc), k2', 'valid'); end

p = [repmat(up1(1,:,:),2,1); up1; repmat(up1(h,:,:),3,1)];
for cc = 1:c, up(2:2:2*h,:,cc) = conv2(p(:,:,cc), k1', 'valid'); end

% fix size
if 2*h > sz(1), up = up( 1:end-1   ,:,:); end
if 2*h < sz(1), up = up([1:end,end],:,:); end
if 2*w < sz(2), up = up(:,[1:end,end],:); end
if 2*w > sz(2), up = up(:, 1:end-1   ,:); end



%% %% the elegant way is to do an appropriate resampling for any output size
%% 
%% % -------o-------o-------o-------o-------o------- n low res samples
%% % -----x---x---x---x---x---x---x---x---x---x----- n --> 2n
%% % -------x---x---x---x---x---x---x---x---x------- n --> 2n-1
%% % ---x---x---x---x---x---x---x---x---x---x---x--- n --> 2n+1
%% 
%% 
%% 
%% % input size
%% h = size(im,1);
%% w = size(im,2);
%% 
%% % output size
%% hup = sz(1);
%% wup = sz(2);
%% 
%% % upsample horizontal
%% up1 = zeros(h,wup);
%% if wup == 2*w,
%% 
%% 	k1 = lanczos3_kernel(.25 + [-3:2]);
%% 	k2 = lanczos3_kernel(.75 + [-3:2]);
%% 	p = [repmat(im (:,1),1,3), im , repmat(im (:,w),1,2)]; up1(:,1:2:wup) = conv2(p, k2 , 'valid');
%% 	p = [repmat(im (:,1),1,2), im , repmat(im (:,w),1,3)]; up1(:,2:2:wup) = conv2(p, k1 , 'valid');
%% 
%% elseif wup == 2*w + 1, 
%% 
%% 	k = lanczos3_kernel(0.5 + [-3:2]);
%% 	p = [repmat(im (:,1),1,3), im , repmat(im (:,w),1,3)]; up1(:,1:2:wup) = conv2(p, k, 'valid');
%% 	up1(:,2:2:wup) = im;
%% 
%% 	up1 = imfilter(up1, fspecial('Gaussian',[1,5],0.7),'symmetric'); % TODO this is WRONG!
%% 
%% elseif wup == 2*w - 1,
%% 
%% 	k = lanczos3_kernel(0.5 + [-3:2]);
%% 	p = [repmat(im (:,1),1,2), im , repmat(im (:,w),1,2)]; up1(:,2:2:wup) = conv2(p, k, 'valid');
%% 	up1(:,1:2:wup) = im;
%% 
%% 	up1 = imfilter(up1, fspecial('Gaussian',[1,5],0.7),'symmetric'); % TODO this is WRONG!
%% 
%% else
%% 	error('Incompatible output size');
%% 	return
%% end
%% 
%% 
%% % upsample vertical
%% up = zeros(hup, wup);
%% if hup == 2*w,
%% 
%% 	k1 = lanczos3_kernel(.25 + [-3:2]);
%% 	k2 = lanczos3_kernel(.75 + [-3:2]);
%% 	p = [repmat(up1(1,:),3,1); up1; repmat(up1(h,:),2,1)]; up(1:2:hup,:) = conv2(p, k2', 'valid');
%% 	p = [repmat(up1(1,:),2,1); up1; repmat(up1(h,:),3,1)]; up(2:2:hup,:) = conv2(p, k1', 'valid');
%% 
%% elseif hup == 2*w + 1, 
%% 
%% 	k = lanczos3_kernel(0.5 + [-3:2]);
%% 	p = [repmat(up1(1,:),3,1); up1; repmat(up1(h,:),3,1)]; up(1:2:hup,:) = conv2(p, k', 'valid');
%% 	up(2:2:hup,:) = up1;
%% 
%% 	up1 = imfilter(up1, fspecial('Gaussian',[5,1],0.7),'symmetric'); % TODO this is WRONG!
%% 
%% elseif hup == 2*w - 1,
%% 
%% 	k = lanczos3_kernel(0.5 + [-3:2]);
%% 	p = [repmat(up1(1,:),2,1); up1; repmat(up1(h,:),2,1)]; up (2:2:hup,:) = conv2(p, k', 'valid');
%% 	up (1:2:hup,:) = up1;
%% 
%% 	up1 = imfilter(up1, fspecial('Gaussian',[5,1],0.7),'symmetric'); % TODO this is WRONG!
%% 
%% else
%% 	error('Incompatible output size');
%% 	return
%% end


%% upsampling from n to 2n

%k1 = lanczos3_kernel(.25 + [-3:2]);
%k2 = lanczos3_kernel(.75 + [-3:2]);
%
%up1 = zeros(h,2*w);
%up = zeros(2*h, 2*w);
%
%p = [repmat(im (:,1),1,3), im , repmat(im (:,w),1,2)]; up1(:,1:2:2*w) = conv2(p, k2 , 'valid');
%p = [repmat(im (:,1),1,2), im , repmat(im (:,w),1,3)]; up1(:,2:2:2*w) = conv2(p, k1 , 'valid');
%p = [repmat(up1(1,:),3,1); up1; repmat(up1(h,:),2,1)]; up (1:2:2*h,:) = conv2(p, k2', 'valid');
%p = [repmat(up1(1,:),2,1); up1; repmat(up1(h,:),3,1)]; up (2:2:2*h,:) = conv2(p, k1', 'valid');


%% upsampling from n to 2n+1
%k = lanczos3_kernel(0.5 + [-3:2]);
%
%up1 = zeros(h,2*w+1);
%up = zeros(2*h+1, 2*w+1);
%
%p = [repmat(im (:,1),1,3), im , repmat(im (:,w),1,3)]; up1(:,1:2:2*w+1) = conv2(p, k , 'valid');
%up1(:,2:2:2*w+1) = im;
%p = [repmat(up1(1,:),3,1); up1; repmat(up1(h,:),3,1)]; up (1:2:2*h+1,:) = conv2(p, k', 'valid');
%up (2:2:2*h+1,:) = up1;

%%%% upsampling from n to 2n-1
%%k = lanczos3_kernel(0.5 + [-3:2]);
%%
%%up1 = zeros(h,2*w-1);
%%up = zeros(2*h-1, 2*w-1);
%%
%%p = [repmat(im (:,1),1,2), im , repmat(im (:,w),1,2)]; up1(:,2:2:2*w-1) = conv2(p, k , 'valid');
%%up1(:,1:2:2*w-1) = im;
%%p = [repmat(up1(1,:),2,1); up1; repmat(up1(h,:),2,1)]; up (2:2:2*h-1,:) = conv2(p, k', 'valid');
%%up (1:2:2*h-1,:) = up1;
%%
%%upf = imfilter(up, fspecial('Gaussian',3,0.5))
