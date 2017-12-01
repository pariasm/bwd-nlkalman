load rnlm.vari.table
rnlm = rnlm_vari;

s = 10;

% each row of the table has
% sigma psz wsz whx wht whvt lambda psnr
%
% out of these, psz, wsz and whvt are constant

t = rnlm(find(rnlm(:,1) == s),[4,5,7,8]);

hxg = 4*s*s*[0:.01:1];
htg = 4*s*s*[0:.01:1];
lg  = [0:.05:1];
[hx,ht,l] = ndgrid(hxg, htg, lg);

if ~exist('P'),
	disp('computing matrix P ...')
	P = reshape(griddatan(t(:,1:3), t(:,4), [hx(:), ht(:), l(:)], 'linear' ), size(hx));
else
	disp('P already computed ... omitting computation')
end

mm = min(P(:));
MM = max(P(:));
%mm = MM-2;
for il = 2:length(lg)-1
	imagesc(hxg, htg, P(:,:,il),[mm MM]),
	axis equal
	axis tight
	colorbar
	title(sprintf('lambda = %f',lg(il)))
	pause,
end

%% % best results for sigma = 10
%% b_hx = hxg(18);
%% b_ht = htg(57);
%% b_l  =  lg(18);
%% 
%% % best results for sigma = 20
%% b_ht = 976;
%% b_hx = 304;
%% b_l  = .95;
%% 
%% % best results for sigma = 40
%% b_ht = 2496;
%% b_hx = 1280;
%% b_l  = .95;
%% 
%% disp('best values:');
%% disp([b_hx b_ht b_l]);
%% 
%% % show table values with similar parameters
%% t(:,1:2) = sqrt(t(:,1:2));
%% disp(t(find((abs(t(:,1) - sqrt(b_hx)) < 5) & ...
%%             (abs(t(:,2) - sqrt(b_ht)) < 5) & ...
%%             (abs(t(:,3) - b_l ) < .3)),:))



% best parameters
%                hx   ht    l   psnr
% sigma = 10 |   68  224  .85   32.5
% sigma = 20 |  304  976  1.0   28.3 
% sigma = 40 | 1608 2995  1.0   24.5
