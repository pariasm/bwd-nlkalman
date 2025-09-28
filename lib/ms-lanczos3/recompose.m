function ret = recompose(prefix, levels, suffix, g, tau, cur)

image = single(iio_read(sprintf('%s%d%s', prefix, cur, suffix)));

%% if (cur == 2)
%% 	image = 2.1*(image-128) + 128;
%% end

if cur >= levels-1 || exist(sprintf('%s%d%s', prefix, cur+1, suffix), 'file') ~= 2 
	% coarsest level
	ret = image;
	return 
end

yL = recompose(prefix, levels, suffix, g, tau, cur+1) ;
yH = image;

% recomposition
sz = size(yH);
if tau,
	H = (yH - lanczos3_up(gblur(lanczos3_down(yH),g),sz));
	L = lanczos3_up(gblur(yL,g),sz);
	H = (abs(H) >= tau) .* H;
	ret = L + H;
else
	ret = yH + lanczos3_up(gblur(yL - lanczos3_down(yH),g),sz);
end


end


function R = gblur(I, s)

pkg load image
if s == 0, R = I;
else
	filter = fspecial( 'gaussian', [1,max(floor(s)*2, 5)], s);
	filter = filter/sum(filter(:));
	for c = 1:size(I,3),
		R(:,:,c) = imfilter(I(:,:,c), filter , 'symmetric');    % horizontal
		R(:,:,c) = imfilter(R(:,:,c), filter', 'symmetric');    % vertical
	end
end

end
