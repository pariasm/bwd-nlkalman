#! /usr/bin/env octave-cli

% add to path
addpath(fileparts(mfilename('fullpath')))

%%% Get list of arguments passed to script
arg_list = argv();

if length(arg_list) < 4,
	disp('Usage: lanczos3_decompose.m input prefix levels suffix')
	exit;
end

input  = arg_list{1}; % input image
prefix = arg_list{2}; % output prefix
levels = arg_list{3}; % number of levels
suffix = arg_list{4}; % output suffix

% read input image
im = single(iio_read(input));

iio_write(sprintf('%s%d%s', prefix, 0, suffix), single(im))

% generate pyramid
levels = str2num(levels);
for s = 1:levels - 1,

	im = lanczos3_down(im);
	iio_write(sprintf('%s%d%s', prefix, s, suffix), single(im))

end
