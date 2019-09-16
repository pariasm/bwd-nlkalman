#! /usr/bin/env octave-cli

% add to path
addpath(fileparts(mfilename('fullpath')))

% get list of arguments passed to script
arg_list = argv();

if length(arg_list) < 4,
	disp('Usage: lanczos3_recompose.m input prefix levels suffix [factor]')
	exit;
end

output = arg_list{1}; % input image
prefix = arg_list{2}; % output prefix
levels = arg_list{3}; % number of levels
suffix = arg_list{4}; % output suffix

% recomposition factor
if length(arg_list) < 5, factor = '0'; else factor = arg_list{5}; end

levels = str2num(levels);
factor = str2num(factor);
 
%disp(prefix)
%disp(levels)
%disp(suffix)
%disp(factor)
%disp(output)

im = recompose(prefix, levels, suffix, factor, 0, 0);
iio_write(output, single(im));
