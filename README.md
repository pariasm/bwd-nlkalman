BNLK | Backward Non-local Kalman Video Denoising
=========================================

* Author    : Pablo Arias <pariasm@gmail.com>, see `AUTHORS`
* Copyright : (C) 2019, Pablo Arias <pariasm@gmail.com>

This code provides an implementation of the video denoising methods described in:

[P. Arias, J.-M. Morel, "Kalman filtering of patches for
frame-recursive video denoising", NTIRE CVPRW
2019.](http://dev.ipol.im/~pariasm/bnlk/)

Please cite [the paper](http://dev.ipol.im/~pariasm/bnlk/bnlk_files/paper_bnlk.pdf) if you use
results obtained with this code in your
research.


-----
COMPILATION
-----------

The code is in C with some BASH helper scripts. Known dependencies are:
* OpenMP: parallelization [optional, but recommended]
* libpng, libtiff and libjpeg: image i/o
* GNU parallel: parallelization in some helper scripts

Compilation was tested on Ubuntu Linux 16.04 and 18.04.
Configure and compile the source code using cmake and make.
It is recommended that you create a folder for building:
```
$ mkdir build; cd build
$ cmake ..
$ make
```

NOTE: By default, the code is compiled with OpenMP multithreaded
parallelization enabled (if your system supports it). Use the
`OMP_NUM_THREADS` enviroment variable to control the number of threads
used.


The compilation populates `build/bin` with the following binaries:
* `nlkalman-flt` non-local Kalman filtering of a frame 
* `nlkalman-smo` RTS smoother of a frame
* `tvl1flow` compute TV-L1 optical flow between two images
* `awgn` add noise to an image
* `iion` convert image to a different format
* `imprintf` display statistics of an image in printf format
* `plambda` evaluate lambda expression at all pixels of an image.
* `decompose` DCT pyramid decomposition
* `recompose` recomposition from a DCT pyramid

In addition, the following helper scripts will be installed in `bin/`
* `nlkalman-seq.sh` computes NL-Kalman filtering (and optionally) the smoothing over a noisy image sequence.
* `nlkalman-seq-gt.sh` given a clean sequence, adds noise, runs `nlkalman-seq.sh` and computes PSNR.
* `msnlkalman-seq.sh` multiscale version of nlkalman-seq.sh (experimental)
* `msnlkalman-seq-gt.sh` given a clean sequence, adds noise, runs `msnlkalman-seq.sh` and computes PSNR.
* `psnr.sh` computes MSE/RMSE/PSNR between two images

-----
USAGE
-----

**Denoising a noisy sequence**

The simplest use is via the helper scripts:

```
nlkalman-seq.sh /my/video/frames-%03d.png first-frame last-frame sigma out-folder [filt-params] [smoo-params] [flow-params]
```

The method reads the video as a sequence of images. The sequence of images is passed
as a pattern in printf format, thus `frame-%03d.png` means that frames have the following 
filenames: `frame-001.png`, `frame-002.png`, etc. The first and last frame
numbers have to given, as well as the standard deviation of the noise. 
The denoising results are stored in the out-folder. The script produces the following
output sequences:
* `bflo_%03d.flo`: backward optical flow (ie flow from frame t to t-1)
* `bocc_%03d.png`: masks of backwards occluded pixels
* `flt1_%03d.tif`: output of 1st NL-Kalman filtering iteration
* `flt2_%03d.tif`: output of 2nd NL-Kalman filtering iteration (if 2nd iteration is enabled)

If smoothing is performed, the following additional sequences will also be left in `out-folder`
* `fflo_%03d.flo`: forward optical flow (from from frame t to t+1)
* `focc_%03d.png`: masks of forward occluded pixels
* `smo1_%03d.tif`: output of the smoothing pass

You can pass options to the filtering and the smoothing thought the optional
arguments `[filt-params]` and `[smoo-params]`. For a list of
all parameters run `nlkalman-flt -h` and `nlkalman-smo -h`. If no parameters are 
given, the parameters are set automatically based on the noise level `sigma`. The
filtering and smoothing parameters have to be passed between quotes.

Some examples: 

```
# Run the denoising with automatic parameters from frame 3 to 56 with noise 10.
nlkalman-seq.sh /my/video/frames-%03d.png 3 56 10 out/path

# Set patch size during both filtering iterations at 12x12, toggle verbose output:
nlkalman-seq.sh /my/video/frames-%03d.png 3 56 10 out/path "--f1_p 12 --f2_p 12 -v 1" 

# Filter with automatic parametes, smoothing with a patch size of 6x6
nlkalman-seq.sh /my/video/frames-%03d.png 3 56 10 out/path "" "--s1_p 6"  

# Filter with automatic parametes, do not enable smoothing
nlkalman-seq.sh /my/video/frames-%03d.png 3 56 10 out/path "" "no"  
```

Finally, you can also provide a string with parameters for the optical flow and occlusions
detection. The string has to have 6 numbers, three parameters for the backward
optical flow computed during filtering and three for the 
forward flow computed for the smoothing pass:

```"fscale-filt data-weight-filt occl-th-filt fscale-smoo data-weight-smoo occl-th-smoo"```

* `fscale` finest scale of the multiscale TV-L1: `0` means the finest scale, and `1` means that the 
optical flow is computed at half resolution and then upscaled (default is `1`).
* `data-weight` data-attachment weight to control the smoothness of the flow (default is `0.25`)
* `occl-th` threshold on the divergence of the flow use to compute occlusions (default is `0.75`)


For example, to run the denoising with automatic filtering and smoothing
parameters but with custom parameters for the optical flows
```
nlkalman-seq.sh /my/video/frames-%03d.png 3 56 10 out/path "" "" "1 0.2 .75 0 0.2 0.75"
```

**Add noise, denoising and compute PSNR**

Finally, if you want to compute the flow on a sequence with synthetic noise and
then compute the PSNR on the result, you can use:
```
nlkalman-seq.sh /my/clean/video/frames-%03d.png first-frame last-frame sigma out-folder [filt-params] [smoo-params] [flow-params]
```
In addition to the previous outputs, you will find in `out-folder`:
* `out-folder/%03d.tif`: frames with noise added (as tif floating point images)
* `out-folder/measures`: text file with RMSE and PSNR computed globally and per-frame



FILES
-----

The following libraries are also included as part of the code:
* For computing the optical flow: [the IPOL
implementation](http://www.ipol.im/pub/art/2013/26/) of
the [TV-L1 optical flow method of Zack et al.](https://link.springer.com/chapter/10.1007/978-3-540-74936-3_22).
* For image I/O: [Enric Meinhardt's iio](https://github.com/mnhrdt/iio).
* For basic image manipulation: a reduced version of [Enric Meinhardt's imscript](https://github.com/mnhrdt/imscript).
* For command line parsing: [Yecheng Fu's argparse](https://github.com/cofyc/argparse).
* For multiscale denoising: [Pierazzo and Facciolo's DCT multiscaler](https://github.com/npd/multiscaler)


The project is organized as follows
```
root/
├── lib/     3rd party libraries
├── scripts/ helper scripts
└── src/     kalman filtering and smoothing code
```


LICENSE
-------

The code of BNLK is licensed under the GNU Affero General Public License v3.0,
see `LICENSE`. The 3rd party libraries are distributed under their own licences
specified inside each folder.

