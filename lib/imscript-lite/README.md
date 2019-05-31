IMSCRIPT-LITE
=============

Some basic image processing tools stolen from
[imscript](https://github.com/mnhrdt/imscript) by Enric Meinhardt-Llopis.

The stolen functions are:
- `iion` convert image format (e.g. tiff to png)
- `plambda` is a very handy tool that allows to evaluate lambda expression in
all pixels of an image. The expression is given using reverse polish notation. `plambda -c` can also be 
used without requiring an image as input, in which it can be used as a
calculator, to evaluate mathematical expression between numbers.

DEPENDENCIES
------------

- libpng
- libtiff
- libjpeg
- libfftw

COMPILATION
-----------

The code is compilable on Unix/Linux (probably on Mac OS as well, but we didn't test). 
We provide a simple Makefile to compile the C code.

