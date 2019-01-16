#!/bin/bash
# This script is used to generate the best results found during 
# training, with the aim of inspecting the visual quality.

# we assume that the binaries are in the same folder as the script
BIN=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# test sequences
seqs=(\
derf-hd/park_joy \
derf-hd/speed_bag \
derf-hd/station2 \
derf-hd/sunflower \
derf-hd/tractor \
)
#derf/stefan_mono \
#derf/tennis_mono \
#derf/foreman_mono \
#derf/bus_mono \
#derf/football_mono \

# seq folder
sf='/home/pariasm/Remote/lime/denoising/data/'

# first and last frame
f0=70
f1=85

# function to run the algorithm
function run_rnlm {

	s="$1"
	p="$2"
	w="$3"
	dth="$4"
	bx="$5"
	bt="$6"
	lambda="$7"

	trialfolder=$(printf "s%02dp%02dw%02ddth%06.2fbx%4.2fbt%05.2fl%5.3f\n" \
		$s $p $w $dth $bx $bt $lambda)
	params=$(printf " -p %d -w %d --dth %06.2f --beta_x %4.2f --beta_t %05.2f --lambda %5.3f" \
		$p $w $dth $bx $bt $lambda)
	
	nseqs=${#seqs[@]}
	for seq in ${seqs[@]}
	do
		echo $BIN/nlkalman-train.sh ${sf}${seq} $f0 $f1 $s $trialfolder/$seq \"$params\"
		time $BIN/nlkalman-train.sh ${sf}${seq} $f0 $f1 $s $trialfolder/$seq  "$params"
	done

}

# run with optimal parameters
run_rnlm 10 8 10 38.0 1.2 4.5 1.0 
run_rnlm 20 8 10 45.0 1.2 4.5 1.0 
run_rnlm 40 8 10 60.0 1.2 4.5 1.0 

# run varying the optimal parameters
#                                   #       bus_mono    football_mono
#                                   #       PSNR 29.27  PSNR 28.54
# run_rnlm 20 8 10 35.0 1.2 4.5 1.0 # dth-- PSNR 28.58  PSNR 28.39
# run_rnlm 20 8 10 55.0 1.2 4.5 1.0 # dth++ PSNR 29.30  PSNR 28.50
# run_rnlm 20 8 10 45.0 0.5 4.5 1.0 # b_x-- PSNR 29.20  PSNR 28.50
# run_rnlm 20 8 10 45.0 4.2 4.5 1.0 # b_x++ PSNR 29.25  PSNR 28.57
# run_rnlm 20 8 10 45.0 1.2 2.5 1.0 # b_t-- PSNR 28.91  PSNR 28.53
# run_rnlm 20 8 10 45.0 1.2 6.5 1.0 # b_t++ PSNR 29.22  PSNR 28.31

# conclusions
# dth : tiene que estar cerca de 40 (lo cual parece un umbral muy grande)
#       por debajo de este umbral aparecen empieza a aparecer ruido en algunos
#       patches. 
# b_x : denoising espacial. La incidencia del denoising espacial dura pocos frames
#       (4 frames?). Por medio del promediado temporal, la secuencia o bien
#       gana detalle, si se empieza de un frame demasiado suave, o bien pierde 
#       ruido 
# b_t : controla el promediado temporal.
# l_x : aparentemente no tiene mucha influencia

