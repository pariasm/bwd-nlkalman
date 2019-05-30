#!/bin/bash
# Fine-tune the algorithm's parameters using a gradient descent

# noise levels
s=10

# parameters of gradient descent
niters=1000 # number of iterations
step=0.2    # step to move along the gradient
gs=0.05     # step to estimate numerical gradient

# test sequences
seqs=(\
boxing \
choreography \
demolition \
grass-chopper \
inflatable \
juggle \
kart-turn \
lions \
ocean-birds \
old_town_cross \
snow_mnt \
swing-boy \
varanus-tree \
wings-turn \
)

ff=1
lf=20

# seq folder
sf='/home/pariasm/denoising/data/train-14/dataset/'

output=${1:-"trials"}

# we assume that the binaries are in the same folder as the script
BIN=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# initial parameters

# sigma 10
f_sc=1
f_dw=0.4
f_th=0.75
s_sc=1
s_dw=0.4
s_th=0.75

# function to run the algorithm
function nlk {

	f_dw="$1"
	f_th="$2"
	s_dw="$3"
	s_th="$4"

	folder="$output/tmp/"
	mkdir -p $folder

	prms=$(printf "%d %f %f %d %f %f" $f_sc $f_dw $f_th $s_sc $s_dw $s_th)

	T=$BIN/nlkalman-seq-gt.sh
	for seq in ${seqs[@]}; do
		echo "$T ${sf}${seq}/%03d.png $ff $lf $s $folder/$seq \"-v 0\" \"no\" \"$prms\" > $folder/${seq}-out"
	done | parallel

	f1_mse=0
	f2_mse=0
	s1_mse=0
	nseqs=${#seqs[@]}
	for seq in ${seqs[@]}; do
#		out=$($T ${sf}${seq} $ff $lf $s $folder/$seq  "$params")
		out=$(cat $folder/${seq}-out)
		read -ra mse <<< "$out"
		f1_mse=$(echo "$f1_mse + ${mse[0]}/$nseqs" | bc -l)
		f2_mse=$(echo "$f2_mse + ${mse[1]}/$nseqs" | bc -l)
#		s1_mse=$(echo "$s1_mse + ${mse[2]}/$nseqs" | bc -l)
	done
	echo $f1_mse $f2_mse $s1_mse

	# remove optical flow and occlusion masks, so that they are recomputed
	rm $folder/$seq/s$s/{*.flo,?occ*.png}
}

# gradient descent
for ((i=0; i < $niters; i++))
do
	# performance of current point
	out=$(nlk $f_dw $f_th $s_dw $s_th)
	read -ra mse <<< "$out"
	f1_mse=${mse[0]}
	f2_mse=${mse[1]}
#	s1_mse=${mse[2]}

	printf "%02d %d %08.5f %08.5f %d %08.5f %08.5f %9.6f %9.6f %9.6f\n" \
		$s $f_sc $f_dw $f_th $s_sc $s_dw $s_th \
		$f1_mse $f2_mse $s1_mse >> $output/table

	# numerical gradient
	out_f_dw=$(nlk $(plambda -c "$f_dw $gs +") $f_th $s_dw $s_th)
	out_f_th=$(nlk $f_dw $(plambda -c "$f_th $gs +") $s_dw $s_th)
#	out_s_dw=$(nlk $f_dw $f_th $(plambda -c "$s_dw $gs +") $s_th)
#	out_s_th=$(nlk $f_dw $f_th $s_dw $(plambda -c "$s_th $gs +"))

#	read -ra mse <<< "$out_f_dw"; echo ${mse[1]}
#	read -ra mse <<< "$out_f_th"; echo ${mse[1]}
#	read -ra mse <<< "$out_s_dw"; echo ${mse[1]}
#	read -ra mse <<< "$out_s_th"; echo ${mse[1]}

	grad=($f2_mse $f2_mse $f2_mse $f2_mse)
	read -ra mse <<< "$out_f_dw"; grad[0]=$(plambda -c "${mse[1]} ${grad[0]} - $gs /")
	read -ra mse <<< "$out_f_th"; grad[1]=$(plambda -c "${mse[1]} ${grad[1]} - $gs /")
#	read -ra mse <<< "$out_s_dw"; grad[2]=$(plambda -c "${mse[1]} ${grad[2]} - $gs /")
#	read -ra mse <<< "$out_s_th"; grad[3]=$(plambda -c "${mse[1]} ${grad[3]} - $gs /")

#	echo $out
#	echo $out_f_dw $f_dw ${grad[0]}
#	echo $out_f_th $f_th ${grad[1]}
#	echo $out_s_dw $s_dw ${grad[2]}
#	echo $out_s_th $s_th ${grad[3]}
	
	# update parameters
	f_dw=$(plambda -c "$f_dw ${grad[0]} $step * -")
	f_th=$(plambda -c "$f_th ${grad[1]} $step * -")
#	s_dw=$(plambda -c "$s_dw ${grad[2]} $step * -")
#	s_th=$(plambda -c "$s_th ${grad[3]} $step * -")

#	echo $f_dw
#	echo $f_th
#	echo $s_dw
#	echo $s_th
done
