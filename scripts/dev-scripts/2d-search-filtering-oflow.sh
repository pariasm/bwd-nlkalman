#!/bin/bash
# Fine-tune the algorithm's parameters using a gradient descent

# noise levels
s=40

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
f1_sc=1
f1_dw=0.4
f1_th=0.5
f2_sc=1
f2_dw=0.4
f2_th=0.5

# function to run the algorithm
function nlk {

	f1_dw="$1"
	f1_th="$2"
	f2_dw="$3"
	f2_th="$4"

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

# linear search
for f1_dw in $(seq 0.1 0.1 1.0)
do
	for f2_dw in $(seq 0.1 0.1 1.0)
	do
		echo $f1_dw $f2_dw

		# performance of current point
		out=$(nlk $f1_dw $f2_th $f2_dw $f2_th)
		read -ra mse <<< "$out"
		f1_mse=${mse[0]}
		f2_mse=${mse[1]}
	#	s1_mse=${mse[2]}

		printf "%02d %d %08.5f %08.5f %d %08.5f %08.5f %9.6f %9.6f %9.6f\n" \
			$s $f1_sc $f1_dw $f1_th $f2_sc $f2_dw $f2_th \
			$f1_mse $f2_mse $s1_mse >> $output/table
	done
done

