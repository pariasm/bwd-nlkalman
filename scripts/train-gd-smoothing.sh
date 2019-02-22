#!/bin/bash
# Fine-tune the algorithm's parameters using a gradient descent

# noise levels
s=20

# parameters of gradient descent
niters=1000 # number of iterations
step=0.01   # step to move along the gradient
gs=0.01     # step to estimate numerical gradient

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

#export OMP_NUM_THREADS=1

# we assume that the binaries are in the same folder as the script
BIN=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# initial parameters
f1_p=8
f1_sx=10
f1_st=5
f1_nx=95
f1_nt=51
f1_ntagg=6
f1_bx=3.68 # trainable
f1_bt=2.26 # trainable

f2_p=8
f2_sx=10
f2_st=5
f2_nx=40
f2_nt=21
f2_ntagg=1
f2_bx=0.32 # trainable
f2_bt=1.58 # trainable

s1_p=8
s1_st=5
s1_nt=75
s1_ntagg=$s1_nt
s1_bt=5.80 # trainable

# function to run the algorithm
function nlk {

	f1_bx="$1"
	f1_bt="$2"
	f2_bx="$3"
	f2_bt="$4"
	s1_bt="$5"

#	folder=$(printf "$output/s%02d-nx%02dbx%05.2fnt%02dnta%02dbt%05.2f" \
#		$s $f1_nx $f1_bx $f1_nt $f1_ntagg $f1_bt)
#	folder=$(printf "${folder}-nx%02dbx%05.2fnt%02dnta%02dbt%05.2f" \
#		$f2_nx $f2_bx $f2_nt $f2_ntagg $f2_bt)
#	folder=$(printf "${folder}-nt%02dbt%05.2f\n" \
#		$s1_nt $s1_bt)
	folder="$output/tmp/"
	mkdir -p $folder

	params=$(printf " --f1_p %d --f1_sx %d --f1_st %d --f1_nx %d " $f1_p $f1_sx $f1_st $f1_nx)
	params=$(printf "$params --f1_bx %f --f1_nt %d --f1_nt_agg %d --f1_bt %f " $f1_bx $f1_nt $f1_ntagg $f1_bt)
	params=$(printf "$params --f2_p %d --f2_sx %d --f2_st %d --f2_nx %d " $f2_p $f2_sx $f2_st $f2_nx)
	params=$(printf "$params --f2_bx %f --f2_nt %d --f2_nt_agg %d --f2_bt %f " $f2_bx $f2_nt $f2_ntagg $f2_bt)
	params=$(printf "$params --s1_p %d --f1_st %d --s1_nt %d --s1_nt_agg %d --s1_bt %f" $s1_p $f1_st $s1_nt $s1_ntagg $s1_bt)
	params="$params --s1_full 1"

	T=$BIN/nlkalman-smoothing-train.sh
	for seq in ${seqs[@]}; do
		echo "$T ${sf}${seq} $ff $lf $s $folder/$seq  \"$params\" > $folder/${seq}-out"
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
		s1_mse=$(echo "$s1_mse + ${mse[2]}/$nseqs" | bc -l)
	done
	echo $f1_mse $f2_mse $s1_mse
}

# gradient descent
for ((i=0; i < $niters; i++))
do
	# performance of current point
	out=$(nlk $f1_bx $f1_bt $f2_bx $f2_bt $s1_bt)
	read -ra mse <<< "$out"
	f1_mse=${mse[0]}
	f2_mse=${mse[1]}
	s1_mse=${mse[2]}

	printf "%02d %02d %08.5f %02d %02d %08.5f %02d %08.5f %02d %02d %08.5f %02d %08.5f %9.6f %9.6f %9.6f\n" \
		$s $f1_nx $f1_bx $f1_nt $f1_ntagg $f1_bt $f2_nx $f2_bx $f2_nt $f2_ntagg $f2_bt $s1_nt $s1_bt \
		$f1_mse $f2_mse $s1_mse >> $output/table

	# numerical gradient
	out_f1_bx=$(nlk $(plambda -c "$f1_bx $gs +") $f1_bt $f2_bx $f2_bt $s1_bt)
	out_f1_bt=$(nlk $f1_bx $(plambda -c "$f1_bt $gs +") $f2_bx $f2_bt $s1_bt)
	out_f2_bx=$(nlk $f1_bx $f1_bt $(plambda -c "$f2_bx $gs +") $f2_bt $s1_bt)
	out_f2_bt=$(nlk $f1_bx $f1_bt $f2_bx $(plambda -c "$f2_bt $gs +") $s1_bt)
	out_s1_bt=$(nlk $f1_bx $f1_bt $f2_bx $f2_bt $(plambda -c "$s1_bt $gs +"))

#	read -ra mse <<< "$out_f1_bx"; echo ${mse[2]}
#	read -ra mse <<< "$out_f1_bt"; echo ${mse[2]}
#	read -ra mse <<< "$out_f2_bx"; echo ${mse[2]}
#	read -ra mse <<< "$out_f2_bt"; echo ${mse[2]}
#	read -ra mse <<< "$out_s1_bt"; echo ${mse[2]}

	grad=($s1_mse $s1_mse $s1_mse $s1_mse $s1_mse)
	read -ra mse <<< "$out_f1_bx"; grad[0]=$(plambda -c "${mse[2]} ${grad[0]} - $gs /")
	read -ra mse <<< "$out_f1_bt"; grad[1]=$(plambda -c "${mse[2]} ${grad[1]} - $gs /")
	read -ra mse <<< "$out_f2_bx"; grad[2]=$(plambda -c "${mse[2]} ${grad[2]} - $gs /")
	read -ra mse <<< "$out_f2_bt"; grad[3]=$(plambda -c "${mse[2]} ${grad[3]} - $gs /")
	read -ra mse <<< "$out_s1_bt"; grad[4]=$(plambda -c "${mse[2]} ${grad[4]} - $gs /")

#	echo $out
#	echo $out_f1_bx $f1_bx ${grad[0]}
#	echo $out_f1_bt $f1_bt ${grad[1]}
#	echo $out_f2_bx $f2_bx ${grad[2]}
#	echo $out_f2_bt $f2_bt ${grad[3]}
#	echo $out_s1_bt $s1_bt ${grad[4]}
	
	# update parameters
	f1_bx=$(plambda -c "$f1_bx ${grad[0]} $step * -")
	f1_bt=$(plambda -c "$f1_bt ${grad[1]} $step * -")
	f2_bx=$(plambda -c "$f2_bx ${grad[2]} $step * -")
	f2_bt=$(plambda -c "$f2_bt ${grad[3]} $step * -")
	s1_bt=$(plambda -c "$s1_bt ${grad[4]} $step * -")

#	echo $f1_bx
#	echo $f1_bt
#	echo $f2_bx
#	echo $f2_bt
#	echo $s1_bt
done
