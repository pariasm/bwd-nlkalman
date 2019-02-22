#!/bin/bash
# Study the loss landscape along lines

# noise levels
s=20

# parameters of gradient descent
niters=1000 # number of iterations
gs=1        # step

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
f1_bx=3.64
f1_bt=2.23

f2_p=8
f2_sx=10
f2_st=5
f2_nx=40
f2_nt=21
f2_ntagg=1
f2_bx=0.25
f2_bt=1.53

s1_p=8
s1_st=5
s1_nt=75
s1_ntagg=$s1_nt
s1_bt=5.79

# function to run the algorithm
function nlk {

	f1_nx="$1"
	f1_nt="$2"
	f1_ntagg="$3"
	f2_nx="$4"
	f2_nt="$5"
	s1_nt="$6"

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



# linear search
for ((f1_nx=5; f1_nx < 200; f1_nx += 5))
do
	# performance of current point
	out=$(nlk $f1_nx $f1_nt $f1_ntagg $f2_nx $f2_nt $s1_nt)
	read -ra mse <<< "$out"
	f1_mse=${mse[0]}
	f2_mse=${mse[1]}
	s1_mse=${mse[2]}

	printf "%02d %02d %08.5f %02d %02d %08.5f %02d %08.5f %02d %02d %08.5f %02d %08.5f %9.6f %9.6f %9.6f\n" \
		$s $f1_nx $f1_bx $f1_nt $f1_ntagg $f1_bt $f2_nx $f2_bx $f2_nt $f2_ntagg $f2_bt $s1_nt $s1_bt \
		$f1_mse $f2_mse $s1_mse >> $output/table
done

