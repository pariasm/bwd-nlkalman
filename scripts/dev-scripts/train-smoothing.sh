#!/bin/bash
# Tune the algorithm's parameters

# noise levels
sigmas=(10 20 40)

# number of trials
ntrials=10000

# test sequences
seqs=(\
train-14/dataset/boxing \
train-14/dataset/choreography \
train-14/dataset/demolition \
train-14/dataset/grass-chopper \
train-14/dataset/inflatable \
train-14/dataset/juggle \
train-14/dataset/kart-turn \
train-14/dataset/lions \
train-14/dataset/ocean-birds \
train-14/dataset/old_town_cross \
train-14/dataset/snow_mnt \
train-14/dataset/swing-boy \
train-14/dataset/varanus-tree \
train-14/dataset/wings-turn \
)

#derf-hd-train/park_joy \
#derf-hd-train/speed_bag \
#derf-hd-train/station2 \
#derf-hd-train/sunflower \
#derf-hd-train/tractor \
# derf/bus_mono \
# derf/foreman_mono \
# derf/football_mono \
# derf/tennis_mono \
# derf/stefan_mono \

# seq folder
#sf='/mnt/nas-pf/'
sf='/home/pariasm/denoising/data/'

output=${1:-"trials"}

export OMP_NUM_THREADS=1

# we assume that the binaries are in the same folder as the script
BIN=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

for ((i=0; i < $ntrials; i++))
do
	# choose randomly a noise level
	r=$(awk -v M=2 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}')
	s=${sigmas[$r]}

	# parameters for first stage filtering
	f1_p=8
	f1_sx=10
	f1_st=5
	f1_bx=$(awk -v M=8  -v        s=$RANDOM 'BEGIN{srand(s); print rand()*M}')
	f1_bt=$(awk -v M=8  -v        s=$RANDOM 'BEGIN{srand(s); print rand()*M}')
	f1_nx=$(awk -v M=99 -v S=2 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M - S) + S)}')
	f1_nt=$(awk -v M=99 -v S=1 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M - S) + S)}')
	f1_ntagg=$(awk -v M=10 -v S=1 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M - S) + S)}')
	if (( f1_ntagg > f1_nt )); then f1_ntagg=$f1_nt; fi

	# parameters for second stage filtering
	f2_p=8
	f2_sx=10
	f2_st=5
	f2_bx=$(awk -v M=8          -v s=$RANDOM 'BEGIN{srand(s); print rand()*M}')
	f2_bt=$(awk -v M=8          -v s=$RANDOM 'BEGIN{srand(s); print rand()*M}')
	f2_nx=$(awk -v M=100 -v S=2 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M - S) + S)}')
	f2_nt=$(awk -v M=100 -v S=1 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M - S) + S)}')
	f2_ntagg=$(awk -v M=10  -v S=1 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M - S) + S)}')
	if (( f2_ntagg > f2_nt )); then f2_ntagg=$f2_nt; fi

	# parameters for smoothing
	s1_p=8
	s1_st=5
	s1_bt=$(awk -v M=8          -v s=$RANDOM 'BEGIN{srand(s); print rand()*M}')
	s1_nt=$(awk -v M=100 -v S=1 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M - S) + S)}')
	s1_ntagg=$s1_nt

#	echo $s $f1_nx $f1_bx $f1_nt $f1_ntagg $f1_bt $f2_nx $f2_bx $f2_nt $f2_ntagg $f2_bt $s1_nt $s1_bt

	trialfolder=$(printf "$output/s%02d-nx%02dbx%05.2fnt%02dnta%02dbt%05.2f" \
		$s $f1_nx $f1_bx $f1_nt $f1_ntagg $f1_bt)
	trialfolder=$(printf "${trialfolder}-nx%02dbx%05.2fnt%02dnta%02dbt%05.2f" \
		$f2_nx $f2_bx $f2_nt $f2_ntagg $f2_bt)
	trialfolder=$(printf "${trialfolder}-nt%02dbt%05.2f\n" \
		$s1_nt $s1_bt)

	params=$(printf " --f1_p %d --f1_sx %d --f1_st %d --f1_nx %d " $f1_p $f1_sx $f1_st $f1_nx)
	params=$(printf "$params --f1_bx %f --f1_nt %d --f1_nt_agg %d --f1_bt %f " $f1_bx $f1_nt $f1_ntagg $f1_bt)
	params=$(printf "$params --f2_p %d --f2_sx %d --f2_st %d --f2_nx %d " $f2_p $f2_sx $f2_st $f2_nx)
	params=$(printf "$params --f2_bx %f --f2_nt %d --f2_nt_agg %d --f2_bt %f " $f2_bx $f2_nt $f2_ntagg $f2_bt)
	params=$(printf "$params --s1_p %d --f1_st %d --s1_nt %d --s1_nt_agg %d --s1_bt %f" $s1_p $f1_st $s1_nt $s1_ntagg $s1_bt)
	params="$params --s1_full 1"

	f1_mse=0
	f2_mse=0
	s1_mse=0
	nseqs=${#seqs[@]}
	ff=1
	lf=20
	if [ ! -d $trialfolder ]
	then
		for seq in ${seqs[@]}
		do
			echo "$BIN/nlkalman-smoothing-train.sh ${sf}${seq} $ff $lf $s $trialfolder \"$params\""
			out=$($BIN/nlkalman-smoothing-train.sh ${sf}${seq} $ff $lf $s $trialfolder  "$params")
			read -ra mse <<< "$out"
			f1_mse=$(echo "$f1_mse + ${mse[0]}/$nseqs" | bc -l)
			f2_mse=$(echo "$f2_mse + ${mse[1]}/$nseqs" | bc -l)
			s1_mse=$(echo "$s1_mse + ${mse[2]}/$nseqs" | bc -l)
			echo $f1_mse $f2_mse $s1_mse
		done
	fi
	
	printf "%02d %02d %05.2f %02d %02d %05.2f %02d %05.2f %02d %02d %05.2f %02d %05.2f %7.4f %7.4f %7.4f\n" \
		$s $f1_nx $f1_bx $f1_nt $f1_ntagg $f1_bt $f2_nx $f2_bx $f2_nt $f2_ntagg $f2_bt $s1_nt $s1_bt \
		$f1_mse $f2_mse $s1_mse >> $output/table

	rm $trialfolder/*.tif

done
