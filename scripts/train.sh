#!/bin/bash
# Tune the algorithm's parameters

# noise levels
sigmas=(10 20 40 30)

# fixed parameters
pszs=(4 8 12)
wszs=(5 10 15)

# number of trials
ntrials=1000

# test sequences
seqs=(\
derf-hd-train/park_joy \
derf-hd-train/speed_bag \
derf-hd-train/station2 \
derf-hd-train/sunflower \
derf-hd-train/tractor \
)
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
echo $BIN

for ((i=0; i < $ntrials; i++))
do
	# randomly draw noise level and parameters

	# noise level
	r=$(awk -v M=2 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}')
	s=${sigmas[$r]}

	# patch size
#	r=$(awk -v M=2 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}')
#	p=${pszs[$r]}

	# search region
#	r=$(awk -v M=2 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}')
#	w=${wszs[$r]}

	p=8
	w=10

	# spatial and temporal weights
#	dth=$(awk -v M=60 -v S=0 -v s=$RANDOM 'BEGIN{srand(s); print rand()*(M - S) + S}')
	np=$(awk -v M=99 -v S=0 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M - S) + S)}')
	bx=$(awk -v M=8 -v s=$RANDOM 'BEGIN{srand(s); print rand()*M}')
	bt=$(awk -v S=2 -v M=12 -v s=$RANDOM 'BEGIN{srand(s); print rand()*(M - S) + S}')
	lambda=$(awk -v s=$RANDOM 'BEGIN{srand(s); print rand()}')

#	echo $s $dth $bx $bt $lambda
	echo $s $np $bx $bt $lambda

#	trialfolder=$(printf "$output/s%02dp%02dw%02ddth%06.2fbx%4.2fbt%05.2fl%5.3f\n" \
#		$s $p $w $dth $bx $bt $lambda)
	trialfolder=$(printf "$output/s%02dp%02dw%02dnp%02dbx%4.2fbt%05.2fl%5.3f\n" \
		$s $p $w $np $bx $bt $lambda)

#	params=$(printf " -p %d -w %d --dth %06.2f --beta_x %4.2f --beta_t %05.2f --lambda %5.3f" \
#		$p $w $dth $bx $bt $lambda)
	params=$(printf " -p %d -w %d --npatches %d --beta_x %4.2f --beta_t %05.2f --lambda %5.3f" \
		$p $w $np $bx $bt $lambda)

	mpsnr=0
	nseqs=${#seqs[@]}
	ff=70
	lf=89
	if [ ! -d $trialfolder ]
	then
		for seq in ${seqs[@]}
		do
			echo  "$BIN/nlkalman-train.sh ${sf}${seq} $ff $lf $s $trialfolder \"$params\""
			psnr=$($BIN/nlkalman-train.sh ${sf}${seq} $ff $lf $s $trialfolder  "$params")
			mpsnr=$(echo "$mpsnr + $psnr/$nseqs" | bc -l)
			echo $psnr $mpsnr
		done
	fi
	
#	printf "%2d %2d %2d %06.2f %4.2f %05.2f %5.3f %7.4f\n" \
#		$s $p $w $dth $bx $bt $lambda $mpsnr >> $output/table

	printf "%2d %2d %2d %2d %4.2f %05.2f %5.3f %7.4f\n" \
		$s  $p  $w  $np $bx   $bt  $lambda $mpsnr >> $output/table

	rm $trialfolder/*.tif

done
