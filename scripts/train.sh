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
derf/bus_mono \
derf/foreman_mono \
derf/football_mono \
derf/tennis_mono \
derf/stefan_mono \
)
#tut/gsalesman \

# seq folder
sf='/home/pariasm/denoising/data/'

output=${1:-"trials"}

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
	W=$((4*s*s))
	whx=$(awk -v M=$W -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}')
	wht=$(awk -v M=$W -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}')
	whv=$(awk -v M=$W -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}')
#	whv=0

	lambtv=$(awk -v s=$RANDOM 'BEGIN{srand(s); print rand()}')

	trialfolder=$(printf "$output/rnlm.s%02d.p%02d.w%02d.whx%04d.wht%04d.whv%04d.lambtv%5.3f\n" \
		$s $p $w $whx $wht $whv $lambtv)

	params=$(printf " -p %d -w %d --whx %d --wht %d --whtv %d --lambtv %f" \
		$p $w $whx $wht $whv $lambtv)

	mpsnr=0
	nseqs=${#seqs[@]}
	nf=15
	if [ ! -d $trialfolder ]
	then
		for seq in ${seqs[@]}
		do
			echo "./vnlm_train.sh ${sf}${seq} 1 $nf $s $trialfolder \"$params\""
			./vnlm_train.sh ${sf}${seq} 1 $nf $s $trialfolder "$params"
			psnr=$(./vnlm_train.sh ${sf}${seq} 1 $nf $s $trialfolder "$params")
			mpsnr=$(echo "$mpsnr + $psnr/$nseqs" | bc -l)
			#echo $mpsnr
		done
	fi
	
	printf "%2d %2d %2d %4d %4d %4d %5.3f %7.4f\n" \
		$s $p $w $whx $wht $whv $lambtv $mpsnr >> $output/table

	rm $trialfolder/*.tif

done

#	r=$(awk -v m=1 -v M=10 -v s=$RANDOM 'BEGIN{srand(s); print int(m+rand()*(M-m+1))}')
