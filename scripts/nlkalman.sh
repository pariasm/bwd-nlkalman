#!/bin/bash
# Evals vnlm using ground truth

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
PRM=$6 # denoiser parameters

mkdir -p $OUT/s$SIG
OUT=$OUT/s$SIG

# we assume that the binaries are in the same folder as the script
DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# error checking {{{1
for i in $(seq $FFR $LFR);
do
	file=$(printf $SEQ $i)
	if [ ! -f $file ]
	then
		echo ERROR: $file not found
		exit 1
	fi
done

# check if forward optical flow is needed for smoothing
SMOO=0
if [[ $PRMS == *--s1_p* ]]; then SMOO=1; fi

# compute backward optical flow {{{1
TVL1="$DIR/tvl1flow"
FSCALE=2
for i in $(seq $((FFR+1)) $LFR);
do
	file=$(printf $OUT"/%03d_b.flo" $i)
	if [ ! -f $file ]
	then
		$TVL1 $(printf $SEQ $i) \
				$(printf $SEQ $((i-1))) \
				$file \
				0 0.25 0.2 0.3 100 $FSCALE 0.5 5 0.01 0;
	fi
done
cp $(printf $OUT"/%03d_b.flo" $((FFR+1))) $(printf $OUT"/%03d_b.flo" $FFR)

# compute forward optical flow {{{1
if [ $SMOO ]
then
	for i in $(seq $((FFR)) $((LFR-1)));
	do
		file=$(printf $OUT"/%03d_f.flo" $i)
		if [ ! -f $file ]
		then
			$TVL1 $(printf $SEQ $i) \
					$(printf $SEQ $((i+1))) \
					$file \
					0 0.25 0.2 0.3 100 $FSCALE 0.5 5 0.01 0;
		fi
	done
	cp $(printf $OUT"/%03d_f.flo" $((LFR-1))) $(printf $OUT"/%03d_f.flo" $LFR)
fi

# compute occlusion masks {{{1
for i in $(seq $FFR $LFR);
do
	# backward occlusion masks
	file=$(printf $OUT"/occ_%03d_b.png" $i)
	if [ ! -f $file ]
	then
		plambda $(printf $OUT"/%03d_b.flo" $i) \
				"x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs 0.5 > 255 *" \
				-o $file
	fi

	# forward occlusion masks
	if [ $SMOO ]
	then
		file=$(printf $OUT"/occ_%03d_f.png" $i)
		if [ ! -f $file ]
		then
			plambda $(printf $OUT"/%03d_f.flo" $i) \
					"x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs 0.5 > 255 *" \
					-o $file
		fi
	fi
done

# run denoising {{{1
if [ $SMOO ]
then
	echo $DIR/nlkalman-bwd -i $SEQ \
	 --bflow $OUT"/%03d_b.flo" --boccl $OUT"/occ_%03d_b.png" \
	 --fflow $OUT"/%03d_f.flo" --foccl $OUT"/occ_%03d_f.png" \
	 -f $FFR -l $LFR -s $SIG $PRM \
	 --filt1 $OUT"/flt1-%03d.tif" \
	 --filt2 $OUT"/flt2-%03d.tif" \
	 --smoo1 $OUT"/smo1-%03d.tif"
	$DIR/nlkalman-bwd -i $SEQ \
	 --bflow $OUT"/%03d_b.flo" --boccl $OUT"/occ_%03d_b.png" \
	 --fflow $OUT"/%03d_f.flo" --foccl $OUT"/occ_%03d_f.png" \
	 -f $FFR -l $LFR -s $SIG $PRM \
	 --filt1 $OUT"/flt1-%03d.tif" \
	 --filt2 $OUT"/flt2-%03d.tif" \
	 --smoo1 $OUT"/smo1-%03d.tif"
else
	echo $DIR/nlkalman-bwd \
	 -i $SEQ -o $OUT"/%03d_b.flo" -k $OUT"/occ_%03d_b.png" \
	 -f $FFR -l $LFR -s $SIG $PRM \
	 --filt1 $OUT"/flt1-%03d.tif" \
	 --filt2 $OUT"/flt2-%03d.tif"
	$DIR/nlkalman-bwd \
	 -i $SEQ -o $OUT"/%03d_b.flo" -k $OUT"/occ_%03d_b.png" \
	 -f $FFR -l $LFR -s $SIG $PRM \
	 --filt1 $OUT"/flt1-%03d.tif" \
	 --filt2 $OUT"/flt2-%03d.tif"
fi

# vim:set foldmethod=marker:
