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
 
# add noise {{{1
for i in $(seq $FFR $LFR);
do
	file=$(printf $OUT/"%03d.tif" $i)
	if [ ! -f $file ]
	then
		export SRAND=$RANDOM;
		awgn $SIG $(printf $SEQ $i) $file
	fi
done

# compute optical flow {{{1
TVL1="$DIR/tvl1flow"
FSCALE=1
DW=0.30
TH=0.75
#FSCALE=2
#DW=0.30
for i in $(seq $((FFR+1)) $LFR);
do
	file=$(printf $OUT"/tvl1_${FSCALE}_${DW}_%03d_b.flo" $i)
	if [ ! -f $file ]
	then
		$TVL1 $(printf $OUT"/%03d.tif" $i) \
				$(printf $OUT"/%03d.tif" $((i-1))) \
				$file \
				0 0.25 0.2 $DW 100 $FSCALE 0.5 5 0.01 0; 
	fi
done
cp $(printf $OUT"/tvl1_${FSCALE}_${DW}_%03d_b.flo" $((FFR+1))) $(printf $OUT"/tvl1_${FSCALE}_${DW}_%03d_b.flo" $FFR)

for i in $(seq $FFR $((LFR-1)));
do
	file=$(printf $OUT"/tvl1_${FSCALE}_${DW}_%03d_f.flo" $i)
	if [ ! -f $file ]
	then
		$TVL1 $(printf $OUT"/%03d.tif" $i) \
				$(printf $OUT"/%03d.tif" $((i+1))) \
				$file \
				0 0.25 0.2 $DW 100 $FSCALE 0.5 5 0.01 0; 
	fi
done
cp $(printf $OUT"/tvl1_${FSCALE}_${DW}_%03d_f.flo" $((LFR-1))) $(printf $OUT"/tvl1_${FSCALE}_${DW}_%03d_f.flo" $LFR)

# compute occlusion masks {{{1
TH=0.75
#TH=0.50
for i in $(seq $FFR $LFR);
do
	file=$(printf $OUT"/occl_${FSCALE}_${DW}_${TH}_%03d_b.png" $i)
	if [ ! -f $file ]
	then
		plambda $(printf $OUT"/tvl1_${FSCALE}_${DW}_%03d_b.flo" $i) \
				"x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs $TH > 255 *" \
				-o $file
	fi

	file=$(printf $OUT"/occl_${FSCALE}_${DW}_${TH}_%03d_f.png" $i)
	if [ ! -f $file ]
	then
		plambda $(printf $OUT"/tvl1_${FSCALE}_${DW}_%03d_f.flo" $i) \
				"x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs $TH > 255 *" \
				-o $file
	fi
done

# run denoising {{{1
$DIR/nlkalman-bwd \
 -i $OUT"/%03d.tif" -f $FFR -l $LFR -s $SIG \
 --bflow $OUT"/tvl1_${FSCALE}_${DW}_%03d_b.flo" \
 --fflow $OUT"/tvl1_${FSCALE}_${DW}_%03d_f.flo" \
 --boccl $OUT"/occl_${FSCALE}_${DW}_${TH}_%03d_b.png" \
 --foccl $OUT"/occl_${FSCALE}_${DW}_${TH}_%03d_f.png" \
 --filt1 $OUT"/flt1_${FSCALE}_${DW}_test3_%03d.tif" $PRM \
 --filt2 $OUT"/flt2_${FSCALE}_${DW}_test3_%03d.tif" $PRM \
 --smoo1 $OUT"/smo1_${FSCALE}_${DW}_test3_%03d.tif" $PRM

## # run denoising script
## $DIR/nlkalman.sh "$OUT/%03d.tif" $FFR $LFR $SIG $OUT "$PRM"
 
# compute psnr {{{1
for i in $(seq $FFR $((LFR-1)));
do
	# we remove a band of 10 pixels from each side of the frame
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt2_${FSCALE}_${DW}_test3_%03d.tif" $i) m 10)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt")
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *")
done

echo "Frame RMSE " ${MM[*]}  > $OUT/measures_${FSCALE}_${DW}_test3
echo "Frame PSNR " ${PP[*]} >> $OUT/measures_${FSCALE}_${DW}_test3

# Global measures (from 4th frame)
SS=0
n=0
for i in $(seq $((FFR+3)) $((LFR-1)));
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /")
	n=$((n+1))
done

RMSE=$(plambda -c "$SS sqrt")
PSNR=$(plambda -c "255 $RMSE / log10 20 *")
echo "Total RMSE $RMSE" >> $OUT/measures_${FSCALE}_${DW}_test3
echo "Total PSNR $PSNR" >> $OUT/measures_${FSCALE}_${DW}_test3


# vim:set foldmethod=marker:
