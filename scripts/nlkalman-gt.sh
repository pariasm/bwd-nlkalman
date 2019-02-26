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
		$TVL1 $(printf $OUT"/%03d.tif" $i) \
				$(printf $OUT"/%03d.tif" $((i-1))) \
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
			$TVL1 $(printf $OUT"/%03d.tif" $i) \
					$(printf $OUT"/%03d.tif" $((i+1))) \
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
	echo $DIR/nlkalman-bwd -i $OUT"/%03d.tif" \
	 --bflow $OUT"/%03d_b.flo" --boccl $OUT"/occ_%03d_b.png" \
	 --fflow $OUT"/%03d_f.flo" --foccl $OUT"/occ_%03d_f.png" \
	 -f $FFR -l $LFR -s $SIG $PRM \
	 --filt1 $OUT"/flt1-%03d.tif" \
	 --filt2 $OUT"/flt2-%03d.tif" \
	 --smoo1 $OUT"/smo1-%03d.tif"
	$DIR/nlkalman-bwd -i $OUT"/%03d.tif" \
	 --bflow $OUT"/%03d_b.flo" --boccl $OUT"/occ_%03d_b.png" \
	 --fflow $OUT"/%03d_f.flo" --foccl $OUT"/occ_%03d_f.png" \
	 -f $FFR -l $LFR -s $SIG $PRM \
	 --filt1 $OUT"/flt1-%03d.tif" \
	 --filt2 $OUT"/flt2-%03d.tif" \
	 --smoo1 $OUT"/smo1-%03d.tif"
else
	echo $DIR/nlkalman-bwd \
	 -i $OUT"/%03d.tif" -o $OUT"/%03d_b.flo" -k $OUT"/occ_%03d_b.png" \
	 -f $FFR -l $LFR -s $SIG $PRM \
	 --filt1 $OUT"/flt1-%03d.tif" \
	 --filt2 $OUT"/flt2-%03d.tif"
	$DIR/nlkalman-bwd \
	 -i $OUT"/%03d.tif" -o $OUT"/%03d_b.flo" -k $OUT"/occ_%03d_b.png" \
	 -f $FFR -l $LFR -s $SIG $PRM \
	 --filt1 $OUT"/flt1-%03d.tif" \
	 --filt2 $OUT"/flt2-%03d.tif"
fi

## # run denoising script
## $DIR/nlkalman.sh "$OUT/%03d.tif" $FFR $LFR $SIG $OUT "$PRM"
 

# filter 1 : frame-by-frame psnr {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F1 - Frame RMSE " ${MM[*]}  > $OUT/measures
echo "F1 - Frame PSNR " ${PP[*]} >> $OUT/measures

# filter 1 : global psnr {{{1
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F1MSE=$SS
F1RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
F1PSNR=$(plambda -c "255 $F1RMSE / log10 20 *" 2>/dev/null)
echo "F1 - Total RMSE $F1RMSE" >> $OUT/measures
echo "F1 - Total PSNR $F1PSNR" >> $OUT/measures

# filter 2 : frame-by-frame psnr {{{1
for i in $(seq $FFR $LFR);
do
	# we remove a band of 0 pixels from each side of the frame
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt2-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F2 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "F2 - Frame PSNR " ${PP[*]} >> $OUT/measures

# filter 2 : global psnr {{{1
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F2MSE=$SS
F2RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
F2PSNR=$(plambda -c "255 $F2RMSE / log10 20 *" 2>/dev/null)
echo "F2 - Total RMSE $F2RMSE" >> $OUT/measures
echo "F2 - Total PSNR $F2PSNR" >> $OUT/measures

# smoother : frame-by-frame psnr {{{1
if [ $SMOO ]
then
	for i in $(seq $FFR $((LFR-1)));
	do
		# we remove a band of 0 pixels from each side of the frame
		MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"smo1-%03d.tif" $i) m 0 2>/dev/null)
		MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
		PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
	done

	echo "S1 - Frame RMSE " ${MM[*]} >> $OUT/measures
	echo "S1 - Frame PSNR " ${PP[*]} >> $OUT/measures
fi

# smoother : global psnr {{{1
if [ $SMOO ]
then
	SS=0
	n=0
	for i in $(seq $((FFR+0)) $LFR);
	do
		SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
		n=$((n+1))
	done

	S1MSE=$SS
	S1RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
	S1PSNR=$(plambda -c "255 $S1RMSE / log10 20 *" 2>/dev/null)
	echo "S1 - Total RMSE $S1RMSE" >> $OUT/measures
	echo "S1 - Total PSNR $S1PSNR" >> $OUT/measures
fi

if [ $SMOO ]; then printf "%f %f %f\n" $F1MSE $F2MSE $S1MSE;
else               printf "%f %f\n"    $F1MSE $F2MSE;
fi


# vim:set foldmethod=marker:
