#!/bin/bash
# Evals vnlm using ground truth

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
PRM=$6 # denoiser parameters
MSPRM=$7 # multiscaler parameters

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

# run denoising script {{{1
$DIR/msnlkalman-seq.sh "$OUT/%03d.tif" $FFR $LFR $SIG $OUT "$PRM" $MSPRM

# multi-scale filter 1 : frame-by-frame psnr {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F1 - Frame RMSE " ${MM[*]}  > $OUT/measures
echo "F1 - Frame PSNR " ${PP[*]} >> $OUT/measures

# multi-scale filter 1 : global psnr {{{1
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

# multi-scale filter 2 : frame-by-frame psnr {{{1
for i in $(seq $FFR $LFR);
do
	# we remove a band of 0 pixels from each side of the frame
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt2-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F2 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "F2 - Frame PSNR " ${PP[*]} >> $OUT/measures

# multi-scale filter 2 : global psnr {{{1
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

# single-scale filter 1 : frame-by-frame psnr {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"ms0-flt1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F1 - Frame RMSE " ${MM[*]}  > $OUT/ss-measures
echo "F1 - Frame PSNR " ${PP[*]} >> $OUT/ss-measures

# single-scale filter 1 : global psnr {{{1
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
echo "F1 - Total RMSE $F1RMSE" >> $OUT/ss-measures
echo "F1 - Total PSNR $F1PSNR" >> $OUT/ss-measures

# single-scale filter 2 : frame-by-frame psnr {{{1
for i in $(seq $FFR $LFR);
do
	# we remove a band of 0 pixels from each side of the frame
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"ms0-flt2-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F2 - Frame RMSE " ${MM[*]} >> $OUT/ss-measures
echo "F2 - Frame PSNR " ${PP[*]} >> $OUT/ss-measures

# single-scale filter 2 : global psnr {{{1
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
echo "F2 - Total RMSE $F2RMSE" >> $OUT/ss-measures
echo "F2 - Total PSNR $F2PSNR" >> $OUT/ss-measures


# vim:set foldmethod=marker:
