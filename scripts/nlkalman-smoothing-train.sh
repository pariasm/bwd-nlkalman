#!/bin/bash
# This script is used by train.sh. It uses a set of precomputed 
# noisy sequences with their optical flow. To speed up training,
# the noise is already added, and the optical flow is already 
# computed.

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
PRM=$6 # denoiser parameters

# output folder
mkdir -p $OUT
 
# folders with pre-computed data
ORIG="$SEQ/%03d.png"
NISY="$SEQ/s${SIG}/%03d.tif"
BFLOW="$SEQ/s${SIG}/%03d_b.flo"
BOCCL="$SEQ/s${SIG}/occ_%03d_b.png"
FFLOW="$SEQ/s${SIG}/%03d_f.flo"
FOCCL="$SEQ/s${SIG}/occ_%03d_f.png"
#BFLOW="$SEQ/s${SIG}/tvl1_%03d_b.flo"
#BOCCL="$SEQ/s${SIG}/occl_%03d_b.png"
#FFLOW="$SEQ/s${SIG}/tvl1_%03d_f.flo"
#FOCCL="$SEQ/s${SIG}/occl_%03d_f.png"

# we assume that the binaries are in the same folder as the script
DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# run denoising
$DIR/nlkalman-bwd \
 -i $NISY --bflow $BFLOW --boccl $BOCCL --fflow $FFLOW --foccl $FOCCL \
 -f $FFR -l $LFR -s $SIG $PRM \
 --filt1 $OUT"/flt1-%03d.tif" \
 --filt2 $OUT"/flt2-%03d.tif" \
 --smoo1 $OUT"/smo1-%03d.tif"

# filter 1 : frame-by-frame psnr {{{1
for i in $(seq $FFR $LFR);
do
	# we remove a band of 10 pixels from each side of the frame
	MM[$i]=$(psnr.sh $(printf $ORIG $i) $(printf $OUT/"flt1-%03d.tif" $i) m 10 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F1 - Frame RMSE " ${MM[*]}  > $OUT/measures
echo "F1 - Frame PSNR " ${PP[*]} >> $OUT/measures

# filter 1 : global psnr (from 11th frame) {{{1
SS=0
n=0
for i in $(seq $((FFR+10)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F1RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
F1PSNR=$(plambda -c "255 $F1RMSE / log10 20 *" 2>/dev/null)
echo "F1 - Total RMSE $F1RMSE" >> $OUT/measures
echo "F1 - Total PSNR $F1PSNR" >> $OUT/measures

# filter 2 : frame-by-frame psnr {{{1
for i in $(seq $FFR $LFR);
do
	# we remove a band of 10 pixels from each side of the frame
	MM[$i]=$(psnr.sh $(printf $ORIG $i) $(printf $OUT/"flt2-%03d.tif" $i) m 10 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F2 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "F2 - Frame PSNR " ${PP[*]} >> $OUT/measures

# filter 2 : global psnr (from 11th frame) {{{1
SS=0
n=0
for i in $(seq $((FFR+10)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F2RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
F2PSNR=$(plambda -c "255 $F2RMSE / log10 20 *" 2>/dev/null)
echo "F2 - Total RMSE $F2RMSE" >> $OUT/measures
echo "F2 - Total PSNR $F2PSNR" >> $OUT/measures

# smoother : frame-by-frame psnr {{{1
for i in $(seq $FFR $((LFR-1)));
do
	# we remove a band of 10 pixels from each side of the frame
	MM[$i]=$(psnr.sh $(printf $ORIG $i) $(printf $OUT/"smo1-%03d.tif" $i) m 10 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "S1 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "S1 - Frame PSNR " ${PP[*]} >> $OUT/measures

# smoother : global psnr (from 11th frame) {{{1
SS=0
n=0
for i in $(seq $((FFR+10)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

S1RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
S1PSNR=$(plambda -c "255 $S1RMSE / log10 20 *" 2>/dev/null)
echo "S1 - Total RMSE $S1RMSE" >> $OUT/measures
echo "S1 - Total PSNR $S1PSNR" >> $OUT/measures

echo $F1PSNR $F2PSNR $S1PSNR


# vim:set foldmethod=marker:
