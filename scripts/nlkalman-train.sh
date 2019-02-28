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
FLOW="$SEQ/s${SIG}/tvl1_%03d_b.flo"
OCCL="$SEQ/s${SIG}/occl_%03d_b.png"

# we assume that the binaries are in the same folder as the script
DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# run denoising
$DIR/nlkalman-seq \
 -i $NISY -o $FLOW -k $OCCL -f $FFR -l $LFR -s $SIG \
 -d $OUT"/deno_%03d.tif" $PRM

# frame-by-frame psnr
for i in $(seq $FFR $LFR);
do
	# we remove a band of 10 pixels from each side of the frame
	MM[$i]=$(psnr.sh $(printf $ORIG $i) $(printf $OUT/"deno_%03d.tif" $i) m 10 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "Frame RMSE " ${MM[*]}  > $OUT/measures
echo "Frame PSNR " ${PP[*]} >> $OUT/measures

# global psnr (from 11th frame)
SS=0
n=0
for i in $(seq $((FFR+10)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
PSNR=$(plambda -c "255 $RMSE / log10 20 *" 2>/dev/null)
echo "Total RMSE $RMSE" >> $OUT/measures
echo "Total PSNR $PSNR" >> $OUT/measures
echo $PSNR


# vim:set foldmethod=marker:
