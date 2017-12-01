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
for i in $(seq $((FFR+1)) $LFR);
do
	file=$(printf $OUT"/%03d_b.flo" $i)
	if [ ! -f $file ]
	then
		$TVL1 $(printf $OUT"/%03d.tif" $i) \
				$(printf $OUT"/%03d.tif" $((i-1))) \
				$file \
				0 0.25 0.2 0.3 100 0.5 5 0.01 0; 
	fi
done
cp $(printf $OUT"/%03d_b.flo" $((FFR+1))) $(printf $OUT"/%03d_b.flo" $FFR)

# # downsample for optical flow {{{1
# for i in $(seq $FFR $LFR);
# do
# 	filein=$(printf $OUT/"%03d.tif" $i)
# 	file=$(printf $OUT/"h%03d.tif" $i)
# 	if [ ! -f $file ]
# 	then
# 		blur gauss 1.39 $filein | downsa f 2 - $file
# 	fi
# done
# 
# # compute optical flow {{{1
# TVL1="/home/pariasm/Work/optical_flow/algos/tvl1flow_3/tvl1flow"
# for i in $(seq $((FFR+1)) $LFR);
# do
# 	file=$(printf $OUT"/h%03d_b.flo" $i)
# 	if [ ! -f $file ]
# 	then
# 		$TVL1 $(printf $OUT"/h%03d.tif" $i) \
# 				$(printf $OUT"/h%03d.tif" $((i-1))) \
# 				$file \
# 				0 0.25 0.2 0.3 100 0.5 5 0.01 0; 
# 	fi
# done
# cp $(printf $OUT"/h%03d_b.flo" $((FFR+1))) $(printf $OUT"/h%03d_b.flo" $FFR)
# 
# # upscale optical flow {{{1
# for i in $(seq $FFR $LFR);
# do
# 	filein=$(printf $OUT/"h%03d_b.flo" $i)
# 	file=$(printf $OUT/"%03d_b.flo" $i)
# 	if [ ! -f $file ]
# 	then
# 		plambda $filein "2 x *" | upsa 2 2 - $file
# 	fi
# done

# run denoising {{{1
$DIR/test-dct \
 -i $OUT"/%03d.tif" -o $OUT"/%03d_b.flo" -f $FFR -l $LFR -s $SIG \
 -d $OUT"/deno_%03d.tif" $PRM

# compute psnr {{{1
for i in $(seq $FFR $LFR);
do
	# we remove a band of 10 pixels from each side of the frame
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"deno_%03d.tif" $i) m 10)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt")
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *")
done

echo "Frame RMSE " ${MM[*]}  > $OUT/measures
echo "Frame PSNR " ${PP[*]} >> $OUT/measures

# Global measures (from 4th frame)
SS=0
n=0
for i in $(seq $((FFR+3)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /")
	n=$((n+1))
done

RMSE=$(plambda -c "$SS sqrt")
PSNR=$(plambda -c "255 $RMSE / log10 20 *")
echo "Total RMSE $RMSE" >> $OUT/measures
echo "Total PSNR $PSNR" >> $OUT/measures


# vim:set foldmethod=marker:
