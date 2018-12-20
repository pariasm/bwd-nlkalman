#!/bin/bash
# Computes optical flow and occlusions for a sequence

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
OUT=$4 # output folder

mkdir -p $OUT

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
 
# compute optical flow {{{1
TVL1="$DIR/tvl1flow"
FSCALE=1
for i in $(seq $((FFR+1)) $LFR);
do
	file=$(printf $OUT"/%03d_b_s1.flo" $i)
	if [ ! -f $file ]
	then
		$TVL1 $(printf $SEQ $i) \
		      $(printf $SEQ $((i-1))) \
		      $file \
		      0 0.25 0.2 0.3 100 $FSCALE 0.5 5 0.01 0;
	fi
done
cp $(printf $OUT"/%03d_b_s1.flo" $((FFR+1))) $(printf $OUT"/%03d_b_s1.flo" $FFR)

# compute occlusion masks {{{1
for i in $(seq $FFR $LFR);
do
	file=$(printf $OUT"/occ_%03d_b_s1_t.75.png" $i)
	if [ ! -f $file ]
	then
		plambda $(printf $OUT"/%03d_b_s1.flo" $i) \
		    "x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs 0.75 > 255 *" \
		    -o $file
	fi
done

