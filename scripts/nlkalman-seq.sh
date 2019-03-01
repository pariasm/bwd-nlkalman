#!/bin/bash
# Runs nlkalman filtering frame by frame

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
PRM=$6 # denoiser parameters

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

# check if forward optical flow is needed for smoothing
SMOO=0
if [[ $PRMS == *--s1_p* ]]; then SMOO=1; fi

FLT1="$OUT/flt1-%03d.tif"
FLT2="$OUT/flt2-%03d.tif"
FLOW="$OUT/%03d_b.flo"
OCCL="$OUT/occ_%03d_b.png"

# first frame {{{1
i=$FFR
NLK="$DIR/nlkalman-flt"
echo $NLK -i $(printf $SEQ $i) -s $SIG $PRM \
 --flt11 $(printf $FLT1 $i) \
 --flt21 $(printf $FLT2 $i)
$NLK -i $(printf $SEQ $i) -s $SIG $PRM \
 --flt11 $(printf $FLT1 $i) \
 --flt21 $(printf $FLT2 $i)

# rest of sequence {{{1
TVL1="$DIR/tvl1flow"
FSCALE=0; DW=0.80; TH=0.75

for i in $(seq $((FFR+1)) $LFR);
do

	# compute backward optical flow {{{2
	file=$(printf $FLOW $i)
	if [ ! -f $file ]; then
#		      $(printf $SEQ $((i-1))) \
		$TVL1 $(printf $SEQ $i) \
		      $(printf $FLT2 $((i-1))) \
		      $file \
		      0 0.25 0.2 $DW 100 $FSCALE 0.5 5 0.01 0;
	fi

	# backward occlusion masks {{{2
	file=$(printf $OCCL $i)
	if [ ! -f $file ]; then
		plambda $(printf $FLOW $i) \
		  "x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs 0.5 > 255 *" \
		  -o $file
	fi

	# denoise {{{2
	echo $NLK -i $(printf $SEQ $i) -s $SIG $PRM \
	 -o $(printf $FLOW $i) -k $(printf $OCCL $i)\
	 --flt10 $(printf $FLT1 $((i-1))) --flt11 $(printf $FLT1 $i) \
	 --flt20 $(printf $FLT2 $((i-1))) --flt21 $(printf $FLT2 $i)
	$NLK -i $(printf $SEQ $i) -s $SIG $PRM \
	 -o $(printf $FLOW $i) -k $(printf $OCCL $i)\
	 --flt10 $(printf $FLT1 $((i-1))) --flt11 $(printf $FLT1 $i) \
	 --flt20 $(printf $FLT2 $((i-1))) --flt21 $(printf $FLT2 $i)

done

# vim:set foldmethod=marker:
