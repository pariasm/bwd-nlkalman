#!/bin/bash
# Runs nlkalman filtering frame by frame

SEQ=$1 # filtered sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
FPM=${6:-""} # filtering parameters
SPM=${7:-""} # smoothing parameters

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

# filter first frame {{{1
FLT1="$OUT/flt1-%03d.tif"
FLT2="$OUT/flt2-%03d.tif"

i=$FFR
NLK="$DIR/nlkalman-flt"
echo $NLK -i $(printf $SEQ $i) -s $SIG $FPM \
 --flt11 $(printf $FLT1 $i) \
 --flt21 $(printf $FLT2 $i)
$NLK -i $(printf $SEQ $i) -s $SIG $FPM \
 --flt11 $(printf $FLT1 $i) \
 --flt21 $(printf $FLT2 $i)

# filter rest of sequence {{{1
TVL1="$DIR/tvl1flow"
FSCALE=0; DW=0.80; TH=0.75

FLOW="$OUT/bflo-%03d.flo"
OCCL="$OUT/bocc-%03d.png"

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

	# run filtering {{{2
	echo $NLK -i $(printf $SEQ $i) -s $SIG $FPM \
	 -o $(printf $FLOW $i) -k $(printf $OCCL $i)\
	 --flt10 $(printf $FLT1 $((i-1))) --flt11 $(printf $FLT1 $i) \
	 --flt20 $(printf $FLT2 $((i-1))) --flt21 $(printf $FLT2 $i)
	$NLK -i $(printf $SEQ $i) -s $SIG $FPM \
	 -o $(printf $FLOW $i) -k $(printf $OCCL $i)\
	 --flt10 $(printf $FLT1 $((i-1))) --flt11 $(printf $FLT1 $i) \
	 --flt20 $(printf $FLT2 $((i-1))) --flt21 $(printf $FLT2 $i)

done

# smooth sequence {{{1
TVL1="$DIR/tvl1flow"
FSCALE=0; DW=0.80; TH=0.75

# exit if no smoothing required
if [[ $SPM == "no" ]]; then exit 0; fi

NLK="$DIR/nlkalman-smo"
FLOW="$OUT/fflo-%03d.flo"
OCCL="$OUT/bocc-%03d.png"
SMO1="$OUT/smo1-%03d.tif"

# last frame
ln -srf $(printf $FLT2 $LFR) $(printf $SMO1 $LFR)

for i in $(seq $((LFR-1)) -1 $FFR)
do

	# compute forward optical flow {{{2
	file=$(printf $FLOW $i)
	if [ ! -f $file ]; then
		$TVL1 $(printf $FLT2 $i) \
		      $(printf $SMO1 $((i+1))) \
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

	# run smoothing {{{2
	echo $NLK --flt1 $(printf $FLT2 $i) --smo0 $(printf $SMO1 $((i+1))) \
	 -s $SIG $SPM -o $(printf $FLOW $i) -k $(printf $OCCL $i)\
	 --smo1 $(printf $SMO1 $i)
	$NLK --flt1 $(printf $FLT2 $i) --smo0 $(printf $SMO1 $((i+1))) \
	 -s $SIG $SPM -o $(printf $FLOW $i) -k $(printf $OCCL $i)\
	 --smo1 $(printf $SMO1 $i)

done

# vim:set foldmethod=marker:
