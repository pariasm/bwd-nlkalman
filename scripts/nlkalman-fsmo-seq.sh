#!/bin/bash
# Runs nlkalman filtering frame by frame. After filtering, applies a smoothing
# step but forward

SEQ=$1 # filtered sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
FPM=${6:-""} # filtering parameters
SPM=${7:-""} # smoothing parameters
OPM=${8:-"1 0.25 0.75 1 0.25 0.75"} # optical flow parameters

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

# binaries
NLK="$DIR/nlkalman-flt"
NLS="$DIR/nlkalman-smo"
TVL1="$DIR/tvl1flow"

# optical flow parameters
read -ra O <<< "$OPM"
FSCALE1=${O[0]}; DW1=${O[1]}; TH1=${O[2]}; NPROC=8
FSCALE2=${O[3]}; DW2=${O[4]}; TH2=${O[5]};
# nproc tau lambda theta nscales fscale zfactor nwarps epsilon verbos
OFPRMS="$NPROC 0 $DW1 0 0 $FSCALE1";

# filenames
FLOW="$OUT/bflo-%03d.flo"
OCCL="$OUT/bocc-%03d.png"
FLT1="$OUT/flt1-%03d.tif"
FLT2="$OUT/flt2-%03d.tif"
SMO1="$OUT/smo1-%03d.tif"

# filter first frame {{{1
i=$FFR
$NLK -i $(printf $SEQ $i) -s $SIG $FPM \
 --flt11 $(printf $FLT1 $i) \
 --flt21 $(printf $FLT2 $i)

# init smoothed sequence
if [[ $SPM != "no" ]];
then
	cp $(printf $FLT2 $FFR) $(printf $SMO1 $FFR)
fi


# filter rest of sequence {{{1
for i in $(seq $((FFR+1)) $LFR);
do

	# compute backward optical flow {{{2
	F=$(printf $FLOW $i)
	if [ ! -f $F ]; then
		$TVL1 $(printf $SEQ $i) \
		      $(printf $FLT2 $((i-1))) \
		      $F $OFPRMS;
	fi

	# backward occlusion masks {{{2
	O=$(printf $OCCL $i)
	if [ ! -f $O ]; then
		$DIR/plambda $F \
		  "x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs $TH1 > 255 *" \
		  -o $O
	fi

	# run filtering {{{2
	$NLK -i $(printf $SEQ $i) -s $SIG $FPM --f2_p 0 -o $F -k $O \
	 --flt10 $(printf $FLT1 $((i-1))) --flt11 $(printf $FLT1 $i)

	$NLK -i $(printf $SEQ $i) -s $SIG $FPM --f1_p 0 -o $F -k $O \
	 --flt11 $(printf $FLT1 $i) \
	 --flt20 $(printf $FLT2 $((i-1))) --flt21 $(printf $FLT2 $i)

	# run forward smoothing {{{2
	if [[ $SPM != "no" ]]; then
		$NLS --flt1 $(printf $FLT2 $i) --smo0 $(printf $SMO1 $((i-1))) \
		 -s $SIG $SPM -o $F -k $O --smo1 $(printf $SMO1 $i)
	fi

done

# vim:set foldmethod=marker:
