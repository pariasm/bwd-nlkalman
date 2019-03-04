#!/bin/bash
# Runs nlkalman filtering frame by frame

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
PRM=${6:-""}      # denoiser parameters
PYR_LVL=${7:--1}  # number of scales
PYR_DWN=${8:-2}   # downsampling factor
PYR_REC=${9:-0.7} # recomposition ratio

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

# we assume that the binaries are in the same folder as the script
DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# check if forward optical flow is needed for smoothing
SMOO=0
if [[ $PRMS == *--s1_p* ]]; then SMOO=1; fi

# determine number of levels based on the image size
if [ $PYR_LVL -eq -1 ];
then
	PIXELS=$(imprintf "%N" $(printf $SEQ $FFR))
	printf -v PIXELS "%.f" "$PIXELS"
	echo $PIXELS
	if   [ ${PIXELS} -lt  500000 ]; then PYR_LVL=1
	elif [ ${PIXELS} -lt 2000000 ]; then PYR_LVL=2
	elif [ ${PIXELS} -lt 8000000 ]; then PYR_LVL=3
	else                                 PYR_LVL=4
	fi
fi

echo "Scales: $PYR_LVL"

# create folders for scales
mkdir -p $OUT

NLK="$DIR/nlkalman-flt"
TVL1="$DIR/tvl1flow"
DECO="$DIR/decompose"
RECO="$DIR/recompose"
FSCALE=0; DW=0.80; TH=0.75
for i in $(seq $FFR $LFR);
do
	echo frame $i

	# compute pyramid
	$DECO $(printf "$SEQ" $i) "$OUT/ms" $PYR_LVL "-"$(printf %03d.tif $i)

	for ((l=PYR_LVL-1; l>=0; --l))
	do
		NSY=$(printf "$OUT/ms%d-%03d.tif"       $l $i)
		F11=$(printf "$OUT/ms%d-flt1-%03d.tif"  $l $i)
		F21=$(printf "$OUT/ms%d-flt2-%03d.tif"  $l $i)
		LSIG=$(bc <<< "scale=2; $SIG / ${PYR_DWN}^$l")

		if [ $i -gt $FFR ]; then
			F10=$(printf "$OUT/ms%d-flt1-%03d.tif"  $l $((i-1)))
			F20=$(printf "$OUT/ms%d-flt2-%03d.tif"  $l $((i-1)))
			FLW=$(printf "$OUT/ms%d-%03d-b.flo"     $l $i)
			OCC=$(printf "$OUT/ms%d-occ-%03d-b.png" $l $i)

			# compute backward optical flow {{{2
			if [ ! -f $FLW ]; then
				$TVL1 $NSY $F20 $FLW 0 0.25 0.2 $DW 100 $FSCALE 0.5 5 0.01 0;
			fi
		
			# backward occlusion masks {{{2
			if [ ! -f $OCC ]; then
				plambda $FLW \
				  "x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs 0.5 > 255 *" \
				  -o $OCC
			fi
		
			# denoise {{{2
#			echo $NLK -i $NSY -s $LSIG $PRM -o $FLW -k $OCC \
#				--flt10 $F10 --flt11 $F11 --flt20 $F20 --flt21 $F21
			$NLK -i $NSY -s $LSIG $PRM -o $FLW -k $OCC \
				--flt10 $F10 --flt11 $F11 --flt20 $F20 --flt21 $F21
		else
			# denoise {{{2
#			echo $NLK -i $NSY -s $LSIG $PRM --flt11 $F11 --flt21 $F21
			$NLK -i $NSY -s $LSIG $PRM --flt11 $F11 --flt21 $F21
		fi
	done

#	MSF1="$OUT/flt1-"$(printf %03.1f-%03d.tif $PYR_REC $i)
#	MSF2="$OUT/flt2-"$(printf %03.1f-%03d.tif $PYR_REC $i)
	MSF1="$OUT/flt1-"$(printf %03d.tif $i)
	MSF2="$OUT/flt2-"$(printf %03d.tif $i)
	$RECO "$OUT/ms" $PYR_LVL "-flt1-"$(printf %03d.tif $i) $MSF1 -c $PYR_REC
	$RECO "$OUT/ms" $PYR_LVL "-flt2-"$(printf %03d.tif $i) $MSF2 -c $PYR_REC
done

# vim:set foldmethod=marker:
