#!/bin/bash
# Runs nlkalman filtering frame by frame

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
FPM=${6:-""}       # filtering parameters
SPM=${7:-""}       # filtering parameters
PYR_LVL=${8:--1}   # number of scales
PYR_REC=${9:-0.7}  # recomposition ratio
PYR_DWN=${10:-2}   # downsampling factor

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

# create output folder
mkdir -p $OUT

# multiscale filtering {{{1
NLKF="$DIR/nlkalman-flt"
TVL1="$DIR/tvl1flow"
DECO="$DIR/decompose"
RECO="$DIR/recompose"
FSCALE=1; DW=0.40; TH=0.75
for i in $(seq $FFR $LFR);
do
	echo filtering frame $i

	# compute pyramid
	$DECO $(printf "$SEQ" $i) "$OUT/ms" $PYR_LVL "-"$(printf %03d.tif $i)
	if [ $i -gt $FFR ]; then
		$DECO "$OUT/flt1-"$(printf %03d.tif $((i-1))) "$OUT/ma" $PYR_LVL "-flt1-"$(printf %03d.tif $((i-1)))
		$DECO "$OUT/flt2-"$(printf %03d.tif $((i-1))) "$OUT/ma" $PYR_LVL "-flt2-"$(printf %03d.tif $((i-1)))
	fi

	for ((l=PYR_LVL-1; l>=0; --l))
	do
		NSY=$(printf "$OUT/ms%d-%03d.tif"       $l $i)
		F11=$(printf "$OUT/ms%d-flt1-%03d.tif"  $l $i)
		F21=$(printf "$OUT/ms%d-flt2-%03d.tif"  $l $i)
		LSIG=$(bc <<< "scale=2; $SIG / ${PYR_DWN}^$l")

		if [ $i -gt $FFR ]; then
#			F10=$(printf "$OUT/ms%d-flt1-%03d.tif" $l $((i-1)))
#			F20=$(printf "$OUT/ms%d-flt2-%03d.tif" $l $((i-1)))
			F10=$(printf "$OUT/ma%d-flt1-%03d.tif" $l $((i-1)))
			F20=$(printf "$OUT/ma%d-flt2-%03d.tif" $l $((i-1)))
			FLW=$(printf "$OUT/ms%d-bflo-%03d.flo" $l $i)
			OCC=$(printf "$OUT/ms%d-bocc-%03d.png" $l $i)

			# compute backward optical flow {{{2
			if [ ! -f $FLW ]; then
				$TVL1 $NSY $F20 $FLW 0 0.25 0.2 $DW 100 $FSCALE 0.5 5 0.01 0;
			fi

			# backward occlusion masks {{{2
			if [ ! -f $OCC ]; then
				$DIR/plambda $FLW \
				  "x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs 0.5 > 255 *" \
				  -o $OCC
			fi

			# run filtering {{{2
#			echo $NLKF -i $NSY -s $LSIG $FPM -o $FLW -k $OCC \
#				--flt10 $F10 --flt11 $F11 --flt20 $F20 --flt21 $F21
			$NLKF -i $NSY -s $LSIG $FPM -o $FLW -k $OCC \
				--flt10 $F10 --flt11 $F11 --flt20 $F20 --flt21 $F21
		else
			# run filtering {{{2
#			echo $NLKF -i $NSY -s $LSIG $FPM --flt11 $F11 --flt21 $F21
			$NLKF -i $NSY -s $LSIG $FPM --flt11 $F11 --flt21 $F21
		fi
	done

#	MSF1="$OUT/flt1-"$(printf %03.1f-%03d.tif $PYR_REC $i)
#	MSF2="$OUT/flt2-"$(printf %03.1f-%03d.tif $PYR_REC $i)
	MSF1="$OUT/flt1-"$(printf %03d.tif $i)
	MSF2="$OUT/flt2-"$(printf %03d.tif $i)
	$RECO "$OUT/ms" $PYR_LVL "-flt1-"$(printf %03d.tif $i) $MSF1 -c $PYR_REC
	$RECO "$OUT/ms" $PYR_LVL "-flt2-"$(printf %03d.tif $i) $MSF2 -c $PYR_REC
done

# multiscale smoothing {{{1

# exit if no smoothing required
if [[ $SPM == "no" ]]; then exit 0; fi

# last frame
for l in $(seq $((PYR_LVL-1)) -1 0)
do
	cp -sf $(printf      "ms%d-flt2-%03d.tif"  $l $LFR) \
	       $(printf "$OUT/ms%d-smo1-%03d.tif"  $l $LFR)
#	cp -srf $(printf "$OUT/ms%d-flt2-%03d.tif"  $l $LFR) \
#	        $(printf "$OUT/ms%d-smo1-%03d.tif"  $l $LFR)
done
cp -sf $(printf "flt2-%03d.tif" $LFR) $(printf "$OUT/smo1-%03d.tif" $LFR)

NLKS="$DIR/nlkalman-smo"
FSCALE=1; DW=0.40; TH=0.75
for i in $(seq $((LFR-1)) -1 $FFR)
do
	echo smoothing frame $i

#	$DECO "$OUT/smo1-"$(printf %03d.tif $((i+1))) "$OUT/ma" $PYR_LVL "-smo1-"$(printf %03d.tif $((i+1)))

	for ((l=PYR_LVL-1; l>=0; --l))
	do
		F1=$(printf "$OUT/ms%d-flt2-%03d.tif"  $l $i)
		S1=$(printf "$OUT/ms%d-smo1-%03d.tif"  $l $i)
		S0=$(printf "$OUT/ms%d-smo1-%03d.tif"  $l $((i+1)))
#		S0=$(printf "$OUT/ma%d-smo1-%03d.tif"  $l $((i+1)))
		LSIG=$(bc <<< "scale=2; $SIG / ${PYR_DWN}^$l")

		FLW=$(printf "$OUT/ms%d-fflo-%03d.flo" $l $i)
		OCC=$(printf "$OUT/ms%d-focc-%03d.png" $l $i)

		# compute forward optical flow {{{2
		if [ ! -f $FLW ]; then
			$TVL1 $F1 $S0 $FLW 0 0.25 0.2 $DW 100 $FSCALE 0.5 5 0.01 0;
		fi

		# forward occlusion masks {{{2
		if [ ! -f $OCC ]; then
			$DIR/plambda $FLW \
			  "x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs 0.5 > 255 *" \
			  -o $OCC
		fi

		# smoothing {{{2
		$NLKS --flt1 $F1 --smo0 $S0 -o $FLW -k $OCC -s $SIG $SPM --smo1 $S1
	done

#	MSS1="$OUT/smo1-"$(printf %03.1f-%03d.tif $PYR_REC $i)
	MSS1="$OUT/smo1-"$(printf %03d.tif $i)
	$RECO "$OUT/ms" $PYR_LVL "-smo1-"$(printf %03d.tif $i) $MSS1 -c $PYR_REC
done

# vim:set foldmethod=marker:
