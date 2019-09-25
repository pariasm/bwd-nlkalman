#!/bin/bash
# Runs nlkalman filtering frame by frame

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
FPM=${6:-""}       # filtering parameters
SPM=${7:-""}       # smoothing parameters
OPM=${8:-"1 0.25 0.75 1 0.25 0.75"} # optical flow parameters
PYR_LVL=${9:--1}   # number of scales
PYR_REC=${10:-0.7} # recomposition ratio

PYR_DWN=2   # downsampling factor


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

	echo "Scales: $PYR_LVL"
fi

# create output folder
mkdir -p $OUT

read -ra O <<< "$OPM"
FSCALE1=${O[0]}; DW1=${O[1]}; TH1=${O[2]}; NPROC=1
FSCALE2=${O[3]}; DW2=${O[4]}; TH2=${O[5]};

# multiscale filtering {{{1
NLKF="$DIR/nlkalman-flt"
TVL1="$DIR/tvl1flow"
PATH_MULTISCALE="$DIR/../../lib/ms-lanczos3"
DECO="$PATH_MULTISCALE/lanczos3_decompose.m"
RECO="$PATH_MULTISCALE/lanczos3_recompose.m"
for i in $(seq $FFR $LFR);
do
#	MSF1="$OUT/flt1-"$(printf %03.1f-%03d.tif $PYR_REC $i)
#	MSF2="$OUT/flt2-"$(printf %03.1f-%03d.tif $PYR_REC $i)
	S0F1=$(printf "$OUT/ms0-flt1-%03d.tif" $i)
	S0F2=$(printf "$OUT/ms0-flt2-%03d.tif" $i)
	MSF1=$(printf "$OUT/flt1-%03d.tif" $i)
	MSF2=$(printf "$OUT/flt2-%03d.tif" $i)
#	if [ ! -f $MSF2 ] || [ ! -f $MSF1 ] || [ ! -f $S0F2 ] || [ ! -f $S0F1 ]
#	then
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
				read -ra O <<< "$OPM"
				FSCALE=${O[0]}; DW=${O[1]}; TH=${O[2]}; NPROC=2
				if [ ! -f $FLW ]; then
					OFPRMS="$NPROC 0 $DW 0 0 $FSCALE";
					$TVL1 $NSY $F20 $FLW $OFPRMS;
				fi

				# backward occlusion masks {{{2
				if [ ! -f $OCC ]; then
					$DIR/plambda $FLW \
					  "x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs $TH > 255 *" \
					  -o $OCC
				fi

				# run filtering {{{2
	#			echo $NLKF -i $NSY -s $LSIG $FPM -o $FLW -k $OCC \
	#				--flt10 $F10 --flt11 $F11 --flt20 $F20 --flt21 $F21
				$NLKF -i $NSY -s $LSIG $FPM -o $FLW -k $OCC \
					--flt10 $F10 --flt11 $F11 --flt20 $F20 --flt21 $F21

				rm $FLW $OCC $F10 $F20
			else
				# run filtering {{{2
	#			echo $NLKF -i $NSY -s $LSIG $FPM --flt11 $F11 --flt21 $F21
				$NLKF -i $NSY -s $LSIG $FPM --flt11 $F11 --flt21 $F21
			fi
		done

		$RECO $MSF1 "$OUT/ms" $PYR_LVL "-flt1-"$(printf %03d.tif $i) $PYR_REC
		$RECO $MSF2 "$OUT/ms" $PYR_LVL "-flt2-"$(printf %03d.tif $i) $PYR_REC
#	fi

done

# multiscale smoothing {{{1

# exit if no smoothing required
if [[ $SPM == "no" ]]; then exit 0; fi

# last frame
for l in $(seq $((PYR_LVL-1)) -1 0)
do
	cp $(printf "$OUT/ms%d-flt2-%03d.tif"  $l $LFR) \
	   $(printf "$OUT/ms%d-smo1-%03d.tif"  $l $LFR)
done
cp $(printf "$OUT/flt2-%03d.tif" $LFR) $(printf "$OUT/smo1-%03d.tif" $LFR)

NLKS="$DIR/nlkalman-smo"
for i in $(seq $((LFR-1)) -1 $FFR)
do
#	MSS1="$OUT/smo1-"$(printf %03.1f-%03d.tif $PYR_REC $i)
	S0S1=$(printf "$OUT/ms0-smo1-%03d.tif" $i)
	MSS1=$(printf "$OUT/smo1-%03d.tif" $i)
#	if [ ! -f $MSS1 ] || [ ! -f $S0S1 ]
#	then
		echo smoothing frame $i

		$DECO "$OUT/smo1-"$(printf %03d.tif $((i+1))) "$OUT/ma" $PYR_LVL "-smo1-"$(printf %03d.tif $((i+1)))

		for ((l=PYR_LVL-1; l>=0; --l))
		do
			F1=$(printf "$OUT/ms%d-flt2-%03d.tif"  $l $i)
			S1=$(printf "$OUT/ms%d-smo1-%03d.tif"  $l $i)
	#		S0=$(printf "$OUT/ms%d-smo1-%03d.tif"  $l $((i+1)))
			S0=$(printf "$OUT/ma%d-smo1-%03d.tif"  $l $((i+1)))
			LSIG=$(bc <<< "scale=2; $SIG / ${PYR_DWN}^$l")

			FLW=$(printf "$OUT/ms%d-fflo-%03d.flo" $l $i)
			OCC=$(printf "$OUT/ms%d-focc-%03d.png" $l $i)

			# compute forward optical flow {{{2
			read -ra O <<< "$OPM"
			FSCALE=${O[3]}; DW=${O[4]}; TH=${O[5]}; NPROC=2
			if [ ! -f $FLW ]; then
				OFPRMS="$NPROC 0 $DW 0 0 $FSCALE";
				$TVL1 $F1 $S0 $FLW $OFPRMS;
			fi

			# forward occlusion masks {{{2
			if [ ! -f $OCC ]; then
				$DIR/plambda $FLW \
				  "x(0,0)[0] x(-1,0)[0] - x(0,0)[1] x(0,-1)[1] - + fabs $TH > 255 *" \
				  -o $OCC
			fi

			# smoothing {{{2
			$NLKS --flt1 $F1 --smo0 $S0 -o $FLW -k $OCC -s $SIG $SPM --smo1 $S1

			rm $FLW $OCC $S0
		done

		$RECO $MSS1 "$OUT/ms" $PYR_LVL "-smo1-"$(printf %03d.tif $i) $PYR_REC
#	fi
done

# vim:set foldmethod=marker:
