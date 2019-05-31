#!/bin/bash
# Runs nlkalman-seq.sh comparing the output with the ground truth

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
FPM=${6:-""} # filtering parameters
SPM=${7:-""} # smoothing parameters
OPM=${8:-"1 0.40 0.75 1 0.40 0.75"} # optical flow parameters

mkdir -p $OUT/
OUT=$OUT/

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
		$DIR/awgn $SIG $(printf $SEQ $i) $file
	fi
done

# run denoising script {{{1
$DIR/nlkalman-seq.sh "$OUT/%03d.tif" $FFR $LFR $SIG $OUT "$FPM" "$SPM" "$OPM"

# reset first frame for psnr computation {{{1
FFR=$((FFR+0))

# psnr for filter 1 {{{1
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	m=$($DIR/psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$($DIR/$DIR/plambda -c "$m sqrt" 2>/dev/null)
	PP[$i]=$($DIR/$DIR/plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
	SS=$($DIR/plambda -c "$m $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F1MSE=$SS
F1RMSE=$($DIR/plambda -c "$SS sqrt" 2>/dev/null)
F1PSNR=$($DIR/plambda -c "255 $F1RMSE / log10 20 *" 2>/dev/null)
echo "F1 - Frame RMSE " ${MM[*]}  > $OUT/measures
echo "F1 - Frame PSNR " ${PP[*]} >> $OUT/measures
echo "F1 - Total RMSE $F1RMSE"   >> $OUT/measures
echo "F1 - Total PSNR $F1PSNR"   >> $OUT/measures

# psnr for filter 2 {{{1
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	m=$($DIR/psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt2-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$($DIR/plambda -c "$m sqrt" 2>/dev/null)
	PP[$i]=$($DIR/plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
	SS=$($DIR/plambda -c "$m $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F2MSE=$SS
F2RMSE=$($DIR/plambda -c "$SS sqrt" 2>/dev/null)
F2PSNR=$($DIR/plambda -c "255 $F2RMSE / log10 20 *" 2>/dev/null)
echo "F2 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "F2 - Frame PSNR " ${PP[*]} >> $OUT/measures
echo "F2 - Total RMSE $F2RMSE"   >> $OUT/measures
echo "F2 - Total PSNR $F2PSNR"   >> $OUT/measures

# exit if no smoothing required {{{2
if [[ $SPM == "no" ]];
then 
	# convert tif to png (to save space)
	for i in $(seq $FFR $LFR);
	do
		ii=$(printf %03d $i)
		echo "$DIR/iion $OUT/flt1-${ii}.tif $OUT/flt1-${ii}.png && rm $OUT/flt1-${ii}.tif"
		echo "$DIR/iion $OUT/flt2-${ii}.tif $OUT/flt2-${ii}.png && rm $OUT/flt2-${ii}.tif"
	done | parallel

	printf "%f %f\n" $F1MSE $F2MSE;
	exit 0;
fi

# psnr for smoother {{{1
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	m=$($DIR/psnr.sh $(printf $SEQ $i) $(printf $OUT/"smo1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$($DIR/plambda -c "$m sqrt" 2>/dev/null)
	PP[$i]=$($DIR/plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
	SS=$($DIR/plambda -c "$m $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

S1MSE=$SS
S1RMSE=$($DIR/plambda -c "$SS sqrt" 2>/dev/null)
S1PSNR=$($DIR/plambda -c "255 $S1RMSE / log10 20 *" 2>/dev/null)
echo "S1 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "S1 - Frame PSNR " ${PP[*]} >> $OUT/measures
echo "S1 - Total RMSE $S1RMSE" >> $OUT/measures
echo "S1 - Total PSNR $S1PSNR" >> $OUT/measures

# convert tif to png (to save space) {{{2
for i in $(seq $FFR $LFR);
do
	ii=$(printf %03d $i)
	echo "$DIR/iion $OUT/flt1-${ii}.tif $OUT/flt1-${ii}.png && rm $OUT/flt1-${ii}.tif"
	echo "$DIR/iion $OUT/flt2-${ii}.tif $OUT/flt2-${ii}.png && rm $OUT/flt2-${ii}.tif"
	echo "$DIR/iion $OUT/smo1-${ii}.tif $OUT/smo1-${ii}.png && rm $OUT/smo1-${ii}.tif"
done | parallel

printf "%f %f %f\n" $F1MSE $F2MSE $S1MSE;

# vim:set foldmethod=marker:
