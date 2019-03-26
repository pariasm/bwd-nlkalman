#!/bin/bash
# Runs nlkalman-seq.sh comparing the output with the ground truth

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
FPM=${6:-""} # filtering parameters
SPM=${7:-""} # smoothing parameters

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

# run denoising script {{{1
$DIR/nlkalman-seq.sh "$OUT/%03d.tif" $FFR $LFR $SIG $OUT "$FPM" "$SPM"

# psnr for filter 1 {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F1 - Frame RMSE " ${MM[*]}  > $OUT/measures
echo "F1 - Frame PSNR " ${PP[*]} >> $OUT/measures

# global psnr
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F1MSE=$SS
F1RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
F1PSNR=$(plambda -c "255 $F1RMSE / log10 20 *" 2>/dev/null)
echo "F1 - Total RMSE $F1RMSE" >> $OUT/measures
echo "F1 - Total PSNR $F1PSNR" >> $OUT/measures

# psnr for filter 2 {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt2-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F2 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "F2 - Frame PSNR " ${PP[*]} >> $OUT/measures

# global psnr
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F2MSE=$SS
F2RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
F2PSNR=$(plambda -c "255 $F2RMSE / log10 20 *" 2>/dev/null)
echo "F2 - Total RMSE $F2RMSE" >> $OUT/measures
echo "F2 - Total PSNR $F2PSNR" >> $OUT/measures

# exit if no smoothing required
if [[ $SPM == "no" ]]; then printf "%f %f\n" $F1MSE $F2MSE; exit 0; fi

# psnr for smoother {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$(psnr.sh $(printf $SEQ $i) $(printf $OUT/"smo1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$(plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$(plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "S1 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "S1 - Frame PSNR " ${PP[*]} >> $OUT/measures

# global psnr
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$(plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

S1MSE=$SS
S1RMSE=$(plambda -c "$SS sqrt" 2>/dev/null)
S1PSNR=$(plambda -c "255 $S1RMSE / log10 20 *" 2>/dev/null)
echo "S1 - Total RMSE $S1RMSE" >> $OUT/measures
echo "S1 - Total PSNR $S1PSNR" >> $OUT/measures

# convert tif to png (to save space) {{{1
for i in $(seq $FFR $LFR);
do
	ii=$(printf %03d $i)
	echo "plambda $OUT/flt1-${ii}.tif x -o $OUT/flt1-${ii}.png && rm $OUT/flt1-${ii}.tif"
	echo "plambda $OUT/flt2-${ii}.tif x -o $OUT/flt2-${ii}.png && rm $OUT/flt2-${ii}.tif"
	echo "plambda $OUT/smo1-${ii}.tif x -o $OUT/smo1-${ii}.png && rm $OUT/smo1-${ii}.tif"
done | parallel

printf "%f %f %f\n" $F1MSE $F2MSE $S1MSE;

# vim:set foldmethod=marker:
