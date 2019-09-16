#!/bin/bash
# Runs mlnlkalman-seq.sh comparing the output with the ground truth

SEQ=$1 # sequence path
FFR=$2 # first frame
LFR=$3 # last frame
SIG=$4 # noise standard dev.
OUT=$5 # output folder
FPM=${6:-""} # filtering parameters
SPM=${7:-""} # smoothing parameters
MPM=${8:-""} # multiscaler parameters

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
		$DIR/awgn $SIG $(printf $SEQ $i) $file
	fi
done

# run denoising script {{{1
$DIR/msnlkalman-lz3-seq.sh "$OUT/%03d.tif" $FFR $LFR $SIG $OUT "$FPM" "$SPM" $MPM

# psnr for multi-scale filter 1 {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$($DIR/psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$($DIR/plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$($DIR/plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F1 - Frame RMSE " ${MM[*]}  > $OUT/measures
echo "F1 - Frame PSNR " ${PP[*]} >> $OUT/measures

# global psnr
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$($DIR/plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F1MSE=$SS
F1RMSE=$($DIR/plambda -c "$SS sqrt" 2>/dev/null)
F1PSNR=$($DIR/plambda -c "255 $F1RMSE / log10 20 *" 2>/dev/null)
echo "F1 - Total RMSE $F1RMSE" >> $OUT/measures
echo "F1 - Total PSNR $F1PSNR" >> $OUT/measures

# psnr for multi-scale filter 2 {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$($DIR/psnr.sh $(printf $SEQ $i) $(printf $OUT/"flt2-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$($DIR/plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$($DIR/plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F2 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "F2 - Frame PSNR " ${PP[*]} >> $OUT/measures

# global psnr
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$($DIR/plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F2MSE=$SS
F2RMSE=$($DIR/plambda -c "$SS sqrt" 2>/dev/null)
F2PSNR=$($DIR/plambda -c "255 $F2RMSE / log10 20 *" 2>/dev/null)
echo "F2 - Total RMSE $F2RMSE" >> $OUT/measures
echo "F2 - Total PSNR $F2PSNR" >> $OUT/measures

# exit if no smoothing required
if [[ $SPM == "no" ]]; then printf "%f %f\n" $F1MSE $F2MSE; exit 0; fi

# psnr for multi-scale smoother {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$($DIR/psnr.sh $(printf $SEQ $i) $(printf $OUT/"smo1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$($DIR/plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$($DIR/plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "S1 - Frame RMSE " ${MM[*]} >> $OUT/measures
echo "S1 - Frame PSNR " ${PP[*]} >> $OUT/measures

# global psnr
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$($DIR/plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F2MSE=$SS
F2RMSE=$($DIR/plambda -c "$SS sqrt" 2>/dev/null)
F2PSNR=$($DIR/plambda -c "255 $F2RMSE / log10 20 *" 2>/dev/null)
echo "S1 - Total RMSE $F2RMSE" >> $OUT/measures
echo "S1 - Total PSNR $F2PSNR" >> $OUT/measures

# psnr for single-scale filter 1 {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$($DIR/psnr.sh $(printf $SEQ $i) $(printf $OUT/"ms0-flt1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$($DIR/plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$($DIR/plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F1 - Frame RMSE " ${MM[*]}  > $OUT/ss-measures
echo "F1 - Frame PSNR " ${PP[*]} >> $OUT/ss-measures

# global psnr
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$($DIR/plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F1MSE=$SS
F1RMSE=$($DIR/plambda -c "$SS sqrt" 2>/dev/null)
F1PSNR=$($DIR/plambda -c "255 $F1RMSE / log10 20 *" 2>/dev/null)
echo "F1 - Total RMSE $F1RMSE" >> $OUT/ss-measures
echo "F1 - Total PSNR $F1PSNR" >> $OUT/ss-measures

# psnr for single-scale filter 2 {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$($DIR/psnr.sh $(printf $SEQ $i) $(printf $OUT/"ms0-flt2-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$($DIR/plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$($DIR/plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "F2 - Frame RMSE " ${MM[*]} >> $OUT/ss-measures
echo "F2 - Frame PSNR " ${PP[*]} >> $OUT/ss-measures

# global psnr
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$($DIR/plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F2MSE=$SS
F2RMSE=$($DIR/plambda -c "$SS sqrt" 2>/dev/null)
F2PSNR=$($DIR/plambda -c "255 $F2RMSE / log10 20 *" 2>/dev/null)
echo "F2 - Total RMSE $F2RMSE" >> $OUT/ss-measures
echo "F2 - Total PSNR $F2PSNR" >> $OUT/ss-measures

# psnr for single-scale smoother {{{1
for i in $(seq $FFR $LFR);
do
	MM[$i]=$($DIR/psnr.sh $(printf $SEQ $i) $(printf $OUT/"ms0-smo1-%03d.tif" $i) m 0 2>/dev/null)
	MM[$i]=$($DIR/plambda -c "${MM[$i]} sqrt" 2>/dev/null)
	PP[$i]=$($DIR/plambda -c "255 ${MM[$i]} / log10 20 *" 2>/dev/null)
done

echo "S1 - Frame RMSE " ${MM[*]} >> $OUT/ss-measures
echo "S1 - Frame PSNR " ${PP[*]} >> $OUT/ss-measures

# global psnr
SS=0
n=0
for i in $(seq $((FFR+0)) $LFR);
do
	SS=$($DIR/plambda -c "${MM[$i]} 2 ^ $n $SS * + $((n+1)) /" 2>/dev/null)
	n=$((n+1))
done

F2MSE=$SS
F2RMSE=$($DIR/plambda -c "$SS sqrt" 2>/dev/null)
F2PSNR=$($DIR/plambda -c "255 $F2RMSE / log10 20 *" 2>/dev/null)
echo "S1 - Total RMSE $F2RMSE" >> $OUT/ss-measures
echo "S1 - Total PSNR $F2PSNR" >> $OUT/ss-measures

# vim:set foldmethod=marker:
