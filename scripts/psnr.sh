#! /bin/bash
I=$1
O=$2
W=${3:-"a"}
C=${4:-0}

if [ $C -gt 0 ]
then
	F=$(mktemp /tmp/psnr-crop.XXXXXXXXX -d)
	w=$(imprintf "%w" $I)
	h=$(imprintf "%h" $I)
	crop $C $C $((w-C)) $((h-C)) $I $F/i.tif
	crop $C $C $((w-C)) $((h-C)) $O $F/o.tif
	I=$F/i.tif
	O=$F/o.tif
fi

MSE=`plambda $I $O "x y - 2 ^" | imprintf "%v" -`
RMSE=`plambda -c "${MSE} sqrt"`
PSNR=`plambda -c "255 ${RMSE} / log10 20 *"`

# clean
if [ $C -gt 0 ]
then
	rm -rf $F
fi

if [[ $W == "p" ]]; then
	echo $PSNR
elif [[ $W == "r" ]]; then
	echo $RMSE
elif [[ $W == "m" ]]; then
	echo $MSE
else
	echo RMSE: $RMSE
	echo PSNR: $PSNR
fi


