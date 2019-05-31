#! /bin/bash
I=$1
O=$2
W=${3:-"a"}

# we assume that the binaries are in the same folder as the script
DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

MSE=$($DIR/plambda $I $O "x y - 2 ^" | $DIR/imprintf "%v" -)
RMSE=$($DIR/plambda -c "${MSE} sqrt")
PSNR=$($DIR/plambda -c "255 ${RMSE} / log10 20 *")

# clean
if [ $C -gt 0 ]; then rm -rf $F; fi

if   [[ $W == "p" ]]; then echo $PSNR
elif [[ $W == "r" ]]; then echo $RMSE
elif [[ $W == "m" ]]; then	echo $MSE
else
	echo RMSE: $RMSE
	echo PSNR: $PSNR
fi


