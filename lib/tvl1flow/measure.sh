#!/bin/bash
PATH_RMSE=~/nlbayes/build/bin/comp_psnr
PATH_VQM=~/tvl1flow_3/vqm

PO=${1}
PM=${2}
F=${3:-1}
L=${4:-150}
O=${5:-"./"}

./tvl1flow_sequence.sh $PO $F $L $O/tvl1_%03d_ori.flo
./tvl1flow_sequence.sh $PM $F $L $O/tvl1_%03d_den.flo
$PATH_VQM $PO $F $L $O/tvl1_%03d_ori.flo $O/diff_%03d_ori_ofori.tiff
$PATH_VQM $PM $F $L $O/tvl1_%03d_den.flo $O/diff_%03d_den_ofden.tiff
$PATH_VQM $PM $F $L $O/tvl1_%03d_ori.flo $O/diff_%03d_den_ofori.tiff
$PATH_VQM $PO $F $L $O/tvl1_%03d_den.flo $O/diff_%03d_ori_ofden.tiff
$PATH_RMSE -i "$O"/diff_%03d_ori_ofori.tiff -r "$O"/diff_%03d_den_ofori.tiff -f $F -l $((L-1))
$PATH_RMSE -i "$O"/diff_%03d_ori_ofden.tiff -r "$O"/diff_%03d_den_ofden.tiff -f $F -l $((L-1))
