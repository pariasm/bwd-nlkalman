#! /bin/bash

# seq folder
sf='/home/pariasm/Remote/lime/denoising/data/'

ff=1
lf=20

seqs=(\
train-14/dataset/boxing \
train-14/dataset/choreography \
)
#train-14/dataset/demolition \
#train-14/dataset/grass-chopper \
#train-14/dataset/inflatable \
#train-14/dataset/juggle \
#train-14/dataset/kart-turn \
#train-14/dataset/lions \
#train-14/dataset/ocean-birds \
#train-14/dataset/old_town_cross \
#train-14/dataset/snow_mnt \
#train-14/dataset/swing-boy \
#train-14/dataset/varanus-tree \
#train-14/dataset/wings-turn \

s="$1"
f1_nx="$2"
f1_bx="$3"
f1_nt="$4"
f1_nta="$5"
f1_bt="$6"

f2_nx="${7:-0}"
f2_bx="${8:-0}"
f2_nt="${9:-0}"
f2_nta="${10:-0}"
f2_bt="${11:-0}"

s1_nt="${12:-0}"
s1_bt="${13:-0}"

f1_p=8
f1_sx=10
f1_st=5

if (( f2_nx > 0 )); then
	f2_p=8
	f2_sx=10
	f2_st=5
else
	f2_p=0
	f2_sx=0
	f2_st=0
fi

if (( s1_nt > 0 )); then
	s1_p=8
	s1_st=5
else
	s1_p=0
	s1_st=0
fi
s1_nta=$s1_nt

output=.
trialfolder=$(printf "$output/s%02d-nx%02dbx%05.2fnt%02dnta%02dbt%05.2f" \
	$s $f1_nx $f1_bx $f1_nt $f1_nta $f1_bt)
trialfolder=$(printf "${trialfolder}-nx%02dbx%05.2fnt%02dnta%02dbt%05.2f" \
	$f2_nx $f2_bx $f2_nt $f2_nta $f2_bt)
trialfolder=$(printf "${trialfolder}-nt%02dbt%05.2f\n" \
	$s1_nt $s1_bt)

params=$(printf " --f1_p %d --f1_sx %d --f1_st %d --f1_nx %d " $f1_p $f1_sx $f1_st $f1_nx)
params=$(printf "$params --f1_bx %f --f1_nt %d --f1_nt_agg %d --f1_bt %f " $f1_bx $f1_nt $f1_nta $f1_bt)
params=$(printf "$params --f2_p %d --f2_sx %d --f2_st %d --f2_nx %d " $f2_p $f2_sx $f2_st $f2_nx)
params=$(printf "$params --f2_bx %f --f2_nt %d --f2_nt_agg %d --f2_bt %f " $f2_bx $f2_nt $f2_nta $f2_bt)
params=$(printf "$params --s1_p %d --f1_st %d --s1_nt %d --s1_nt_agg %d --s1_bt %f" $s1_p $f1_st $s1_nt $s1_nta $s1_bt)

f1_mse=0
f2_mse=0
s1_mse=0
nseqs=${#seqs[@]}
for seq in ${seqs[@]}
do
	echo "./nlkalman-smoothing-train.sh ${sf}${seq} $ff $lf $s $trialfolder/$(basename $seq) \"$params\""
	out=$(./nlkalman-smoothing-train.sh ${sf}${seq} $ff $lf $s $trialfolder/$(basename $seq)  "$params")
	read -ra mse <<< "$out"
	f1_mse=$(echo "$f1_mse + ${mse[0]}/$nseqs" | bc -l)
	f2_mse=$(echo "$f2_mse + ${mse[1]}/$nseqs" | bc -l)
	s1_mse=$(echo "$s1_mse + ${mse[2]}/$nseqs" | bc -l)
	echo ${mse[0]} ${mse[1]} ${mse[2]}
	echo $f1_mse $f2_mse $s1_mse
done
printf "%02d %02d %05.2f %02d %02d %05.2f %02d %05.2f %02d %02d %05.2f %02d %05.2f %7.4f %7.4f %7.4f\n" \
	$s $f1_nx $f1_bx $f1_nt $f1_nta $f1_bt $f2_nx $f2_bx $f2_nt $f2_nta $f2_bt $s1_nt $s1_bt \
	$f1_mse $f2_mse $s1_mse
