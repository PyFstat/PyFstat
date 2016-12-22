#!/bin/bash

. /home/gregory.ashton/lalsuite-install/etc/lalapps-user-env.sh
export PATH="/home/gregory.ashton/anaconda2/bin:$PATH"
export MPLCONFIGDIR=/home/gregory.ashton/.config/matplotlib

for ((n=0;n<10;n++))
do
/home/gregory.ashton/anaconda2/bin/python generate_data.py "$1" /local/user/gregory.ashton --no-template-counting --no-interactive
done
cp /local/user/gregory.ashton/MCResults_"$1".txt $(pwd)/CollectedOutput
