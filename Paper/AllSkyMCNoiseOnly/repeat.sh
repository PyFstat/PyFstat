#!/bin/bash

. /home/gregory.ashton/lalsuite-install/etc/lalapps-user-env.sh
export PATH="/home/gregory.ashton/anaconda2/bin:$PATH"
export MPLCONFIGDIR=/home/gregory.ashton/.config/matplotlib

rm /local/user/gregory.ashton/MCResults*txt 
for ((n=0;n<100;n++))
do
/home/gregory.ashton/anaconda2/bin/python generate_data.py "$1" /local/user/gregory.ashton --quite --no-template-counting
done
cp /local/user/gregory.ashton/MCResults*txt /home/gregory.ashton/PyFstat/Paper/AllSkyMCNoiseOnly/CollectedOutput
