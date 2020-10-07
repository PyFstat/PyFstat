#!/usr/bin/env bash
# Get ephemeris files and export env variable to access them.
# Output path is accepted as an input; otherwise, falls back
# to ../.ephemeris.
# This script does NOT alter any other bash script.

if [ -z $1 ]
then
    this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    out_dir=${this_dir}/../.ephemeris/
    echo "No output path was given"
    echo "Falling back to ${out_dir}"
else
    out_dir=$1
    echo "Output path: ${out_dir}"
fi

mkdir -p ${out_dir}
wget -nc https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/earth00-40-DE405.dat.gz -P ${out_dir} 
wget -nc https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/sun00-40-DE405.dat.gz -P ${out_dir}
export LALPULSAR_DATADIR=${out_dir}
