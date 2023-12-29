#!/bin/bash
module load gnuplot texlive

if [ "$1" = "" ]; then
    echo "usage: run_vef.sh <folder>"
    exit
fi

# cd $1
~/local/bin/vef_mixer -i $1/VEFT.main -o $1/output_trace.vef
rm -rf ~/VEF-Traces/python
PATH=$PATH:~/local/bin ~/local/bin/offline-vef-analysis.sh $1/output_trace.vef python
