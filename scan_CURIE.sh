#!/bin/bash
#MSUB -r scan       # Job name
#MSUB -n 32         # Number of MPI processes
#MSUB -T 1800       # Job elapsed time limit
#MSUB -o scan_%I.o  # Standard output
#MSUB -e scan_%I.e  # Error output
#MSUB -q standard   # Choosing standard nodes
#MSUB -Q test       # The queue name
#MSUB -A gen6661    # Project

set -x
cd /ccc/cont003/home/gen6661/stolpovm/Qubic/ScanningStrategy
ccc_mprun -n ${BRIDGE_MSUB_NPROC} python script_oel.py
