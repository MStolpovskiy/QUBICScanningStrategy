#PBS -q debug
#PBS -l mppwidth=11520
##PBS -l walltime=01:00:00
#PBS -N scan
#PBS -j oe
#PBS -V

unset PYOPERATORS_NO_MPI

export NUM_TASKS_PER_NODE=2

export NUM_TASKS_PER_SOCKET=$((NUM_TASKS_PER_NODE / 2))
export NUM_NODES=$((PBS_NP / 24))
export NUM_TASKS=$((NUM_NODES * NUM_TASKS_PER_NODE))
export OMP_NUM_THREADS=$((24 / NUM_TASKS_PER_NODE))
cd $WD/ScanningStrategy
rnum=$RANDOM

aprun -n $NUM_TASKS -N $NUM_TASKS_PER_NODE -S $NUM_TASKS_PER_SOCKET -d $OMP_NUM_THREADS -cc depth python-mpi script_scan_ss_var_realiz.py -p $P -v $V -s $rnum
