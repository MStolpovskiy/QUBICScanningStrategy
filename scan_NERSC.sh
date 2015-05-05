#PBS -q debug # killable
#PBS -l mppwidth=1104 #2160
##PBS -l walltime=08:00:00
#PBS -N scan
#PBS -j oe
#PBS -V

unset PYOPERATORS_NO_MPI

export NUM_TASKS_PER_NODE=2

export NUM_TASKS_PER_SOCKET=$((NUM_TASKS_PER_NODE / 2))
export NUM_NODES=$((PBS_NP / 24))
export NUM_TASKS=$((NUM_NODES * NUM_TASKS_PER_NODE))
export OMP_NUM_THREADS=$((24 / NUM_TASKS_PER_NODE))
cd /project/projectdirs/qubic/stolpovs/ScanningStrategy

aprun -n $NUM_TASKS -N $NUM_TASKS_PER_NODE -S $NUM_TASKS_PER_SOCKET -d $OMP_NUM_THREADS -cc depth python-mpi script_oel.py
