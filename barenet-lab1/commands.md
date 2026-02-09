srun --account=csci_ga_3033-2025fa --partition=n1s8-t4-1 --gres=gpu:1 --time=1:00:00 --pty /bin/bash
singularity exec --bind /scratch --nv --overlay  /scratch/sr7463/overlay-25GB-500K.ext3:ro /scratch/sr7463/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash
source /ext3/env.sh
conda activate bdml_env
cd barenet-lab1
git pull
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
./test