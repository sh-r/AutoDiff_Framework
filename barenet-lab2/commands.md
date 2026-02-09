ssh burst
srun --account=csci_ga_3033-2025fa --partition=n1s8-t4-1 --gres=gpu:1 --time=1:00:00 --pty /bin/bash
singularity exec --bind /scratch --nv --overlay  /scratch/sr7463/overlay-25GB-500K.ext3:ro /scratch/sr7463/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash
source /ext3/env.sh
conda activate bdml_env
cd /scratch/sr7463
cd barenet-lab2
git pull
pytest -v test_ag_tensor.py::test_add