singularity exec --bind /scratch --nv --overlay /scratch/xy2053/overlay-25GB-500K.ext3:rw /scratch/xy2053/ubuntu-20.04.3.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate dl_base
cd /scratch/xy2053/2025SP/2572_DeepLearning/codes/
python ./train.py
"