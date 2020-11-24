export PYTHONNOUSERSITE=True
conda activate BERT4NLU
export PYTHONPATH=$PYTHONPATH:./



# process data
see src/dataset/dataset_utils/readmd.md



# train
# 单机单卡
CUDA_VISIBLE_DEVICES=2 python src/main.py

# 单机多卡
CUDA_VISIBLE_DEVICES=2,3 python src/main.py


# val
# modify src/config/config.py
CUDA_VISIBLE_DEVICES=2,3 python src/main.py


# tensorboard
tensorboard --logdir resource/records/personachat/run1/log --port 6006

