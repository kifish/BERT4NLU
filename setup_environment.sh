export PYTHONNOUSERSITE=True
~/anaconda3/bin/conda create -n BERT4NLU python=3.7

conda activate BERT4NLU
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

conda info --env

pip install tqdm

pip install transformers==3.3.1


conda install tensorboard -y

pip install nltk==3.5
# pip install sklearn


pip install psutil
conda install matplotlib
pip install jupyterlab


conda env export -n BERT4NLU -f BERT4NLU_ENV.yml
