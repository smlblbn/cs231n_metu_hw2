# Get BBP dataset Part_1
#wget https://archive.ics.uci.edu/ml/machine-learning-databases/00340/data.zip
unzip data.zip
shopt -s extglob
rm -r !(Part_1.mat|get_dataset.sh)