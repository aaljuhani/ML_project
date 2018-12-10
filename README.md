# histosearch
Machine Learning Model to classify detect Mitoses cell structures

# Dataset
#### DatasetURL for TUPAC challenge:http://tupac.tue-image.nl/node/3Direct URL for the Mitosis 
#### Dataset:https://drive.google.com/drive/u/0/folders/0B--ztKW0d17XbXNPQVY5VWxiZkU#list

## Installation Instructions

#### Prerequierment:
Numpy: conda install -c conda-forge numpy 
npm: conda install -c conda-forge nodejs 
 


#### create conda env
conda env create -n histosearch 


#### then activate the conda virtual environment
source activate histosearch


## To Run the Model
python recognize.py --data <data_DIR> --gt <groundtruth_DIR>

example:
python recognize.py --data data/tiles --gt data/groundtruth
