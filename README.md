# histosearch
Machine Learning Model to classify detect Mitoses cell structures

## Installation Instructions

#### Prerequierment:
Numpy: conda install -c conda-forge numpy 
npm: conda install -c conda-forge nodejs 
 


#### create conda env
conda env create -n histosearch 


#### then activate the conda virtual environment
source activate histosearch


## To Run the Model
python recognize.py --data <data_DIR> --gt <groundtruth_DIR> --lbn_points <Local Binary Pattern Points Param> --lbn_radius <Local Binary Pattern Radius Param>

example:
python recognize.py --data data/tiles --gt data/groundtruth --lbn_points 8 --lbn_radius 1
