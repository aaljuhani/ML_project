# histosearch
Machine Learning Model to classify detect Mitoses cell structures

## Installation Instructions
conda env create -n histosearch -f histosearch.yml

#### then activate the conda virtual environment
source activate label-coach


## To Run the Model
python recognize.py --data <data_DIR> --gt <groundtruth_DIR> --lbn_points <Local Binary Pattern Points Param> --lbn_radius <Local Binary Pattern Radius Param>

example:
python recognize.py --data data/tiles --gt data/groundtruth --lbn_points 8 --lbn_radius 1
