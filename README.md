# histoSearch
Machine Learning Model to classify detect Mitoses cell structures

# Github: https://github.com/aaljuhani/histosearch

# Dataset
URL for TUPAC challenge: http://tupac.tue-image.nl/node/3Direct  
URL for the Mitosis Dataset: https://drive.google.com/drive/u/0/folders/0B--ztKW0d17XbXNPQVY5VWxiZkU#list

## Installation Instructions

#### Prerequierment:
Numpy: conda install -c conda-forge numpy 
npm: conda install -c conda-forge nodejs 
opencv: pip install opencv=3.4.2
pip install opencv-contrib=3.4.2

#### create conda env
conda env create -n histosearch 


#### then activate the conda virtual environment
source activate histosearch


## To Run the Model
python recognize.py --data <data_DIR> --gt <groundtruth_DIR>

example:
python recognize.py --data data/tiles --gt data/groundtruth

## To Run the CNN Model on a single node
Tensorflow-GPU and Keras should be installed.

python resnet50.py --data <data_DIR> --gt <groundtruth_DIR> -lbn_points 8 -lbn_r 1

example:
python resnet50.py -model nn -d mitoses_image_data/mitoses_image_data/mitoses_image_dataset -gt mitoses_image_data/TUPAC_groundtruth/mitoses_ground_truth/mitoses_ground_truth -lbn_points 8 -lbn_r 1

## To run the CNN Model on multiple nodes
Horovod, openMPI, NCCL2 should be installed.

mpirun -np <# of nodes> -hostfile <list of hostnames> -tag-output -bind-to none -map-by node -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python resnet50_distributed.py -model nn --data <data_DIR> --gt <groundtruth_DIR> -lbn_points 8 -lbn_r 1

example:
mpirun -np 4 -hostfile $PBS_NODEFILE -tag-output -bind-to none -map-by node -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python resnet50_distributed.py -model nn -d mitoses_image_data/mitoses_image_data/mitoses_image_dataset -gt mitoses_image_data/TUPAC_groundtruth/mitoses_ground_truth/mitoses_ground_truth -lbn_points 8 -lbn_r 1
