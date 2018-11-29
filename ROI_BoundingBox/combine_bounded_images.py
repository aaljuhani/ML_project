import numpy as np
from PIL import Image
import image_slicer
import pickle


def combine_tiles(tile_file):
    load_tiles = pickle.load(open(tile_file,'rb'))
    image_combine = image_slicer.join(load_tiles)
    image_combine.save('combined_image.png')

if __name__ == "__main__":
    print("Combining the tiled images..")
    tiled_image_files = "tiles_pickle_bound.sav"
    combine_tiles(tiled_image_files)
    print("Tiles combined successfully")
