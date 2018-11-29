#### Learnt by watching the tutorial https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html

### Given an image just apply bounding box on that image and display teh image.
## Possible thoughts, 
# 1.use different colors based on the prediction score
# 2. Try to augment or enlarge teh image region where bounding box is applied.
import sys

from PIL import Image, ImageDraw
import os
import pickle
from copy import deepcopy

def draw_external_bound(positive_outputs):
    images_folder_path = "./sample_full_run"
    tile_file = "tiles_pickle.sav"
    load_tiles = pickle.load(open(tile_file, 'rb'))
    #exclusion_image = ["BuckId.jpg"]
    load_new_tiles = []
    for tile in load_tiles:
        print("Image name=",tile.filename)
        #image = Image.open(images_folder_path)
        image = Image.open(tile.filename)
        width, height = image.size
        print("width = ",width, "height==",height)
        #bound_vals = (0 +200, 0 +200, width -200, height-200)
        #bound_vals = (100, 1000, 1000, 2000)
        out_col = (0, 0, 255,255)
        #draw_bound("./Full_Size_Images/TUPAC-TE-319.svs_01_05.png")
        print("tile fn==", tile.filename)
        print("positive op",positive_outputs)

        cur_tile_fn =  tile.filename
        cur_tile_fn = cur_tile_fn[2:]
        print("cur_tile_fn",cur_tile_fn)
        if cur_tile_fn in positive_outputs:
            print("Enetering if")
            tile = draw_bound(tile)
        load_new_tiles.append(tile)
        #draw_bound("./Full_Size_Images/TUPAC-TE-319.svs_01_05.png", bound_vals, out_col)
    pickle.dump(load_new_tiles, open("tiles_pickle_bound.sav", 'wb'))

def draw_bound(tile):
    print("file name=",tile)
    image = Image.open(tile.filename)
    width, height = image.size
    print("width = ", width, "height==", height)
    width_adjust = 0.1 * width #Helpful for drawing the bounding box.
    height_adjust = 0.1 * height #Helpful for drawing teh bounding box.

    bound_vals = (0 + width_adjust, 0 + height_adjust, width - width_adjust, height - height_adjust)
    #bound_vals = (0 + 200, 0 + 200, width - 200, height - 200)
    # bound_vals = (100, 1000, 1000, 2000)
    out_col = (0, 0, 255)
    rimg = tile.image.copy()
    rimg_draw = ImageDraw.Draw(rimg)
    rimg_draw.rectangle(bound_vals, fill=None, outline=out_col, width=25)
    rimg.show()
    #rimg.save(image_name) #Saving the image with bounding box.
    tile.image = rimg.copy()
    #print("showing bounded tile")
    #tile.image.show()
    return tile

if __name__=="__main__":
    images_folder_path = "./sample_full_run"
    tile_file = "tiles_pickle.sav"
    load_tiles = pickle.load(open(tile_file, 'rb'))

    exclusion_image = ["bird.jpg"]
    #for img in os.listdir(images_folder_path):
        #if img in exclusion_image:
           # continue
    for tile in load_tiles:
        print("Image name=",tile.img)
        #image = Image.open(images_folder_path)
        image = Image.open(tile.img)
        width, height = image.size
        print("width = ",width, "height==",height)
        bound_vals = (0 +200, 0 +200, width -200, height-200)
        #bound_vals = (100, 1000, 1000, 2000)
        out_col = (0, 0, 255,255)
        #draw_bound("./Full_Size_Images/TUPAC-TE-319.svs_01_05.png")
        draw_bound(tile)
        #draw_bound("./Full_Size_Images/TUPAC-TE-319.svs_01_05.png", bound_vals, out_col)
    pickle.dump(load_tiles, open("tiles_pickle_bound.sav", 'ab'))
'''
import numpy as np
import cv2 as cv
img = cv.imread('BuckId.jpg',0)
ret,thresh = cv.threshold(img,127,255,0)
im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
print(cnt)
#M = cv.moments(cnt)
#print( M )

x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
'''
'''
import numpy as np
import cv2 as cv

#Create an object for the image
img = cv.imread('BuckId.jpg')

color = (0,255,0)

label = 'Mitosis'

cv.rectangle(img, (100,1000),(1000,2000),color, 2)

cv.putText(img, label, (100 - 10, 1000 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv.imshow("object detection", img)

cv.waitKey()
'''