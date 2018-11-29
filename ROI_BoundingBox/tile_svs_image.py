######## The main functionality is to tile a svs image into smaller pieces #################
# 1. The division will happend based on the size of the full size image.
# 2. The divided images would be sequentially stored in an output directory (Name would hold a counter
# that would be incremented sequentially left -> right)
# 3. Later Perform classification on these individual images and return the
# full vector of classication, a vector of segemnted images (only their location would be useful).

# Source:  http://www.andrewjanowczyk.com/dividing-and-re-merging-large-images-humpty-dumpty/

#tilesize  = [2000, 2000] #The size of the tile is 2000 x 2000
import image_slicer
import PIL.Image
from PIL import Image
import pickle

#Please change the maximum size so that the default value is adjusted.
PIL.Image.MAX_IMAGE_PIXELS = 1011497282371

#img = Image.open('Full_Size_Images/TUPAC-TE-319.svs.tiff')

#img = Image.open('Full_Size_Images/TUPAC-TE-319.svs.tiff')
src_image = "./bird.jpg"
img = Image.open('./bird.jpg')

width, height = img.size
print("width=",width)
print("height=", height)

'''
#Get the size of mitosis image
mit_img = Image.open('Mitosis_TestImage/01/01.tif')
mit_wid, mit_height = mit_img.size

print("mitosis width=",mit_wid)
print("mitosis height=",mit_height)

# 1. Get the size of teh full-size image
# 2. Get the size of the mitosis sample image.
# 3. Divide the full size image to sizes approximately of a mitosis image.
# Formula: count_images = size_fullSize_image/size_of_mitosis image.

tiles_count = (width * height)/(mit_wid * mit_height)
#print("Tiles count=",tiles_count)
'''

tiles_count = 3
tiles = image_slicer.slice(src_image, tiles_count)
pickle.dump(tiles,open("tiles_pickle.sav",'wb'))

# The sliced images will be in the directory of slicable image.,..
