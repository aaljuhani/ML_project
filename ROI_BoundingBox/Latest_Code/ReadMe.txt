#########################################
# Instructions for Histosearch Process	#
#					#	
#########################################

Step #1: Replace the source image and destination folder where tiles need to be stored.
 Update tile_svs_image.py, Perform_Classification_SVM.py, generate_bounding_box.py, combine_bounded_images, recognize.py

Step #2: Run tile_svs_image.py

--- This might take approximately 20 minutes depending on size of WSI (whole size image). 
---- Mostly a 1.5 GB image takes 25 minutes.

Make sure the tiles and corresponding pickle file with meta data are created.

Step #3: Run Perform_Classification_SVM.py

This file performs classification on tiles, stores the result of those tiles
and applies a bounding box on mitosis positive images.

Step #4: Run combine_bounded_images

It combines the tiles into single image.

*** Optional ***** For a preview do following:
Step #5: Compress teh huge image and take a screenshot.

