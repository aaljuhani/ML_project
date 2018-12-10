# 1. Gets all the images from user specified "Directory"
# 2. Performs classification on images one by one and stores the result
#       1 --> Mitosis region
#       0 ---> Non-Mitosis Region

import generate_bounding_box
import os
import recognize
from sklearn.externals import joblib
from PIL import Image, ImageDraw
import os
import pickle
import math

def perform_classification(image,model):
    data = "./fullrun_3"

    gt = "./mitoses_ground_truth"

    #Model_path = "./sift_svm_20181204-095339.pkl" 
    Model_path = model
    #Pred_path = "./fullrun_1/TUPAC-TE-200.svs_01_07.png"
    Pred_path = image

    mitoses_model = recognize.recognize(data, gt)
    mitoses_model.sift.MBkmeans = joblib.load('./kmeans_20181204-095339.pkl')
    predict_val = mitoses_model.predict(Pred_path, Model_path) 

    print("predict val =", predict_val)
    print("only binary==",predict_val[0])
    if math.isnan(predict_val[0]):
        return 0
    else:
        return predict_val[0]

file_list=os.listdir('./fullrun_3')
for file in file_list[:]: # filelist[:] makes a copy of filelist.
    if not(file.endswith(".png")):
        file_list.remove(file)
#print(file_list)

#Now file_list is a list which contains all the PNG files.
print("length==",len(file_list)) #The count is 156 in our example...

count = 1 #To check how many images are processed.
#image_names = [] #List of images
classif_dict = {} #List of classification output

for tile in file_list:
    classif = 0
    image = "fullrun_3/" +tile
    print("Image=",image)
    model = "./sift_svm_20181204-095339.pkl"
    res = perform_classification(image,model)
    #print("Image name=",tile,"count=",count)
    #if count%10 == 0:
    if res==1:
        print("Inside if")
        classif = 1
    classif_dict[tile] = classif
    #classif = 0
    count += 1

print("dic len=",len(classif_dict))

positive_outputs = [] #Store the list of all teh images that were positively classified.

for x, y in classif_dict.items():
    print("key=",x,"value=", y)
    if y ==1:
        positive_outputs.append(x) #Append the image name to the list.
#generate_bounding_box.draw_bound('./'+x)
generate_bounding_box.draw_external_bound(positive_outputs) #Pass the list of positively classified images as input.

