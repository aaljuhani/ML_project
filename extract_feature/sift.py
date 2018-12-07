
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import vq
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed
from timeit import default_timer as timer
import multiprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
import time
from sklearn.externals import joblib



class SIFT:
	def __init__(self, num_codes, iter):
		self.num_codes = num_codes
		self.iter = iter
		self.TIMESTR = time.strftime("%Y%m%d-%H%M%S")
		self.k = num_codes
		self.MBkmeans = MiniBatchKMeans(n_clusters=self.k)#, verbose=1)

	def extract(self, paths):
		# Visual Bag of Word model

		# List where all the descriptors are stored
		des_list = []
		
		print ('k = ', self.k)
		# Parallel(n_jobs=2, prefer="threads")(
			# for image_path in tqdm(image_paths):
				# im = cv2.imread(image_path)
				# # kpts = fea_det.detect(im)
				# kpts, des = self.sift.detectAndCompute(im, None)
				# des_list.append((des))   
		# )
		
		print ('\nNo of cores:', multiprocessing.cpu_count())
		cores = 16
		start = timer()
		r = Parallel(n_jobs=cores)(delayed(self.get_features)(image, label) for image, label in tqdm(paths))
		des_list, tr_labels = zip(*r)
		end = timer()
		print('\nTime taken to extract SIFT features',(end - start)) # Time in seconds
		
		tot_kpts = 0
		for des in tqdm(des_list):
			tot_kpts += len(des)
		
		print ('no of images:', len(des_list))
		print ('\ntot_kpts',tot_kpts)
		descriptors = np.empty((tot_kpts, 128), dtype='float32')

		print ('\n******* stack the descriptors *******')
		start = timer()
		# print ('des_list', len(des_list), tr_labels)
		# # Stack all the descriptors vertically in a numpy array
		# descriptors = des_list[0]
		j = 0
		for descriptor in (des_list):
			l = len(descriptor)
			# print ('keypoints:', l)
			descriptors[j:(j+l),:] = descriptor
			j += l
			# np.vstack((descriptors, descriptor))  
		# print ('descriptors.shape',descriptors.shape)
		print ('j, tot_kpts', j, tot_kpts)
		end = timer()
		print('\nTime taken to stack descriptor',(end - start)) # Time in seconds

		print ('\n******* applying kmeans algorithm *******')
		start = timer()
		# Perform k-means clustering
		# voc, variance = kmeans(descriptors, self.num_codes, self.iter) 
		# print (voc.shape, variance.shape)
		# output = KMeans(n_clusters=k, n_jobs=cores, verbose=2).fit(descriptors)
		# MBkmeans = MiniBatchKMeans(n_clusters=k, verbose=2)
		stdSlr = StandardScaler().fit(descriptors)
		scaled_des = stdSlr.transform(descriptors)
		output = self.MBkmeans.fit(scaled_des)
		voc = output.cluster_centers_
		end = timer()
		print('\nTime taken to generate codebook',(end - start)) # Time in seconds
		
		joblib.dump(self.MBkmeans, 'logs/kmeans_'+self.TIMESTR+'.pkl', compress=9)
		
		print ('\n******* generate histogram for each image *******')
		start = timer()
		# Calculate the histogram of features
		im_features = np.zeros((len(des_list), self.k), "float32")
		j = 0
		for i in tqdm(range(len(des_list))):
			# words, distance = vq(des_list[i][1],voc)
			l = len(des_list[i])
			words = output.labels_[j:j+l]
			for w in words:
				im_features[i][w] += 1
			j += l
		end = timer()
		print('Time taken to create histogram',(end - start)) # Time in seconds

		print ('\n******* scale the features *******')
		# Scaling the words
		stdSlr = StandardScaler().fit(im_features)
		scaled_features = stdSlr.transform(im_features)
		
		print ('\n******* Features and labels generated *******')

		return scaled_features, tr_labels
		
	def get_features(self, image_path, label_path=None):
		im = cv2.imread(image_path)
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		new_px = 500
		# we need to keep in mind aspect ratio so the image does
		# not look skewed or distorted -- therefore, we calculate
		# the ratio of the new image to the old image
		r = float(new_px) / gray.shape[1]
		dim = (new_px, int(gray.shape[0] * r))

		# perform the actual resizing of the image and show it
		resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

		sift = cv2.xfeatures2d.SIFT_create()
		kpts, des = sift.detectAndCompute(resized, None)
		label = None
		if label_path is not None:
			label = 1 if os.path.isfile(label_path) else 0
		
		return des, label
		
	def gen_histogram(self, des):
		im_features = np.zeros((self.k), "float32")
		stdSlr = StandardScaler().fit(des)
		scaled_des = stdSlr.transform(des)
		self.MBkmeans = self.MBkmeans.set_params(verbose=0)
		words = self.MBkmeans.predict(scaled_des)
		# words = output.labels_
		for w in words:
			im_features[w] += 1
			
		stdSlr = StandardScaler().fit(im_features.reshape(-1, 1))
		scaled_features = stdSlr.transform(im_features.reshape(-1, 1))
		
		return scaled_features
		


