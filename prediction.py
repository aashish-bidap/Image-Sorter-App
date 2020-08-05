import tensorflow as tf
from tensorflow import keras
from os import walk
from os import listdir
from os.path import isfile, join
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

class ImageClassifier:
	def __init__(self,image_dir):
		
		self.image_dir = image_dir
		#'/Users/abhishekbidap/Desktop/my_stuff/tkinter/Images/'
		self.model_path = '/Users/abhishekbidap/Desktop/my_stuff/tkinter/venv/model_dogsvscats'

	def model_import(self):
		"""
		loading the trained model 
		"""
		model = keras.models.load_model(self.model_path)
		#print(model.summary())
		return model

	def importing_files(self):

		"""
		importing the images to be predicted	
		"""
		image_files	= [join(self.image_dir,filename) for filename in listdir(self.image_dir)]
		return image_files

	def read_and_prep_images(self,img_height=224, img_width=224):

		"""
		preparing the images before feeding the images for the model predictions.
		"""
		image_paths = self.importing_files()
		print("image_paths",image_paths)
		for i in image_paths:
			if 'DS_Store' in i:
				image_paths.remove(i)
		
		imgs = [load_img(img_path,color_mode='rgb',target_size=(img_height, img_width)) for img_path in image_paths]
		img_array = np.array([img_to_array(img) for img in imgs])
		output = preprocess_input(img_array)
		return output,image_paths

	def image_predict(self):

		"""
		model predictions
		"""
		predictions={}
		model = self.model_import()
		print("Model Import Done")
		
		test_data,image_paths = self.read_and_prep_images()
		print("Read and Preparation of the data is done")
		preds = model.predict(test_data)

		for img_paths,pred in zip(image_paths,np.argmax(preds,axis=1)):
  			if pred == 1:
  				#print(img_paths,": Predicted as Upright")
  				predictions[img_paths]='Upright'
  			else:
  				#print(img_paths,": Predicted as Sideways")
  				predictions[img_paths]='Sideways'
		return predictions
