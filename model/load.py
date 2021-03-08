
import numpy as np 
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
from skimage.io import imread, imshow
from skimage.transform import resize

import tensorflow as tf 

def init():
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")
	print("Loaded Model from Disk")

	loaded_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

	graph = tf.compat.v1.get_default_graph()

	return loaded_model, graph