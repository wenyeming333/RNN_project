from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array, list_pictures
import numpy as np
# create the base pre-trained model
def load_model(include_top=False):
	return InceptionV3(weights='imagenet', include_top=include_top)

#for layer_i in range(len(base_model.layers)):
#	print layer_i, base_model.layers[layer_i].output_shape

def load_get_output_fn(model, num_layer=200):
	fn = K.function([model.layers[0].input, 
					 K.learning_phase()],
					[model.layers[num_layer].output])
					
	return fn
	
def get_output_from_fn(fn, output):
	return fn([output, 0])[0]

def load_im_csv(file_path):
	import pandas as pd
	header_names = ['r_id','youtube_id','time','x','y','w','h','id']
	return pd.read_csv(file_path, header=0, names=header_names)
	
def image_generator(list_of_files, crop_size=None, to_grayscale=False, scale=1, shift=0, target_size=None):
	img_shape = img_to_array(load_img(list_of_files[0], to_grayscale, target_size=target_size)).shape
	
	images = np.empty(shape=[0, img_shape[0], img_shape[1], img_shape[2]])
	
	for filename in list_of_files:

		img = img_to_array(load_img(filename, to_grayscale, target_size=target_size))

		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)
		
		cropped_img = random_crop(img, crop_size) if crop_size else img
		
		images = np.append(images, img, axis=0)
		
	return images * scale -shift
	
def random_crop(image, crop_size):
	height, width = image.shape[1:]
	dy, dx = crop_size
	if width < dx or height < dy:
		return None
	x = np.random.randint(0, width - dx + 1)
	y = np.random.randint(0, height - dy + 1)
	return image[:, y:(y+dy), x:(x+dx)]
		


def load_image_from_dir(dir):
	images = list_pictures(dir)
	return images, image_generator(images)
	
def preprocess_input(x):
	x /= 255.
	x -= 0.5
	x *= 2.
	return x