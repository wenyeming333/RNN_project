from variables import *
from preprocess.inception_v3 import *
import time
import cPickle as Pickle


def saveProcessAllData():
	from os import listdir
	from glob import glob 
	
	videos = listdir(processed_dir)
	model = InceptionV3(include_top=False, weights='imagenet')
	
	for v in videos:
		clips = listdir('{}/{}'.format(processed_dir, v))
		for clip in clips:
		
			start_time = time.time()
			img_gen, len_gen = load_image_from_dir('{}/{}/{}'.format(processed_dir, v, clip))
			import pdb
			pdb.set_trace()
			clip_result = model.predict_generator(img_gen, len_gen)
			
			print('Processing {}-{}:\n\tThe program takes {} seconds to run'.format(v, clip,
																	time.time()-start_time))
																	
			#np.save('npyfile.npy',clip_result)
			#with open('Picklefile.pkl','wb') as f:
			#	Pickle.dump(clip_result,f)
				
			
			
def main():
	saveProcessAllData()
	
if __name__ == '__main__':
	main()