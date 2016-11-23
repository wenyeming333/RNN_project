import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import cPickle as pickle
from scipy import ndimage
from variables import *


class VideoSolver(object):
	def __init__(self, model, data, val_data, **kwargs):
	    """
	    Required Arguments:
	    - model: Show Attend and Tell caption generating model
	    - data: Training data; dictionary with the following keys:
	        - frame_features: Feature vectors of shape (, , ,)
			- player_features: Feature vectors of shape (, , ,)
	        - labels: ....................
	        - file_names: Image file names of shape (82783, )
	    - val_data: validation data; for print out BLEU scores for each epoch.
	    Optional Arguments:
	    - n_epochs: The number of epochs to run for training.
	    - batch_size: Mini batch size.
	    - update_rule: A string giving the name of an update rule among the followings: 
	        - 'sgd'
	        - 'momentum'
	        - 'adam'
	        - 'adadelta'
	        - 'adagrad'
	        - 'rmsprop' 
	    - learning_rate: Learning rate; default value is 0.03.
	    - print_every: Integer; training losses will be printed every print_every iterations.
	    - save_every: Integer; model variables will be saved every save_every epoch.
	    - image_path: String; path for images (for attention visualization)
	    - pretrained_model: String; pretrained model path 
	    - model_path: String; model path for saving 
	    - test_model: String; model path for test 
	    """

	    self.model = model
	    self.data = data
	    self.val_data = val_data
	    self.n_epochs = kwargs.pop('n_epochs', 10)
	    self.batch_size = kwargs.pop('batch_size', 100)
	    self.update_rule = kwargs.pop('update_rule', 'adam')
	    self.learning_rate = kwargs.pop('learning_rate', 0.01)
	    self.print_every = kwargs.pop('print_every', 100)
	    self.save_every = kwargs.pop('save_every', 1)
	    self.image_path = kwargs.pop('data_path', '/home/ethan/video_proj/processed_dataolga/')
	    self.log_path = kwargs.pop('log_path', './log/')
	    self.model_path = kwargs.pop('model_path', './model/')
	    self.pretrained_model = kwargs.pop('pretrained_model', None)
	    self.num_train = kwargs.pop('num_train', None)
	    self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

	    # Set optimizer by update rule
	    if self.update_rule == 'adam':
	    	self.optimizer = tf.train.AdamOptimizer
	    elif self.update_rule == 'rmsprop':
	    	self.optimizer = tf.train.RMSPropOptimizer

	    if not os.path.exists(self.model_path):
	    	os.makedirs(self.model_path)
	    if not os.path.exists(self.log_path):
	    	os.makedirs(self.log_path)

	def train(self):
	    # Train/Val dataset
	    n_examples = self.num_train
	    n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
	    frame_features = self.data['frame_features']
	    player_features = self.data['player_features']
	    labels = self.data['label']
	    val_frame_features = self.val_data['frame_features']
	    val_player_features = self.val_data['player_features']
	    n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

	    # Build graphs for training model and sampling captions
	    loss = self.model.build_model()
	    tf.get_variable_scope().reuse_variables() # ????
	    _, _, generated_captions = self.model.build_sampler(max_len=20)

	    # Train op
	    with tf.name_scope('optimizer'):
		    optimizer = self.optimizer(learning_rate=self.learning_rate)
		    grads = tf.gradients(loss, tf.trainable_variables())
		    grads_and_vars = list(zip(grads, tf.trainable_variables())) # ?????
		    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
	       
	    # Summary op   
	    tf.scalar_summary('batch_loss', loss)
	    for var in tf.trainable_variables():
		    tf.histogram_summary(var.op.name, var)
	    for grad, var in grads_and_vars:
		    tf.histogram_summary(var.op.name+'/gradient', grad)
	    
	    summary_op = tf.merge_all_summaries() 

	    print "The number of epoch: %d" %self.n_epochs
	    print "Data size: %d" %n_examples
	    print "Batch size: %d" %self.batch_size
	    print "Iterations per epoch: %d" %n_iters_per_epoch
	    
	    config = tf.ConfigProto(allow_soft_placement = True)
	    #config.gpu_options.per_process_gpu_memory_fraction=0.9
	    config.gpu_options.allow_growth = True
	    with tf.Session(config=config) as sess:
		    tf.initialize_all_variables().run()
		    summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
		    saver = tf.train.Saver(max_to_keep=40)

		    if self.pretrained_model is not None:
				print "Start training with pretrained Model.."
				saver.restore(sess, self.pretrained_model)

		    prev_loss = -1
		    curr_loss = 0
		    start_t = time.time()

		    for e in range(self.n_epochs):
		        rand_idxs = np.random.permutation(n_examples)
		        frame_features = frame_features[rand_idxs]
		        player_features = player_features[rand_idxs]
		        labels = labels[rand_idxs]

	        	for i in range(n_iters_per_epoch):
			        frame_features_batch = frame_features[i*self.batch_size:(i+1)*self.batch_size]
			        player_features_batch = player_features[i*self.batch_size:(i+1)*self.batch_size]
			        labels_batch = labels[i*self.batch_size:(i+1)*self.batch_size]
			        
			        # How to use feed_dict !!!!!!!!!
			        feed_dict = {self.model.frame_features: frame_features_batch,\
			         self.model.player_features: player_features_batch, self.model.labels: labels_batch}
			        _, l = sess.run([train_op, loss], feed_dict)
			        curr_loss += l

	        		if i % 10 == 0:
				        summary = sess.run(summary_op, feed_dict)
				        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

			        if (i+1) % self.print_every == 0:
			            print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)


			    # Check indention!!!!!!!!!
		        print "Previous epoch loss: ", prev_loss
		        print "Current epoch loss: ", curr_loss
		        print "Elapsed time: ", time.time() - start_t

		        if (e+1) % self.save_every == 0:
			        saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
			        print "model-%s saved." %(e+1)
	        
		        prev_loss = curr_loss
		        curr_loss = 0
    


    #### TO DO: Test function!!!!!!!
	def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
	    '''
	    Args:
	    - data: dictionary with the following keys:
	      - features: Feature vectors of shape (5000, 196, 512)
	      - file_names: Image file names of shape (5000, )
	      - captions: Captions of shape (24210, 17) 
	      - image_idxs: Indices for mapping caption to image of shape (24210, ) 
	      - features_to_captions: Mapping feature to captions (5000, 4~5)
	    - split: 'train', 'val' or 'test'
	    - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
	    - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
	    '''

	    frame_features = data['frame_features']
	    player_features = data['player_features']

	    # Build a graph for sampling captions
	    alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)
	    
	    config = tf.ConfigProto(allow_soft_placement=True)
	    config.gpu_options.allow_growth = True # ?????????????
	    with tf.Session(config=config) as sess:
	    	saver = tf.train.Saver()
		    saver.restore(sess, self.test_model)
		    _, features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
		    feed_dict = {self.model.frame_features: frame_features_batch,\
			         self.model.player_features: player_features_batch, self.model.labels: labels_batch}
		    alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
		    decoded = decode_captions(sam_cap, self.model.idx_to_word)

	    	if attention_visualization:
	        	for n in range(10):
	        		print "Sampled Caption: %s" %decoded[n]

				    # Plot original image
				    img_path = os.path.join(self.image_path, image_files[n])
				    img = ndimage.imread(img_path)
				    plt.subplot(4, 5, 1)
				    plt.imshow(img)
				    plt.axis('off')

			        # Plot images with attention weights
			        words = decoded[n].split(" ")
			        for t in range(len(words)):
			            if t>18:
			                break
			            plt.subplot(4, 5, t+2)
			            plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
			            #plt.text(0, 20, bts[n,t], color='black', backgroundcolor='white', fontsize=12)
			            plt.imshow(img)
			            alp_curr = alps[n,t,:].reshape(14,14)
			            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
			            plt.imshow(alp_img, alpha=0.85)
			            plt.axis('off')
			        plt.show()

	    	if save_sampled_captions:
		        all_sam_cap = np.ndarray((features.shape[0], 20))
		        num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
		        for i in range(num_iter):
			        features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
			        feed_dict = { self.model.features: features_batch }
			        all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)  
		        all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
		        save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" %(split,split))