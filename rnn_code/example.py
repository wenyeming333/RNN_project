# clips: 527, 195, 672

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import prettytensor as pt 
import numpy as np
from load_data import *
from load_imdb_movie_data import *
from util import *
#from __future__ import print_function

slim = tf.contrib.slim
slim_predict = tf.contrib.slim


class SiameseModel:

	conv1_size = 6
	conv1_chan = 5
	conv2_size = 6
	conv2_chan = 14
	conv3_size = 13
	conv3_chan = 60

	n_out = 40
	device = '/gpu:2'
	#fc2_size = 256

	def __init__(self, folder_path, model_path, base_lr=0.001, img_size=64, grad_clip=5, gray=True, seed=111,
				normalization='-1:1', moving_average_decay=1.0, gpu_memory_fraction=1.0,
				decov_loss_factor=0.0):

		self.gray = gray
		self.channel_size = 1 if gray else 3

		self.img_size = img_size
		self.seed = seed
		self.base_lr = base_lr
		#self._set_model(grad_clip=5)
		self.folder_path = folder_path
		self.normalization = normalization
		self.model_path = model_path
		self.moving_average_decay = moving_average_decay
		self.decov_loss_factor = decov_loss_factor
		self.weight_decay = 0.0
		self.whiten = True

		self.gpu_memory_fraction = gpu_memory_fraction
		#self.temp()

	def set_data(self, train_data, valid_data, data_dir, zca=None):
		self._train_data = train_data
		self._valid_data = valid_data
		self.data_dir = data_dir
		self._zca = zca
		
	def get_pair_batch(self, batch_size, mode='train'):
		return self.data.get_pairs(mode, batch_size)

	def get_next_batch(self, batch_size, mode='train'):
		from load_data import *

		if mode == 'train':
			return loadNextBatch(self._train_data, batch_size, img_size=self.img_size,
								data_dir = self.data_dir, norm_mode=self.normalization,
								zca=self._zca, gray=self.gray)

		else:
			return loadNextBatch(self._valid_data, batch_size, img_size=self.img_size,
								data_dir = self.data_dir, norm_mode=self.normalization,
								zca=self._zca, gray=self.gray)


	def set_movie_data(self):
		self.movie_data = MovieData(self.img_size, self.gray, self.whiten)
		
	def set_triplet_data(self):
		self.data = Data(img_size=self.img_size, gray=self.gray)

	def get_triplet_batch(self, batch_size, mode='train', flip=True, whiten=True):
		return self.data.get_batch(mode, batch_size, flip, whiten)

	def _set_model(self, grad_clip=5):
		'''Build the model.'''

		with tf.Graph().as_default():

			tf.set_random_seed(self.seed)
			self.global_step = tf.Variable(0, trainable=False)
			self.phase_train = tf.placeholder(tf.bool, name='phase_train')
			#self.phase_train = True
			
			self.images = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size,
															self.channel_size])

			self.decay_buff = tf.placeholder(tf.int32)
			self.learning_rate = self.base_lr * 0.97 ** (tf.cast(self.decay_buff, tf.float32)//100)
			#self.labels = tf.placeholder(tf.float32, shape=[None, 1])

			self.embeddings = self._get_embedding(self.images, phase_train=self.phase_train)
			self.embedding = slim.fully_connected(self.embeddings, 128, activation_fn=None,
												   scope='Embeddings', reuse=False)

			self.embedding = tf.nn.l2_normalize(self.embedding, 1, 1e-10, name='embeddings')
			self.embed_anchor, self.embed_pos, self.embed_neg = tf.split(0, 3, self.embedding)

			self.triplet_loss = self._triplet_loss()

			self.regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			self.total_loss = tf.add_n([self.triplet_loss] + self.regularization_losses, name='total_loss')

			#add loss summaries
			loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
			losses = tf.get_collection('losses')
			self.loss_averages_op = loss_averages.apply(losses + [self.total_loss])

			self.update_gradient_vars = tf.all_variables()

			with tf.control_dependencies([self.loss_averages_op]):
				opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
				grads = opt.compute_gradients(self.total_loss, self.update_gradient_vars)

			self.apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

			variable_averages = tf.train.ExponentialMovingAverage(
									self.moving_average_decay, self.global_step)

			self.variables_averages_op = variable_averages.apply(tf.trainable_variables())

			with tf.control_dependencies([self.apply_gradient_op, self.variables_averages_op]):
				self.train_op = tf.no_op(name='train')

			self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=8)

			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
			self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

			# Initialize variables
			self.sess.run(tf.initialize_all_variables())
			self.sess.run(tf.initialize_local_variables())

			tf.train.start_queue_runners(sess=self.sess)


	def _get_embedding(self, images, phase_train=True, weight_decay=0.0, keep_probability=1.0):
		self.end_points = {}

		net = self._conv(images, self.channel_size, 64, 7, 7, 2, 2, 'SAME', 'conv1_7x7',
						 phase_train=phase_train, weight_decay=weight_decay)
		self.end_points['conv1'] = net
		net = self._mpool(net, 3, 3, 2, 2, 'SAME', 'pool1')
		self.end_points['pool1'] = net 
		net = self._conv(net, 64, 64, 1, 1, 1, 1, 'SAME', 'conv2_1x1',
						 phase_train=phase_train, weight_decay=weight_decay)
		self.end_points['conv2_1x1'] = net
		net = self._conv(net, 64, 192, 3, 3, 1, 1, 'SAME', 'conv3_3x3',
						 phase_train=phase_train, weight_decay=weight_decay)
		self.end_points['conv3_3x3'] = net
		net = self._mpool(net, 3, 3, 2, 2, 'SAME', 'pool3')
		self.end_points['pool3'] = net

		net = self._inception(net, 192, 1, 64, 96, 128, 16, 32, 3, 32, 1,
								'MAX', 'incept3a', phase_train=phase_train, 
								weight_decay=weight_decay)
		self.end_points['incept3a'] = net
		net = self._inception(net, 256, 1, 64, 96, 128, 32, 64, 3, 64, 1, 
								'MAX', 'incept3b', phase_train=phase_train, 
								weight_decay=weight_decay)
		self.end_points['incept3b'] = net
		net = self._inception(net, 320, 2, 0, 128, 256, 32, 64, 3, 0, 2,
								'MAX', 'incept3c', phase_train=phase_train,
								weight_decay=weight_decay) 
		self.end_points['incept3c'] = net

		net = self._inception(net, 640, 1, 256, 96, 192, 32, 64, 3, 128, 1,
								'MAX', 'incept4a', phase_train=phase_train, 
								weight_decay=weight_decay)
		self.end_points['incept4a'] = net
		net = self._inception(net, 640, 1, 224, 112, 224, 32, 64, 3, 128, 1,
								'MAX', 'incept4b', phase_train=phase_train,
								weight_decay=weight_decay)
		self.end_points['incept4b'] = net
		net = self._inception(net, 640, 1, 192, 128, 256, 32, 64, 3, 128, 1,
								'MAX', 'incept4c', phase_train=phase_train,
								weight_decay=weight_decay)
		self.end_points['incept4c'] = net
		net = self._inception(net, 640, 1, 160, 144, 288, 32, 64, 3, 128, 1,
								'MAX', 'incept4d', phase_train=phase_train,
								weight_decay=weight_decay)
		self.end_points['incept4d'] = net
		net = self._inception(net, 640, 2, 0, 160, 256, 64, 128, 3, 0, 2,
								'MAX', 'incept4e', phase_train=phase_train,
								use_batch_norm=True)
		self.end_points['incept4e'] = net
		
		net = self._inception(net, 1024, 1, 384, 192, 384, 0, 0, 3, 128, 1,
								'MAX', 'incept5a', phase_train=phase_train,
								weight_decay=weight_decay)
		self.end_points['incept5a'] = net
		net = self._inception(net, 896, 1, 384, 192, 384, 0, 0, 3, 128, 1,
								'MAX', 'incept5b', phase_train=phase_train,
								weight_decay=weight_decay)
		self.end_points['incept5b'] = net
		net = self._apool(net,  3, 3, 1, 1, 'VALID', 'pool6')
		self.end_points['pool6'] = net
		net = tf.reshape(net, [-1, 896])
		self.end_points['prelogits'] = net
		net = tf.nn.dropout(net, keep_probability)
		self.end_points['dropout'] = net

		return net


	def _conv(self, inpOp, nIn, nOut, kH, kW, dH, dW, padType, name, phase_train=True, use_batch_norm=True, weight_decay=0.0):
		with tf.variable_scope(name):
			l2_regularizer = lambda t: self._l2_loss(t, weight=weight_decay)
			kernel = tf.get_variable("weights", [kH, kW, nIn, nOut],
				initializer=tf.truncated_normal_initializer(stddev=1e-1),
				regularizer=l2_regularizer, dtype=inpOp.dtype)
			cnv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
			
			if use_batch_norm:
				conv_bn = self._batch_norm(cnv, phase_train)
			else:
				conv_bn = cnv
			biases = tf.get_variable("biases", [nOut], initializer=tf.constant_initializer(), dtype=inpOp.dtype)
			bias = tf.nn.bias_add(conv_bn, biases)
			conv1 = tf.nn.relu(bias)
		return conv1

	def _inception(self, inp, inSize, ks, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2, o4s3, poolType, name, 
				  phase_train=True, use_batch_norm=True, weight_decay=0.0):
	  
		print('name = ', name)
		print('inputSize = ', inSize) 
		print('kernelSize = {3,5}')
		print('kernelStride = {%d,%d}' % (ks,ks))
		print('outputSize = {%d,%d}' % (o2s2,o3s2))
		print('reduceSize = {%d,%d,%d,%d}' % (o2s1,o3s1,o4s2,o1s))
		print('pooling = {%s, %d, %d, %d, %d}' % (poolType, o4s1, o4s1, o4s3, o4s3))
		if (o4s2>0):
			o4 = o4s2
		else:
			o4 = inSize
		print('outputSize = ', o1s+o2s2+o3s2+o4)
		print(' ')
		
		net = []
		
		with tf.variable_scope(name):
			with tf.variable_scope('branch1_1x1'):
				if o1s>0:
					conv1 = self._conv(inp, inSize, o1s, 1, 1, 1, 1, 'SAME', 'conv1x1', 
										phase_train=phase_train, use_batch_norm=use_batch_norm,
										weight_decay=weight_decay)
					net.append(conv1)
		  
			with tf.variable_scope('branch2_3x3'):
				if o2s1>0:
					conv3a = self._conv(inp, inSize, o2s1, 1, 1, 1, 1, 'SAME', 'conv1x1', 
										phase_train=phase_train, use_batch_norm=use_batch_norm, 
										weight_decay=weight_decay)
					conv3 = self._conv(conv3a, o2s1, o2s2, 3, 3, ks, ks, 'SAME', 'conv3x3',
										phase_train=phase_train, use_batch_norm=use_batch_norm,
										weight_decay=weight_decay)
					net.append(conv3)
		  
			with tf.variable_scope('branch3_5x5'):
				if o3s1>0:
					conv5a = self._conv(inp, inSize, o3s1, 1, 1, 1, 1, 'SAME', 'conv1x1',
										phase_train=phase_train, use_batch_norm=use_batch_norm,
										weight_decay=weight_decay)
					conv5 = self._conv(conv5a, o3s1, o3s2, 5, 5, ks, ks, 'SAME', 'conv5x5',
										phase_train=phase_train, use_batch_norm=use_batch_norm,
										weight_decay=weight_decay)
					net.append(conv5)
		  
			with tf.variable_scope('branch4_pool'):
				if poolType=='MAX':
					pool = self._mpool(inp, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
				elif poolType=='L2':
					pool = self._lppool(inp, 2, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
				else:
					raise ValueError('Invalid pooling type "%s"' % poolType)
				
				if o4s2>0:
					pool_conv = self._conv(pool, inSize, o4s2, 1, 1, 1, 1, 'SAME', 'conv1x1',
											phase_train=phase_train, use_batch_norm=use_batch_norm,
											weight_decay=weight_decay)
				else:
					pool_conv = pool
				net.append(pool_conv)

			from tensorflow.python.ops import array_ops
			incept = array_ops.concat(3, net, name=name)
		return incept


	def _l2_loss(self, tensor, weight=1.0, scope=None):
		"""Define a L2Loss, useful for regularize, i.e. weight decay.
		Args:
		  tensor: tensor to regularize.
		  weight: an optional weight to modulate the loss.
		  scope: Optional scope for op_scope.
		Returns:
		  the L2 loss op.
		"""
		with tf.name_scope(scope):
			weight = tf.convert_to_tensor(weight,
										  dtype=tensor.dtype.base_dtype,
										  name='loss_weight')
			loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
		return loss

	def _lppool(self, inpOp, pnorm, kH, kW, dH, dW, padding, name):
		with tf.variable_scope(name):
			if pnorm == 2:
				pwr = tf.square(inpOp)
			else:
				pwr = tf.pow(inpOp, pnorm)
			  
			subsamp = tf.nn.avg_pool(pwr,
								  ksize=[1, kH, kW, 1],
								  strides=[1, dH, dW, 1],
								  padding=padding)
			subsamp_sum = tf.mul(subsamp, kH*kW)
			
			if pnorm == 2:
				out = tf.sqrt(subsamp_sum)
			else:
				out = tf.pow(subsamp_sum, 1/pnorm)
		
		return out

	def _mpool(self, inpOp, kH, kW, dH, dW, padding, name):
		with tf.variable_scope(name):
			maxpool = tf.nn.max_pool(inpOp,
						   ksize=[1, kH, kW, 1],
						   strides=[1, dH, dW, 1],
						   padding=padding)  
		return maxpool

	def _apool(self, inpOp, kH, kW, dH, dW, padding, name):
		with tf.variable_scope(name):
			avgpool = tf.nn.avg_pool(inpOp,
								  ksize=[1, kH, kW, 1],
								  strides=[1, dH, dW, 1],
								  padding=padding)
		return avgpool

	def _batch_norm(self, x, phase_train):
		"""
		Batch normalization on convolutional maps.
		Args:
			x:		   Tensor, 4D BHWD input maps
			n_out:	   integer, depth of input maps
			phase_train: boolean tf.Variable, true indicates training phase
			scope:	   string, variable scope
			affn:	  whether to affn-transform outputs
		Return:
			normed:	  batch-normalized maps
		Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
		"""
		name = 'batch_norm'
		with tf.variable_scope(name):
			phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
			n_out = int(x.get_shape()[3])
			beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
							   name=name+'/beta', trainable=True, dtype=x.dtype)
			gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
								name=name+'/gamma', trainable=True, dtype=x.dtype)
		  
			batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
			ema = tf.train.ExponentialMovingAverage(decay=0.9)
			def mean_var_with_update():
				ema_apply_op = ema.apply([batch_mean, batch_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(batch_mean), tf.identity(batch_var)

			from tensorflow.python.ops import control_flow_ops
			mean, var = control_flow_ops.cond(phase_train,
											  mean_var_with_update,
											  lambda: (ema.average(batch_mean), ema.average(batch_var)))
			normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
		return normed

	def _contrastive_loss(self, margin=30.0):

		self.dist = tf.reduce_sum(tf.square(self.embed_b1 - self.embed_b2), 1)
		y_label = tf.reshape(self.labels, [-1])

		return tf.reduce_mean(y_label*self.dist + (1.0-y_label)*tf.maximum(margin-self.dist,0.0))

	def _triplet_loss(self, alpha=0.50):
		self.pos_dist = tf.reduce_sum(tf.square(self.embed_anchor - self.embed_pos),1)
		self.neg_dist = tf.reduce_sum(tf.square(self.embed_anchor - self.embed_neg),1)
		return tf.reduce_mean(tf.maximum(self.pos_dist - self.neg_dist + alpha, 0.0))
		
	def _decov_loss(self, xs):
		"""Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
		'Reducing Overfitting In Deep Networks by Decorrelating Representation'
		"""
		x = tf.reshape(xs, [int(xs.get_shape()[0]), -1])
		m = tf.reduce_mean(x, 0, True)
		z = tf.expand_dims(x-m, 2)
		corr = tf.reduce_mean(tf.batch_matmul(z, tf.transpose(z, perm=[0,2,1])), 0)
		corr_frob_sqr = tf.reduce_sum(tf.square(corr))
		corr_diag_sqr = tf.reduce_sum(tf.square(tf.diag_part(corr)))
		loss = 0.5*(corr_frob_sqr - corr_diag_sqr)
		return loss 

	def _set_predictor(self):
		with tf.Graph().as_default():
			tf.set_random_seed(self.seed)
			self.global_step = tf.Variable(0, trainable=False)
			self.phase_train = tf.placeholder(tf.bool, name='phase_train')
		
			self.pred_images = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size,
															self.channel_size])

			self.decay_buff = tf.placeholder(tf.float32)
			
			self.predictor_label = tf.placeholder(tf.int64, [None])
			
			self.prelogits = self._get_embedding(self.pred_images, phase_train=self.phase_train)
			self.prelogits = tf.nn.l2_normalize(self.prelogits, 1, 1e-10, name='embeddings')
			self.embed_A, self.embed_B = tf.split(0, 2, self.prelogits)
			self.prelogits = (self.embed_A - self.embed_B)**2
			
			with tf.variable_scope('Logits'):
				n = int(self.prelogits.get_shape()[1])
				m = 2
				w = tf.get_variable('w', shape=[n,m], dtype=tf.float32, 
					initializer=tf.truncated_normal_initializer(stddev=0.1), 
					regularizer=slim.l2_regularizer(self.weight_decay),
					trainable=True)

				b = tf.get_variable('b', [m], initializer=None, trainable=True)
				self.logits = tf.matmul(self.prelogits, w) + b
				
			if self.decov_loss_factor>0.0:
				self.logits_decov_loss = self._decov_loss(self.logits) * self.decov_loss_factor
				tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.logits_decov_loss)
			
			self.pred_embeddings = tf.nn.l2_normalize(self.prelogits, 1, 1e-10, name='embeddings')
			self.pred_learning_rate = self.base_lr * 0.97 ** (self.decay_buff//100)
			tf.scalar_summary('learning_rate', self.pred_learning_rate)
			
			# Calculate the average cross entropy loss across the batch
			self.pred_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
				self.logits, self.predictor_label, name='cross_entropy_per_example')
			self.prob = tf.nn.softmax(self.logits)
			self.prediction = tf.argmax(self.prob, 1)
			
			self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predictor_label, self.prediction), tf.float32), 0)
			self.cross_entropy_mean = tf.reduce_mean(self.pred_cross_entropy, name='cross_entropy')
			tf.add_to_collection('losses',self.cross_entropy_mean)
			
			# Calculate the total losses
			self.pred_regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			self.pred_total_loss = tf.add_n([self.cross_entropy_mean] + self.pred_regularization_losses,
										name='total_loss')
										
			#add loss summaries
			loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
			losses = tf.get_collection('losses')
			self.pred_loss_averages_op = loss_averages.apply(losses + [self.pred_total_loss])

			self.update_gradient_vars = tf.all_variables()

			with tf.control_dependencies([self.pred_loss_averages_op]):
				opt = tf.train.AdamOptimizer(self.pred_learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
				grads = opt.compute_gradients(self.pred_total_loss, self.update_gradient_vars)

			self.apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

			variable_averages = tf.train.ExponentialMovingAverage(
							self.moving_average_decay, self.global_step)

			self.variables_averages_op = variable_averages.apply(tf.trainable_variables())

			with tf.control_dependencies([self.apply_gradient_op, self.variables_averages_op]):
				self.pred_op = tf.no_op(name='train')

			self.pred_saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)
			
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
			self.pred_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

			
			# Initialize variables
			self.pred_sess.run(tf.initialize_all_variables())
			self.pred_sess.run(tf.initialize_local_variables())

			tf.train.start_queue_runners(sess=self.pred_sess)
			
	def predict(self, model_path, clip_name='clip_00672'):#clip_00005
		
		print 'Processing {}...'.format(clip_name)
		self._set_model(grad_clip=5)
		self.set_movie_data()
		
		from skimage.io import imsave
		from skimage.io import imread

		with self.sess.as_default():
			self.restore_model(self.sess, self.saver, model_path)
			
			frame_images, face_images, face_dir = self.movie_data.get_clip(clip_name)
			cast_info = self.movie_data.load_actor_info()
			face_dist = {}
			
			save_img_dir = setFileDirectory(self.folder_path, 'img')
			
			from html_dashboard.codes.main import HTMLFramework

			html_page = HTMLFramework('result',
									html_folder = self.folder_path,
									page_title='Result')
		
			save_frame_images = saveImagesFromMatrix(self.folder_path, frame_images,
													 prefix='img/frame')
													 
			info = ['Clip = {}'.format(clip_name), 'Loaded model: {}'.format(model_path)]
			html_page.set_list(info, sec_name='Informations:')
			
			
			html_page.set_image_table(save_frame_images, width=240, height=100, num_col = 5, sec_name='Frames')
											
			for face_i in range(len(face_dir)):
			
				face_img = face_images[face_i,::]
				face_img = face_img.reshape(1, self.img_size, self.img_size, 1 if self.gray else 3)
				
				min_avg_dist = 4.0
				min_person = None
				
				buff = face_dir[face_i][face_dir[face_i].find(clip_name)+11:face_dir[face_i].rfind('.')]
				person_id = buff[buff.find('id')+2:]
				person_id = person_id[:person_id.find('_')]
				
				save_min_image_path = '{}/{}_min.jpg'.format(save_img_dir, buff)
				save_face_path = '{}/{}.jpg'.format(save_img_dir, buff)
				
				img = imread(face_dir[face_i])
				imsave(save_face_path, img)
				
				if person_id not in face_dist:
					face_dist[person_id] = {}

				for imdb in self.movie_data.IMDB_data.keys():
				
					imdb_images = self.movie_data.get_images_from_path(self.movie_data.IMDB_data[imdb], self.img_size, self.gray, self.whiten)
					
					if type(imdb_images) == type(None):
						continue
					
					
					face_img_repeat = np.repeat(face_img, imdb_images.shape[0], axis=0)
				
					batch_images = np.vstack([imdb_images, face_img_repeat, np.zeros(imdb_images.shape)])
					
					cost, dist = self.sess.run(
						[self.total_loss, self.pos_dist],
						feed_dict={
							self.images: batch_images,
							#self.labels: train_batch['labels'].reshape(batch_size, 1),
							self.decay_buff: 0,
							self.phase_train: False
						}
					)
					
					#import pdb
					#pdb.set_trace()
					if np.sqrt(np.min(dist)) < min_avg_dist:
						min_person = imdb
						min_avg_dist = np.sqrt(np.min(dist))
						min_index = np.argmin(dist)
						imsave(save_min_image_path, imread(self.movie_data.IMDB_data[imdb][min_index]))
						
					
					if imdb not in face_dist[person_id]:
						face_dist[person_id][imdb] = [np.sqrt(np.min(dist))]
					else:
						face_dist[person_id][imdb].append(np.sqrt(np.min(dist)))
						
				for actor_name in cast_info.keys():
					if cast_info[actor_name]['actor_id'] == min_person or cast_info[actor_name]['character_id'] == min_person:
						print '{}: min person: {}({}), min prob: {:.4f}'.format(person_id, actor_name, min_person, min_avg_dist)
						break
				
				html_page.set_image('img/{}.jpg'.format(buff), sec_name='{}'.format(buff))
				html_page.set_image('img/{}_min.jpg'.format(buff))
			
			for person_id in face_dist:
				for imdb, dist1 in face_dist[person_id].iteritems():
					if 'ch' in imdb:
						for actor_name in cast_info.keys():
							if cast_info[actor_name]['character_id'] == imdb:
								dist2 = face_dist[person_id][cast_info[actor_name]['actor_id']]
								dist = []
								for d1, d2 in zip(dist1, dist2):
									dist.append(min(d1, d2))
								face_dist[person_id][cast_info[actor_name]['actor_id']] = dist
								break
			
			for person_id in face_dist:
				tmp = {k:sum(v) for k,v in face_dist[person_id].items() if 'ch' not in k}
				face_dist[person_id] = tmp
				
			print '\n'
			for person_id in face_dist:
				html_list = []
				print '\n' + person_id + ':'
				temp = face_dist[person_id].values()
				temp.sort()
				for imdb, dist in face_dist[person_id].iteritems():
					if dist in temp[:10]:
						for actor_name in cast_info.keys():
							if cast_info[actor_name]['actor_id'] == imdb or cast_info[actor_name]['character_id'] == imdb:
								print '\t person: {}({}), dist: {}'.format(actor_name, imdb, dist)
								html_list.append('person: {}({}), dist: {}'.format(actor_name, imdb, dist))
								break
								
				html_page.set_list(html_list, sec_name='Top 10 distance for {}'.format(person_id))
			html_page.write_html()
			
		return
			
			
			
	def predict2(self, model_path, predictor_path, clip_name='clip_00005'):
		
		self._set_model(grad_clip=5)
		self.set_movie_data()

		with self.sess.as_default():
			self.restore_model(self.sess, self.saver, model_path)
			
			self._set_predictor2()
			with self.pred_sess.as_default():
			
				self.restore_model(self.pred_sess, self.pred_saver, predictor_path)
			
				frame_images, face_images, face_dir = self.movie_data.get_clip(clip_name)
				for face_i in range(len(face_dir)):
				
					face_img = face_images[face_i,::]
					face_img = face_img.reshape(1, self.img_size, self.img_size, 1 if self.gray else 3)
					
					min_avg_prob = 1.0
					min_person = None

					for imdb in self.movie_data.IMDB_data.keys():
					
						imdb_images = self.movie_data.get_images_from_path(self.movie_data.IMDB_data[imdb], self.img_size, self.gray)
						
						if type(imdb_images) == type(None):
							continue
						
						
						face_img_repeat = np.repeat(face_img, imdb_images.shape[0], axis=0)
					
						batch_images = np.vstack([imdb_images, face_img_repeat, np.zeros(imdb_images.shape)])
						
						embeddings, cost = self.sess.run(
							[self.embeddings, self.total_loss],
							feed_dict={
								self.images: batch_images,
								#self.labels: train_batch['labels'].reshape(batch_size, 1),
								self.decay_buff: 0,
								self.phase_train: True
							}
						)
						
						embeddings = np.vstack(np.vsplit(embeddings, 3)[0:2])
						
						pred_class, acc, prob, dist = self.pred_sess.run(
							[self.pred_class, self.accuracy, self.pred,
							self.pred_dist],
							feed_dict={
								self.pred_embeddings: embeddings,
								#self.labels: train_batch['labels'].reshape(batch_size, 1),
								self.pred_decay_buff: 0,
								self.predictor_label: np.zeros((imdb_images.shape[0]))
							}
						)
						#import pdb
						#pdb.set_trace()
						if np.average(prob) < min_avg_prob:
							min_person = imdb
							min_avg_prob = np.average(prob)
					
					buff = face_dir[face_i][face_dir[face_i].find(clip_name)+11:face_dir[face_i].rfind('.png')]
					print '{}: min person: {}, min prob: {:.4f}'.format(buff, min_person, min_avg_prob)
						
			
	def _set_predictor2(self, threshold=0.3):
		with tf.Graph().as_default():
		
			tf.set_random_seed(self.seed)
		
			self.pred_embeddings = tf.placeholder(tf.float32, [None, 896])
			self.pred_embed = slim_predict.fully_connected(self.pred_embeddings, 128, activation_fn=tf.nn.relu,
												   scope='fc_pred', reuse=False)
												   
			self.pred_embed = slim.fully_connected(self.pred_embed,
												   128, activation_fn=None,
												   scope='embeddings', reuse=False)

			self.pred_embed = tf.nn.l2_normalize(self.pred_embed, 1, 1e-10, name='embeddings')
			self.embed_A, self.embed_B = tf.split(0, 2, self.pred_embed)
			
			self.pred_dist = tf.reduce_sum(tf.square(self.embed_A-self.embed_B), 1) - threshold
			#self.pred_dist = slim_predict.fully_connected(self.pred_dist, 1, activation_fn=None,
			#									   scope='Weighted_dist', reuse=False)
									   
			self.predictor_label = tf.cast(tf.placeholder(tf.int64, [None]), tf.float32)
			self.pred_dist = tf.reshape(self.pred_dist, [-1])

			self.pred_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.pred_dist, self.predictor_label))
			self.pred = tf.sigmoid(self.pred_dist)
			
			self.pred_class = tf.sign(self.pred-0.5)
			self.gt_class = tf.to_float(self.predictor_label) * 2.0 - 1.0
			self.accuracy  = tf.reduce_mean(tf.cast(tf.equal(self.pred_class, self.gt_class), tf.float32))

			self.pred_decay_buff = tf.cast(tf.placeholder(tf.int32), tf.float32)
			self.pred_learning_rate = self.base_lr * 0.90 ** (self.pred_decay_buff//100)
			
			self.pred_optimizer = tf.train.AdamOptimizer(learning_rate=self.pred_learning_rate).minimize(self.pred_loss)
			
			#self.init_op = tf.initialize_all_variables()

			self.pred_saver = tf.train.Saver()
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
			self.pred_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
			
			# Initialize variables
			self.pred_sess.run(tf.initialize_all_variables())
			self.pred_sess.run(tf.initialize_local_variables())
			
		print 'Finish loading set predictor'
			
			
	def run_predictor2(self, batch_size, model_path, iters=5000, display=10, plot_freq=100):
		
		self._set_model(grad_clip=5)
		self.set_triplet_data()

		with self.sess.as_default():
			self.restore_model(self.sess, self.saver, model_path)
			
			self._set_predictor2()
			with self.pred_sess.as_default():
			
				self.iterations = []
				self.losses_train = []
				self.accuracies_train = []
				self.losses_valid = []
				self.accuracies_valid = []

				
				
				for i in xrange(iters+1):
				
					pair_batch_TRAIN = self.get_pair_batch(batch_size, 'train')
					#pair_batch = self.get_pair_batch(batch_size, 'test')
					
					#import random
					#pair_batch_TRAIN = random.choice([pair_batch_TRAIN, pair_batch])
					batch_images = np.vstack([pair_batch_TRAIN['imagesA'],
											pair_batch_TRAIN['imagesB'],
											np.zeros(pair_batch_TRAIN['imagesA'].shape)])
											
					embeddings, cost = self.sess.run(
						[self.embeddings, self.total_loss],
						feed_dict={
							self.images: batch_images,
							#self.labels: train_batch['labels'].reshape(batch_size, 1),
							self.decay_buff: i,
							self.phase_train: True
						}
					)
					
					embeddings = np.vstack(np.vsplit(embeddings, 3)[0:2])
					
					_, loss, pred_class, acc, prob, emA, emB, dist = self.pred_sess.run(
						[self.pred_optimizer, self.pred_loss, self.pred_class, self.accuracy, self.pred,
						self.embed_A, self.embed_B, self.pred_dist],
						feed_dict={
							self.pred_embeddings: embeddings,
							#self.labels: train_batch['labels'].reshape(batch_size, 1),
							self.pred_decay_buff: i,
							self.predictor_label: pair_batch_TRAIN['labels']
						}
					)
					

					#import pdb
					#pdb.set_trace()
					if i % display == 0:
						#print prob
						print '[TRAIN]: Iter={}, loss={:.4f}, acc={:.4f}'.format(i, loss, acc)
						pair_batch = self.get_pair_batch(batch_size, 'train')
						batch_images = np.vstack([pair_batch['imagesA'],
												pair_batch['imagesB'],
												np.zeros(pair_batch['imagesA'].shape)])
												
						embeddings, cost = self.sess.run(
							[self.embeddings, self.total_loss],
							feed_dict={
								self.images: batch_images,
								#self.labels: train_batch['labels'].reshape(batch_size, 1),
								self.decay_buff: i,
								self.phase_train: True
							}
						)
						
						embeddings = np.vstack(np.vsplit(embeddings, 3)[0:2])
												
						loss_valid, pred_class, acc_valid, prob = self.pred_sess.run(
							[self.pred_loss, self.pred_class, self.accuracy, self.pred],
							feed_dict={
								self.pred_embeddings: embeddings,
								#self.labels: train_batch['labels'].reshape(batch_size, 1),
								self.pred_decay_buff: i,
								self.predictor_label: pair_batch['labels']
							}
						)
						
						print '[VALID]: Iter={}, loss={:.4f}, acc={:.4f}'.format(i, loss_valid, acc_valid)
					
						self.iterations.append(i)
						self.losses_train.append(loss)
						self.accuracies_train.append(acc)
						self.losses_valid.append(loss_valid)
						self.accuracies_valid.append(acc_valid) 
						
						pair_batch = self.get_pair_batch(batch_size, 'test')
						batch_images = np.vstack([pair_batch['imagesA'],
												pair_batch['imagesB'],
												np.zeros(pair_batch['imagesA'].shape)])
												
						embeddings, cost = self.sess.run(
							[self.embeddings, self.total_loss],
							feed_dict={
								self.images: batch_images,
								#self.labels: train_batch['labels'].reshape(batch_size, 1),
								self.decay_buff: i,
								self.phase_train: True
							}
						)
						
						embeddings = np.vstack(np.vsplit(embeddings, 3)[0:2])
												
						loss_test, pred_class, acc_test, prob = self.pred_sess.run(
							[self.pred_loss, self.pred_class, self.accuracy, self.pred],
							feed_dict={
								self.pred_embeddings: embeddings,
								#self.labels: train_batch['labels'].reshape(batch_size, 1),
								self.pred_decay_buff: i,
								self.predictor_label: pair_batch['labels']
							}
						)
						
						print '[ TEST]: Iter={}, loss={:.4f}, acc={:.4f}\n'.format(i, loss_test, acc_test)
					
					if i % max(100,display) == 0 and i != 0:
						
						import matplotlib.pyplot as plt
						plt.switch_backend('agg')

						plt.close('all')
						
						train_line, = plt.plot(self.iterations, self.losses_train, label='Train')
						valid_line, = plt.plot(self.iterations, self.losses_valid, label='Valid')
						plt.legend(handles=[train_line], loc=1)
						plt.legend(handles=[valid_line], loc=1)
						plt.xlabel('Iteration')
						plt.ylabel('Loss')
						plt.savefig('{}/loss_{}.png'.format(self.folder_path, i))
						
						plt.close('all')
						train_line, = plt.plot(self.iterations, self.accuracies_train, label='Train')
						valid_line, = plt.plot(self.iterations, self.accuracies_valid, label='Valid')
						plt.legend(handles=[train_line], loc=1)
						plt.legend(handles=[valid_line], loc=1)
						plt.xlabel('Iteration')
						plt.ylabel('Accuracy')
						plt.savefig('{}/acc_{}.png'.format(self.folder_path, i))
						
						save_path = self.pred_saver.save(self.pred_sess, 
											"{}/model_{}.ckpt".format(self.model_path, i))
						print("Model saved in file: %s" % save_path)
					
	
		
	def run_predictor(self, batch_size, model_path, iters=5000, display=10, plot_freq=100):

		self._set_predictor()
		self.set_triplet_data()
		
		with self.pred_sess.as_default():

			if model_path:
				self.restore_model(self.sess, model_path)
				
			self.iterations = []
			self.losses_train = []
			self.accuracies_train = []
			self.losses_valid = []
			self.accuracies_valid = []

			for i in xrange(iters+1):

				self.iterations = []
				self.losses_train = []
				self.accuracies_train = []
				self.losses_valid = []
				self.accuracies_valid = []

				pair_batch = self.get_pair_batch(batch_size, 'train')
				batch_images = np.vstack([pair_batch['imagesA'],
								 pair_batch['imagesB']])
								 
				
				_, loss, prob, acc = self.pred_sess.run(
					[self.pred_op, self.pred_total_loss, self.prob, self.accuracy],
					feed_dict={
						self.pred_images: batch_images,
						self.predictor_label: pair_batch['labels'],
						self.decay_buff: i,
						self.phase_train: True
					}
				)
				
				if i % display == 0:
					print prob
					print '[TRAIN]: Iter={}, loss={:.4f}, acc={:.4f}'.format(i, loss, acc)
					pair_batch = self.get_pair_batch(batch_size, 'valid')
					batch_images = np.vstack([pair_batch['imagesA'],
									 pair_batch['imagesB']])
									 
					loss_valid, acc_valid = self.pred_sess.run(
						[self.pred_total_loss, self.accuracy],
						feed_dict={
							self.pred_images: batch_images,
							self.predictor_label: pair_batch['labels'],
							self.decay_buff: i,
							self.phase_train: True
						}
					)
					
					print '[VALID]: Iter={}, loss={:.4f}, acc={:.4f}'.format(i, loss_valid, acc_valid)
					
					self.iterations.append(i)
					self.losses_train.append(loss)
					self.accuracies_train.append(acc)
					self.losses_valid.append(loss_valid)
					self.accuracies_valid.append(acc_valid) 
					
				if i % max(500,display) == 0 and i != 0:
					import matplotlib.pyplot as plt
					plt.switch_backend('agg')

					plt.close('all')
					
					train_line, = plt.plot(self.iterations, self.losses_train, label='Train')
					valid_line, = plt.plot(self.iterations, self.losses_valid, label='Valid')
					plt.legend(handles=[train_line], loc=1)
					plt.legend(handles=[valid_line], loc=1)
					plt.xlabel('Iteration')
					plt.ylabel('Loss')
					plt.savefig('{}/loss_{}.png'.format(self.folder_path, i))
					
					plt.close('all')
					train_line, = plt.plot(self.iterations, self.accuracies_train, label='Train')
					valid_line, = plt.plot(self.iterations, self.accuracies_valid, label='Valid')
					plt.legend(handles=[train_line], loc=1)
					plt.legend(handles=[valid_line], loc=1)
					plt.xlabel('Iteration')
					plt.ylabel('Accuracy')
					plt.savefig('{}/acc_{}.png'.format(self.folder_path, i))
					
					save_path = self.pred_saver.save(self.sess, 
										"{}/model_{}.ckpt".format(self.model_path, i))
					print("Model saved in file: %s" % save_path)
					
					#from html_dashboard.codes.main import HTMLFramework
					
					#html_page = HTMLFramework('performance_{}'.format(i),
					#					 	html_folder = self.folder_path,
					#					 	page_title='Performance on {} iterations'.format(i))
					
					#m = 30
					#imA = saveImagesFromMatrix(self.folder_path, train_batch['imagesA'],
					#						   prefix='{}_{}/trainAnchor'.format('images', i), m=m)
					
					#imB = saveImagesFromMatrix(self.folder_path, train_batch['imagesB'],
					#						   prefix='{}_{}/trainAnchor'.format('images', i), m=m)	
			


	def restore_model(self, sess, saver, model_path):

		#self.sess.run(tf.initialize_all_variables())
		saver.restore(sess, model_path)
		print 'Finished restoring : {}'.format(model_path)


	def get_embedding_vec(self, pretrained, batch_size):
		

		train_batch = self.get_triplet_batch(batch_size, 'train')

		batch_images = np.vstack([train_batch['anchor_imgs'],
								 train_batch['positive_imgs'],
								 train_batch['negative_imgs']])
		

			
				
		return em_a, em_p


	def run_model(self, iters, display, batch_size, plot_freq=20, restore_model=None):

		plot_freq = min(plot_freq, display)

		self._set_model(grad_clip=5)
		self.set_triplet_data()

		with self.sess.as_default():

			self.iterations = []
			self.losses_train = []
			self.accuracies_train = []
			self.losses_valid = []
			self.accuracies_valid = []

			#train_batch = self.get_triplet_batch(batch_size, 'train')
			
			if restore_model:
				self.restore_model(self.sess, self.saver, restore_model)

			for i in xrange(iters+1):

				step = self.sess.run(self.global_step, feed_dict=None)

				train_batch = self.get_triplet_batch(batch_size, 'train', flip=True, whiten=True)
				
				batch_images = np.vstack([train_batch['anchor_imgs_processed'],
										 train_batch['positive_imgs_processed'],
										 train_batch['negative_imgs_processed']])

				cost, triplet_loss,  _, lr, pos_dist, neg_dist, em_a, em_p, em_n = self.sess.run(
					[self.total_loss, self.triplet_loss, self.train_op, self.learning_rate, self.pos_dist,
					self.neg_dist, self.embed_anchor, self.embed_pos, self.embed_neg],
					feed_dict={
						self.images: batch_images,
						#self.labels: train_batch['labels'].reshape(batch_size, 1),
						self.decay_buff: i,
						self.phase_train: True
					}
				)

				if i % plot_freq == 0:
					valid_batch = self.get_triplet_batch(batch_size, 'test')
					batch_images = np.vstack([valid_batch['anchor_imgs_processed'],
											 valid_batch['positive_imgs_processed'],
											 valid_batch['negative_imgs_processed']])

					cost_valid, triplet_loss_valid, lr, pos_dist_valid, neg_dist_valid = self.sess.run(
						[self.total_loss, self.triplet_loss, self.learning_rate, self.pos_dist,
						self.neg_dist],
						feed_dict={
							self.images: batch_images,
							self.decay_buff: i,
							self.phase_train: True
						}
					)

					self.iterations.append(i)
					self.losses_train.append(cost)
					#self.accuracies_train.append(acc)
					self.losses_valid.append(cost_valid)
					#self.accuracies_valid.append(acc_valid)


				if i % display == 0:
					with open('{}/log.txt'.format(self.folder_path), 'ab') as f:
						#prediction = np.int32((prediction+1)/2).reshape(batch_size)
						#prediction_valid = np.int32((prediction_valid+1)/2).reshape(batch_size)
						#pdb.set_trace()
						print 'Step: {} lr: {}'.format(i, lr)
						print '[TRAIN] cost:{}'.format(cost)
						#print 'pred: {}'.format(str(prediction.reshape(batch_size)))
						#print 'real: {}'.format(str(train_batch['labels']))
						print '[VALID] cost:{}'.format(cost_valid)
						#print 'pred: {}'.format(str(prediction_valid.reshape(batch_size)))
						#print 'real: {}\n'.format(str(valid_batch['labels']))

						f.write('Step: {} lr: {}\n'.format(i, lr))
						f.write('[TRAIN] cost:{}\n'.format(cost))
						#f.write('pred_class: {}\n'.format(str(prediction)))
						#f.write('real: {}\n'.format(str(train_batch['labels'])))
						f.write('[VALID] cost:{}\n'.format(cost_valid))
						#f.write('pred_class: {}\n'.format(str(prediction_valid)))
						#f.write('real: {}\n\n'.format(str(valid_batch['labels'])))


				if i % max(200,display) == 0 and i != 0:
					import matplotlib.pyplot as plt
					plt.switch_backend('agg')

					plt.close('all')
					
					train_line, = plt.plot(xrange(0, i+1, plot_freq), self.losses_train, label='Train')
					valid_line, = plt.plot(xrange(0, i+1, plot_freq), self.losses_valid, label='Valid')
					plt.legend(handles=[train_line], loc=1)
					plt.legend(handles=[valid_line], loc=1)
					plt.xlabel('Iteration')
					plt.ylabel('Loss')
					plt.savefig('{}/loss_{}.png'.format(self.folder_path, i))
					'''
					plt.close('all')
					train_line, = plt.plot(xrange(0, i+1, plot_freq), self.accuracies_train, label='Train')
					valid_line, = plt.plot(xrange(0, i+1, plot_freq), self.accuracies_valid, label='Valid')
					plt.legend(handles=[train_line], loc=1)
					plt.legend(handles=[valid_line], loc=1)
					plt.xlabel('Iteration')
					plt.ylabel('Accuracy')
					plt.savefig('{}/acc_{}.png'.format(self.folder_path, i))
					'''

					save_path = self.saver.save(self.sess, "{}/model_{}.ckpt".format(self.model_path, i))
					print("Model saved in file: %s" % save_path)

					from util import *
					directory_i = self.setFileDirectory(self.folder_path, '{}_{}'.format('images', i))
					from html_dashboard.codes.main import HTMLFramework

					m = 30

					html_page = HTMLFramework('performance_{}'.format(i),
										 	html_folder = self.folder_path,
										 	page_title='Performance on {} iterations'.format(i))
											
					anchor_imgs = self.data.get_images_from_path(train_batch['anchor'], flip=False, whiten=False)
					pos_imgs = self.data.get_images_from_path(train_batch['positive'], flip=False, whiten=False)
					neg_imgs = self.data.get_images_from_path(train_batch['negative'], flip=False, whiten=False)
					
					imA = saveImagesFromMatrix(self.folder_path, anchor_imgs,
											   prefix='{}_{}/trainAnchor'.format('images', i), m=m)
					imB = saveImagesFromMatrix(self.folder_path, pos_imgs, 
											   prefix='{}_{}/trainPos'.format('images', i), m=m)
					imC = saveImagesFromMatrix(self.folder_path, neg_imgs, 
											   prefix='{}_{}/trainNeg'.format('images', i), m=m)
					
					html_page.set_list([
										'Step: {} lr: {}'.format(i, lr),
										'[TRAIN] cost:{}'.format(cost),
										#'pred: {}'.format(str(prediction.reshape(batch_size))),
										#'real: {}'.format(str(train_batch['labels'])),
										'[VALID] cost:{}'.format(cost_valid)],
										#'pred: {}'.format(str(prediction_valid.reshape(batch_size))),
										#'real: {}\n'.format(str(valid_batch['labels']))], 
										sec_name='Model Details')

					html_page.set_image('loss_{}.png'.format(i))
					#html_page.set_image('acc_{}.png'.format(i))
					
					captions = []
					#prob = prob.reshape(batch_size)
					
					#for j in range(m):
					#	captions.append(['real label: {}'.format(train_batch['labels'][j]),
					#					 'distance: {:.4f}'.format(dist[j])])
										 #'pred label: {}'.format(prediction[j]),
										 #'class 1 prob: {:.4f}'.format(prob[j])])
					

					#html_page.set_siamese_image_table(imA, imB, captions=captions, sec_name='Training Demo')
					images = []
					for i in range(len(imA)):
						images += [imA[i], imB[i], imC[i]]
						captions.append([''])
						captions.append(['dist: {:.4f}'.format(pos_dist[i])])
						captions.append(['dist: {:.4f}'.format(neg_dist[i])])

					html_page.set_image_table(images, captions=captions, num_col=3, sec_name='Train images')

					html_page.write_html()


