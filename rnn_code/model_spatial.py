#-*- coding: utf-8 -*-
from __future__ import division

from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
import tensorflow as tf
import pandas as pd
import numpy as np
import cPickle
import math
import time
import pdb
import os

from variables import *

np.random.seed(111)
tf.set_random_seed(111)

# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================


class Video_Event_dectection():
	def __init__(self, dim_ctx=2048, dim_embed=256, dim_hidden=256,\
	 n_lstm_steps=20, dropout=True):

		"""
		Args:
		  word_to_idx: word-to-index mapping dictionary.
		  dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
		  dim_embed: (optional) Dimension of word embedding.
		  dim_hidden: (optional) Dimension of all hidden state.
		  n_time_step: (optional) Time step size of LSTM. 
		  prev2out: (optional) previously generated word to hidden state. (see Eq (2) for explanation)
		  ctx2out: (optional) context to hidden state (see Eq (2) for explanation)
		  alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
		  selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
		  dropout: (optional) If true then dropout layer is added.
		"""

		self.ctx_shape = [20, 2048]
		self.dim_embed = dim_embed
		self.player_feature_shape = [None, 20, 10, 2048]
		self.spatial_feature_shape = [None, 20, 10, 20, 40]
		self.dim_ctx = dim_ctx
		self.dim_hidden = dim_hidden
		self.n_lstm_steps = n_lstm_steps

		self.weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
		self.const_initializer = tf.constant_initializer(0.0)
		self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
		# self.reuse = -1
		self.set_data()
		with open('/ais/gobi4/basketball/labels_dict.pkl') as f:
			self.labels_dict = cPickle.load(f)
		with open('/ais/gobi4/basketball/olga_ethan_features/event_labels.pkl') as f:
			self.global_labels = cPickle.load(f)

		# self.batch_size = batch_size

	def set_data(self):
		from util.setData import RNNData
		self.data = RNNData(data_dir, action_path, features_dir)

	def _get_initial_frame_lstm(self, features, mode=1):
		with tf.variable_scope('initial_lstm{}'.format(mode)):
			features_mean = tf.reduce_mean(features, 1)
            
			w_h = tf.get_variable('w_h{}'.format(mode), [self.dim_embed, self.dim_hidden], initializer=self.weight_initializer)
			b_h = tf.get_variable('b_h{}'.format(mode), [self.dim_hidden], initializer=self.const_initializer)
			h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

			w_c = tf.get_variable('w_c{}'.format(mode), [self.dim_embed, self.dim_hidden], initializer=self.weight_initializer)
			b_c = tf.get_variable('b_c{}'.format(mode), [self.dim_hidden], initializer=self.const_initializer)
			c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
			return c, h 
			
	def _get_initial_event_lstm(self, hidden_features, mode=1):
		with tf.variable_scope('initial_event_lstm{}'.format(mode)):
			w_h = tf.get_variable('w_h{}'.format(mode), [self.dim_embed*2, self.dim_hidden], initializer=self.weight_initializer)
			b_h = tf.get_variable('b_h{}'.format(mode), [self.dim_hidden], initializer=self.const_initializer)
			h = tf.nn.tanh(tf.matmul(hidden_features, w_h) + b_h)

			#w_c = tf.get_variable('w_c{}'.format(mode), [self.dim_ctx, self.dim_hidden], initializer=self.weight_initializer)
			w_c = tf.get_variable('w_c{}'.format(mode), [self.dim_embed*2, self.dim_hidden], initializer=self.weight_initializer)
			b_c = tf.get_variable('b_c{}'.format(mode), [self.dim_hidden], initializer=self.const_initializer)
			c = tf.nn.tanh(tf.matmul(hidden_features, w_c) + b_c)
			return c, h
		

	def _frame_embedding(self, inputs, reuse=None):
		with tf.variable_scope('frame_embedding', reuse=reuse):
	
			w = tf.get_variable('w_f', [self.ctx_shape[1], self.dim_embed], initializer=self.emb_initializer)
			
			b = tf.Variable(tf.constant(0.0, shape=[self.dim_embed]))
			inputs = tf.reshape(inputs, (-1, self.ctx_shape[1]))
			x = tf.nn.relu(tf.matmul(inputs, w)+b, name='frame_vector')
			
			return tf.reshape(x, (-1, self.ctx_shape[0], self.dim_embed))

	# dimension!!!!!!! [None, 20,10,self.dim_embed] ????????????
	def _player_embedding(self, inputs, reuse=None):
		with tf.variable_scope('player_embedding', reuse=reuse):
			
			w = tf.get_variable('w_p', [self.player_feature_shape[3], self.dim_embed], initializer=self.emb_initializer)
			b = tf.Variable(tf.constant(0.0, shape=[self.dim_embed]))
			inputs = tf.reshape(inputs, (-1, self.player_feature_shape[3]))
			x = tf.nn.relu(tf.matmul(inputs, w)+b, name='player_vector')
			return tf.reshape(x, (-1, self.ctx_shape[0], 10, self.dim_embed))

	def _player_spatial_embedding(self, inputs, reuse=False):
		with tf.variable_scope('spatial_embedding', reuse=reuse):
			input_shape = inputs.get_shape().as_list()
			inputs = tf.reshape(inputs, (-1, input_shape[3], input_shape[4], 1))

			W_1 = tf.truncated_normal([3, 3, 1, 8], stddev=0.1)
			b_1 = tf.Variable(tf.constant(0.1, shape=[8]))
			h_conv1 = tf.nn.relu(tf.nn.conv2d(inputs, W_1, strides=[1,1,1,1], padding='SAME')+b_1)
			h_conv1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

			W_2 = tf.truncated_normal([3, 3, 8, 16], stddev=0.1)
			b_2 = tf.Variable(tf.constant(0.1, shape=[16]))
			h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_2, strides=[1,1,1,1], padding='SAME')+b_2)
			h_conv2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
			h_2_shape = h_conv2.get_shape().as_list()

			h_conv2_flat = tf.reshape(h_conv2, (-1, h_2_shape[1]*h_2_shape[2]*h_2_shape[3]))
			W_fc1 = tf.truncated_normal([h_2_shape[1]*h_2_shape[2]*h_2_shape[3], self.dim_embed], stddev=0.1)
			b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.dim_embed]))
			h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
			return tf.reshape(h_fc1, (-1, input_shape[1], input_shape[2], self.dim_embed))
			

	def _attention_layer(self, features, reuse=None):
		with tf.variable_scope('attention_layer', reuse=reuse):
			# Remember to change the dimension
			# N is the number of players (10)
			# features is of [None, N], depend on the hidden state from RNN.
			# Multiplied by scalar 4 because we have fw, bw, and event hidden features.
			# fixed dimension
			w = tf.get_variable('w_a', [5 * self.dim_hidden, 1], initializer=self.weight_initializer)
			b = tf.Variable(tf.constant(0.0, shape=[1]))
			features = tf.reshape(features, (-1, 5 * self.dim_hidden))
			em_features = tf.matmul(features, w) + b
			em_features = tf.reshape(em_features, (-1, 10))
			player_weights = tf.nn.softmax(em_features)
			# tf.histogram_summary('player_attention', player_weights)
			return player_weights

	def _prediction_layer(self, features, reuse=None):
		with tf.variable_scope('prediction_layer', reuse=reuse):
			# We have 11 classes of events.
			w_p = tf.get_variable('w_p', [self.dim_hidden, 11], initializer=self.weight_initializer)
			b_p = tf.get_variable('b_p', [11], initializer=self.const_initializer)
			pre_act = tf.matmul(features, w_p) + b_p
			tf.histogram_summary('pre-prediction', pre_act)
			act = tf.nn.relu(pre_act)
			return pre_act

	def build_model(self):
		self.features = tf.placeholder(tf.float32, [None, self.ctx_shape[0], self.ctx_shape[1]])
		self.player_features = tf.placeholder(tf.float32, self.player_feature_shape)
		self.spatial_features = tf.placeholder(tf.float32, self.spatial_feature_shape)

		# self.player_features = []
		# for i in range(self.player_feature_shape[1]):
		# 	# Need to change dimensions later.
		# 	eval('self.player_features_{} = tf.placeholder(tf.float32, [self.batch_size, None, 2048])'.format(i))
		# 	self.player_features.append(eval('self.player_features_{}'.format(i)))

		self.labels = tf.placeholder(tf.float32, [None, 11])

		self.em_frame = self._frame_embedding(self.features)
		self.em_player = self._player_embedding(self.player_features)
		self.em_spatial = self._player_spatial_embedding(self.spatial_features)
		self.em_player = tf.concat(3, [self.em_player, self.em_spatial])

		self.sequence_lengths = tf.placeholder(tf.int64, [None])
		
		reversed_features = tf.reverse_sequence(self.em_frame, self.sequence_lengths, 1, batch_dim=0)

		self.c1, self.h1 = self._get_initial_frame_lstm(features=self.em_frame, mode=1)
		self.c2, self.h2 = self._get_initial_frame_lstm(features=reversed_features, mode=2) # Frame Blstm
		self.c3, self.h3 = self._get_initial_event_lstm(hidden_features=tf.concat(1, [self.h1, self.h2]),
														mode=1) # Event Lstm

		blstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden, state_is_tuple=True)
		blstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden, state_is_tuple=True) # Frame Blstm
		lstm2_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden, state_is_tuple=True) # Event Lstm
		self.loss = 0.0
		
		for inx in range(self.n_lstm_steps):
			reuse = (inx!=0)
			
			with tf.variable_scope('blstm', reuse): ## reuse means?
				with tf.variable_scope('FW') as scope:
					if reuse: scope.reuse_variables()
					_, (self.c1, self.h1) = blstm_fw_cell(self.em_frame[:,inx,:], state=(self.c1, self.h1))
				with tf.variable_scope('BW') as scope:
					if reuse: scope.reuse_variables()
					_, (self.c2, self.h2) = blstm_bw_cell(reversed_features[:,inx,:], state=(self.c2, self.h2))
					
			self.frame_features = tf.concat(1, [self.h1, self.h2, self.h3]) # It actually event and frame features together.

			reshape_frame_features = tf.reshape(self.frame_features, (-1, 1, 768))

			self.att_features = tf.concat(2, [self.em_player[:,inx,:,:], tf.tile(reshape_frame_features, [1, 10, 1])])
			gamma = self._attention_layer(self.att_features, reuse)
			
			new_gamma = tf.expand_dims(gamma, 2)
			expected_features = tf.reduce_sum(tf.mul(new_gamma, self.em_player[:,inx,:,:]), 1)
			
			with tf.variable_scope('event_lstm') as scope:
				if reuse: scope.reuse_variables()
				_, (temp_1, temp_2) = lstm2_cell(expected_features, state = (self.c3, self.h3))
				indicator = 1 - tf.cast((inx >= self.sequence_lengths), tf.float32)
				self.c3, self.h3 = self.c3 + tf.transpose(tf.mul(tf.transpose(temp_1), indicator)), \
				 self.h3 + tf.transpose(tf.mul(tf.transpose(temp_2), indicator))
				# _, (self.c3, self.h3) = lstm2_cell(expected_features, state = (self.c3, self.h3))

		self.prediction_value = self._prediction_layer(self.h3)

		with tf.name_scope('cross_entropy'):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction_value, self.labels))
		
		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_prediction'):
				self.correct_prediction = tf.equal(tf.argmax(self.prediction_value, 1), tf.argmax(self.labels, 1))
			with tf.name_scope('accuracy'):
				self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
				
		tf.scalar_summary('batch_loss', self.loss)
		tf.scalar_summary('accuracy', self.accuracy)
		
	def run_model(self, **kwargs):

		# pop out parameters.
		n_epochs = kwargs.pop('n_epochs', 200)
		batch_size = kwargs.pop('batch_size', 128)
		learning_rate = kwargs.pop('learning_rate', 0.01)
		print_every = kwargs.pop('print_every', 1)
		save_every = kwargs.pop('save_every', 10)
		val_every = kwargs.pop('val_every', 150000000000000000000)
		log_path = kwargs.pop('log_path', 'RNN_logff/')
		model_path = kwargs.pop('model_path', 'RNN_modelff/')
		pretrained_model = kwargs.pop('pretrained_model', None)
		model_name = kwargs.pop('model_name', 'RNN_model')

		# Debug use
		# ins = np.random.choice(np.arange(600), 10, replace=False)

		with open('current_videos_clips_new.pkl', 'rb') as f:
			clips_total = sorted(cPickle.load(f))
			clips_total.remove('/ais/gobi4/basketball/olga_ethan_features/-VcfnuYRhMU/clip_34')
			current_videos_clips = np.array(clips_total)


		train, val = train_test_split(np.arange(len(current_videos_clips)),\
		 train_size = 0.8, random_state = 111)
		train_files = current_videos_clips[train]
		val_files = current_videos_clips[val]
		print 'number of videos in total: ', len(current_videos_clips)
		print 'number of videos for training: ', len(train_files)
		print 'number of videos for validation: ', len(val_files)

		if not os.path.exists(model_path):
			os.makedirs(model_path)
		if not os.path.exists(log_path):
			os.makedirs(log_path)

		# Build graphs for training model.
		self.build_model()

		global_step = tf.Variable(0, name='global_step', trainable=False)
		self.learning_rate = tf.train.exponential_decay(learning_rate, global_step, \
		 	150, 0.96, staircase=False)

		tf.scalar_summary('global_step', global_step)
		tf.scalar_summary('learning_rate', self.learning_rate)

		train_op = tf.train.AdamOptimizer(learning_rate = \
			self.learning_rate).minimize(self.loss, global_step = global_step)

		tf.get_variable_scope().reuse_variables()

		# Train op
		# self.optimizer = tf.train.AdamOptimizer
		# with tf.name_scope('optimizer'):
		# 	optimizer = self.optimizer(learning_rate = self.learning_rate)
		# 	grads = tf.gradients(self.loss, tf.trainable_variables())
		# 	grads_and_vars = list(zip(grads, tf.trainable_variables())) # ?????
		# 	train_op = optimizer.apply_gradients(grads_and_vars = grads_and_vars)
		   
		# Summary op
		# tf.sclar learning rate

		#for var in tf.trainable_variables():
		#	tf.histogram_summary(var.op.name, var)
		#for grad, var in grads_and_vars:
		#	tf.histogram_summary(var.op.name+'/gradient', grad)
		
		summary_op = tf.merge_all_summaries()

		print "The number of epoch: %d" %n_epochs
 		print "Batch size: %d" %batch_size
		
		config = tf.ConfigProto()
		config.gpu_options.allocator_type = 'BFC'
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			tf.initialize_all_variables().run()
			# summary_writer means train_writer here.
			summary_writer = tf.train.SummaryWriter(log_path+'train/', graph=tf.get_default_graph())
			val_writer = tf.train.SummaryWriter(log_path+'val/', graph=tf.get_default_graph())
			saver = tf.train.Saver(max_to_keep=40)

			if pretrained_model is not None:
				print "Start training with pretrained Model.."
				saver.restore(sess, pretrained_model)

			prev_loss = -1.0
			curr_loss = 0.0
			start_t = time.time()

			n_iters_per_epoch = len(train_files) // batch_size

			val_count = 0

			for e in range(n_epochs):


				print "epoch {}".format(e)

				epoch_start_time = time.time()

				epoch_acc = 0.0
				indices = np.random.permutation(len(train_files))
				current_training_files = train_files[indices]
				#next_batch = self.data.next_batch_generator(batch_size)

				i = 0
				next_batch = current_training_files[i*batch_size:min((i+1)*batch_size,len(current_training_files))]
				if len(next_batch) < batch_size: next_batch = None
				
				while not next_batch==None:
					minibatch_start_time = time.time()
					frame_features_batch = np.zeros([batch_size, self.ctx_shape[0], self.ctx_shape[1]], dtype='float32')
					player_features_batch = np.zeros([batch_size, self.ctx_shape[0], 10, self.player_feature_shape[3]], dtype='float32')
					spatial_features_batch = np.zeros([batch_size, self.ctx_shape[0], 10, 20, 40])
					labels_batch = np.zeros([batch_size, 11], dtype='float32')
					seq_len_batch = 20*np.ones([batch_size])

					for j, clip_dir in enumerate(next_batch):
						video_name, clip_id = clip_dir.split('/')[-2], clip_dir.split('/')[-1]
						class_name = self.global_labels[video_name][clip_id]
						labels_batch[j, self.labels_dict.index(class_name)] = 1
						new_frame_features = np.load(os.path.join(clip_dir, 'frame_features.npy'))
						num_frames = min(20, new_frame_features.shape[0])
						frame_features_batch[j,:num_frames,:] = new_frame_features[:num_frames,:]

						with open(os.path.join(clip_dir, 'player_features.pkl')) as f:
							new_player_features = cPickle.load(f)

						with open(os.path.join(clip_dir, 'spatial_features.pkl')) as f:
							try:
								new_spatial_features = cPickle.load(f)
							except Exception as e:
								print clip_dir
								#continue
							
						# assert len(new_player_features) == len(new_spatial_features)

						for frame_id in range(num_frames):
							temp_num_players = min(10, new_player_features[frame_id].shape[0])
							player_features_batch[j, frame_id,:temp_num_players] = new_player_features[frame_id][:temp_num_players,:]
							spatial_features_batch[j, frame_id,:temp_num_players] = new_spatial_features[frame_id][:temp_num_players,:]
						
						#num_player = new_player_features.shape[1]
						#player_features_batch[j,:min(num_frames,20),:min(num_player,10),:] = new_player_features[:min(num_frames,20),:min(num_player,10),:]
						#labels_batch[i,:] = new_event_label

						seq_len_batch[j] = num_frames


					feed_dict = {self.features: frame_features_batch,\
					 self.player_features: player_features_batch, \
					 self.labels: labels_batch, self.sequence_lengths: seq_len_batch,
					 self.spatial_features: spatial_features_batch}
					
					# for i in range(20): feed_dict[eval('self.player_features_{}'.format(i))] = player_features[i]
					# _, l, acc, lr = sess.run([train_op, self.loss, self.accuracy, self.learning_rate], feed_dict)
					_, l, acc = sess.run([train_op, self.loss, self.accuracy], feed_dict)

					epoch_acc += acc
					curr_loss += l

					if i % 1 == 0:
					 	summary = sess.run(summary_op, feed_dict)
					 	summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

					if (i+1) % print_every == 0:
						#print "[TRAIN] epoch: %d, iteration: %d (mini-batch) loss: %.5f, acc: %.5f, lr: %.5f" %(e+1, i+1, l, acc, self.learning_rate)
						print "[TRAIN] epoch: %d, iteration: %d, (mini-batch) loss: %.5f, acc: %.5f" %(e+1, i+1, l, acc)
						with open(log_path+model_name+'.log', 'ab+') as f:
							f.write("[TRAIN] epoch: %d, iteration: %d (mini-batch) loss: %.5f, curr_loss: %.5f, acc: %.5f \n" %(e+1, i+1, l, curr_loss, acc))
					

					############## Validation

					if (i+1) % val_every == 0:
						ii = 0
						val_start = time.time()
						val_l, val_acc = 0.0, 0.0

						val_files = val_files[np.random.permutation(len(val_files))]
						next_batch_val = val_files[:min(batch_size,len(val_files))]
						if len(next_batch_val) < batch_size: next_batch_val = None

						while not next_batch_val==None:
							frame_features_jj = np.zeros([batch_size, self.ctx_shape[0], self.ctx_shape[1]], dtype='float32')
							player_features_jj = np.zeros([batch_size, self.ctx_shape[0], 10, self.player_feature_shape[3]], dtype='float32')
							labels_jj = np.zeros([batch_size, 11], dtype='float32')
							seq_len_jj = 20*np.ones([batch_size])
							for jj, val_dir in enumerate(next_batch_val):
								video_name, clip_id = val_dir.split('/')[-2], val_dir.split('/')[-1]
								class_name = self.global_labels[video_name][clip_id]
								labels_jj[jj, self.labels_dict.index(class_name)] = 1						
								new_frame_features = np.load(os.path.join(val_dir, 'frame_features.npy'))
								num_frames = min(20, new_frame_features.shape[0])
								frame_features_jj[jj,:num_frames,:] = new_frame_features[:num_frames,:]

								with open(os.path.join(val_dir, 'player_features.pkl')) as f:
									new_player_features = cPickle.load(f)
								for frame_id in range(num_frames):
									temp_num_players = min(10, new_player_features[frame_id].shape[0])
									player_features_jj[jj, frame_id,:temp_num_players] = new_player_features[frame_id][:temp_num_players,:]
								seq_len_jj[jj] = num_frames

							feed_dict = {self.features: frame_features_jj,\
						 	 self.player_features: player_features_jj, \
						 	 self.labels: labels_jj, self.sequence_lengths: seq_len_jj}

						 	temp_l, temp_acc = sess.run([self.loss, self.accuracy], feed_dict)

						 	summary = sess.run(summary_op, feed_dict)
						 	val_writer.add_summary(summary, val_count)

						 	#val_l += temp_l
						 	#val_acc += temp_acc
						 	val_count += 1

							ii += 1
							next_batch_val = val_files[ii*batch_size:min((ii+1)*batch_size,len(val_files))]
							if len(next_batch_val) < batch_size: next_batch_val = None
							if ii == 4: next_batch_val = None
						
						# num_val_files = (len(val_files) // batch_size) * batch_size
						# val_l, val_acc = val_l / ii, val_acc / ii

							print 'validation takes {} seconds'.format(time.time()-val_start)

							print "[VAL] epoch: %d, iteration: %d, validation_batch: %d, val_loss: %.5f, val_acc: %.5f"\
						 	 %(e+1, i+1, ii, temp_l, temp_acc)

							with open(log_path+model_name+'.log', 'ab+') as f:
								f.write("[VAL] epoch: %d, iteration: %d, validation_batch: %d, val_loss: %.5f, val_acc: %.5f \n" \
									%(e+1, i+1, ii, temp_l, temp_acc))

					############## Validation

					i += 1
					next_batch = current_training_files[i*batch_size:min((i+1)*batch_size,len(current_training_files))]
					if len(next_batch) < batch_size: next_batch=None
				
				print "Previous epoch loss: ", prev_loss
				print "Current epoch loss: ", curr_loss
				print "Elapsed time: ", time.time() - start_t
				print "accuracy_over_batch: ", epoch_acc / n_iters_per_epoch
				with open(log_path+model_name+'.log', 'ab+') as f:
					f.write("[TRAIN] epoch: %d, epoch_loss: %.5f, epoch_acc: %.5f \n" %(e+1, curr_loss, epoch_acc / n_iters_per_epoch))

				if (e+1) % save_every == 0:
					saver.save(sess, os.path.join(model_path, 'model{}'.format(e+1)))
					print "model-%s saved." %(e+1)
			
				prev_loss = curr_loss
				curr_loss = 0.0
				print 'this epoch took {} seconds to run'.format(time.time()-epoch_start_time)

# Debug
if __name__ == '__main__':
	cell = Video_Event_dectection()
	cell.run_model()
