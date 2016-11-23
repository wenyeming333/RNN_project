#-*- coding: utf-8 -*-
from __future__ import division
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle
from keras.preprocessing import sequence

from variables import *

slim = tf.contrib.slim 
slim_predict = tf.contrib.slim
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
		self.N = 10
		self.dim_embed = dim_embed
		self.player_feature_shape = [None, 10, 131072]
		self.dim_ctx = dim_ctx
		self.dim_hidden = dim_hidden
		self.n_lstm_steps = n_lstm_steps

		self.weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
		self.const_initializer = tf.constant_initializer(0.0)
		self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
		self.reuse = -1
		self.set_data()
		
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
		with tf.variable_scope('initial_lstm{}'.format(mode)):
			w_h = tf.get_variable('w_h{}'.format(mode), [self.dim_embed*2, self.dim_hidden], initializer=self.weight_initializer)
			b_h = tf.get_variable('b_h{}'.format(mode), [self.dim_hidden], initializer=self.const_initializer)
			h = tf.nn.tanh(tf.matmul(hidden_features, w_h) + b_h)

			#w_c = tf.get_variable('w_c{}'.format(mode), [self.dim_ctx, self.dim_hidden], initializer=self.weight_initializer)
			w_c = tf.get_variable('w_c{}'.format(mode), [self.dim_embed*2, self.dim_hidden], initializer=self.weight_initializer)
			b_c = tf.get_variable('b_c{}'.format(mode), [self.dim_hidden], initializer=self.const_initializer)
			c = tf.nn.tanh(tf.matmul(hidden_features, w_c) + b_c)
			return c, h
		

	def _frame_embedding(self, inputs, reuse=False):
		with tf.variable_scope('frame_embedding', reuse=reuse):
	
			w = tf.get_variable('w_f', [self.ctx_shape[1], self.dim_embed], initializer=self.emb_initializer)
			
			b = tf.Variable(tf.constant(0.0, shape=[self.dim_embed]))
			inputs = tf.reshape(inputs, (-1, self.ctx_shape[1]))
			x = tf.nn.relu(tf.matmul(inputs, w)+b, name='frame_vector')
			
			return tf.reshape(x, (-1, self.ctx_shape[0], self.dim_embed))

	# dimension!!!!!!! [20,10,self.dim_embed] ????????????
	def _player_embedding(self, inputs, reuse=False):
		with tf.variable_scope('player_embedding', reuse=reuse):
	
			w = tf.get_variable('w_p', [self.player_feature_shape[2], self.dim_embed], initializer=self.emb_initializer)
			b = tf.Variable(tf.constant(0.0, shape=[self.dim_embed]))
			inputs = tf.reshape(inputs, (-1, self.player_feature_shape[2]))
			#w = tf.get_variable('w_p', [10, self.dim_embed], initializer=self.emb_initializer)
			#x = tf.nn.embedding_lookup(w, inputs, name='player_vector')  # (N, T, M) or (N, M)
			x = tf.nn.relu(tf.matmul(inputs, w)+b, name='player_vector')
			return tf.reshape(x, (-1, self.ctx_shape[0], self.dim_embed))

	def _attention_layer(self, features, reuse=False):
		with tf.variable_scope('attention_layer', reuse=reuse):
			# Remember to change the dimension
			# N is the number of players (10)
			# features is of [None, N], depend on the hidden state from RNN.
			# Multiplied by scalar 4 because we have fw, bw, and event hidden features.
			w = tf.get_variable('w_a', [4 * self.dim_hidden, self.N], initializer=self.weight_initializer)
			b = tf.Variable(tf.constant(0.0, shape=[self.N]))
			# v = tf.get_variable('v_a', [self.dim_hidden, self.N], initializer=self.weight_initializer)
			features = tf.reshape(features, (-1, 4 * self.dim_hidden))
			# gamma = tf.nn.softmax(tf.batch_matmul(tf.tanh(tf.matmul(features, w) + b), v))
			return tf.nn.softmax(tf.tanh(tf.matmul(features, w) + b))

	def _prediction_layer(self, features, reuse=False):
		with tf.variable_scope('prediction_layer', reuse=reuse):
			# We have 11 classes of events.
			w = tf.get_variable('w_p', [self.dim_hidden, 11], initializer=self.weight_initializer)
			return tf.matmul(features, w)

	def build_model(self):
		#batch_size = tf.shape(features)[0]
		self.features = tf.placeholder(tf.float32, [None, self.ctx_shape[0], self.ctx_shape[1]])
		self.player_features = tf.placeholder(tf.float32, self.player_feature_shape)
		self.labels = tf.placeholder(tf.float32, [None, 11])
		# mask = tf.placeholder("float32", [self.batch_size, self.n_lstm_steps])

		self.em_frame = self._frame_embedding(self.features)
		self.em_player = self._player_embedding(self.player_features)
		#reversed_features = tf.nn.rnn._reverse_seq(features, self.ctx_shape[0], 1, batch_dim=0)
		self.sequence_lengths = tf.placeholder(tf.int64, [None])
		
		reversed_features = tf.reverse_sequence(self.em_frame, self.sequence_lengths, 1, batch_dim=0)

		self.c1, self.h1 = self._get_initial_frame_lstm(features=self.em_frame, mode=1)
		self.c2, self.h2 = self._get_initial_frame_lstm(features=reversed_features, mode=2) # Frame Blstm
		self.c3, self.h3 = self._get_initial_event_lstm(hidden_features=tf.concat(1, [self.h1, self.h2]),
														mode=3) # Event Lstm

		blstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden, state_is_tuple=True)
		blstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden, state_is_tuple=True) # Frame Blstm
		lstm2_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden, state_is_tuple=True) # Event Lstm
		self.loss = 0.0
		
		from util.spec_math_ops import *

		for inx in range(self.n_lstm_steps):
			reuse = (inx!=0)
			with tf.variable_scope('blstm', reuse): ## reuse means?
				with tf.variable_scope('FW') as scope:
					if reuse: scope.reuse_variables()
					_, (self.c1, self.h1) = blstm_fw_cell(self.em_frame[:,inx,:], state=(self.c1, self.h1))
				with tf.variable_scope('BW') as scope:
					if reuse: scope.reuse_variables() 
					_, (self.c2, self.h2) = blstm_bw_cell(reversed_features[:,inx,:], state=(self.c2, self.h2))
					
			self.frame_features = tf.concat(1, [self.h1, self.h2, self.h3])
			reshape_frame_features = tf.reshape(self.frame_features, (-1, 1, 768))
			self.att_features = tf.concat(2, [self.player_features, tf.tile(reshape_frame_features, [1, 10, 1])])
			gamma = self._attention_layer(self.att_features, reuse)
			
			expected_features = tf.einsum('ij,kjl->il', gamma, self.player_features)
			with tf.variable_scope('event_lstm') as scope:
				if reuse: scope.reuse_variables()
				_, (self.c3, self.h3) = lstm2_cell(expected_features, state = (self.c3, self.h3))
			self.prediction_value = self._prediction_layer(self.h3, reuse)

			#self.error_mat = tf.sub(tf.ones([self.prediction_value[0].get_shape().as_list()[0], 11]), self.labels)
			
			self.error_mat = 1 - self.labels
			self.loss += tf.square(tf.nn.relu(1 - tf.mul(self.error_mat, self.prediction_value)))

		self.loss *= 0.5
		self.loss = tf.reduce_mean(tf.reduce_sum(self.loss,1))

		return self.loss
		
	def run_model(self, **kwargs):

		# pop out parameters.
		n_epochs = kwargs.pop('n_epochs', 10)
		batch_size = kwargs.pop('batch_size', 100)
		learning_rate = kwargs.pop('learning_rate', 0.01)
		print_every = kwargs.pop('print_every', 100)
		save_every = kwargs.pop('save_every', 1)
		log_path = kwargs.pop('log_path', './log/')
		model_path = kwargs.pop('model_path', './model/')
		pretrained_model = kwargs.pop('pretrained_model', None)

		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
		if not os.path.exists(self.log_path):
			os.makedirs(self.log_path)

		# Build graphs for training model and sampling captions
		loss = self.model.build_model()
		tf.get_variable_scope().reuse_variables()

		self.optimizer = tf.train.AdamOptimizer

		# Train op
		with tf.name_scope('optimizer'):
			optimizer = self.optimizer(learning_rate = learning_rate)
			grads = tf.gradients(loss, tf.trainable_variables())
			grads_and_vars = list(zip(grads, tf.trainable_variables())) # ?????
			train_op = optimizer.apply_gradients(grads_and_vars = grads_and_vars)
		   
		# Summary op
		tf.scalar_summary('batch_loss', loss)
		for var in tf.trainable_variables():
			tf.histogram_summary(var.op.name, var)
		for grad, var in grads_and_vars:
			tf.histogram_summary(var.op.name+'/gradient', grad)
		
		summary_op = tf.merge_all_summaries()

		print "The number of epoch: %d" %n_epochs
 		print "Batch size: %d" %batch_size
		
		config = tf.ConfigProto(allow_soft_placement = True)
		#config.gpu_options.per_process_gpu_memory_fraction=0.9
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			tf.initialize_all_variables().run()
			summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
			saver = tf.train.Saver(max_to_keep=40)

			if self.pretrained_model is not None:
				print "Start training with pretrained Model.."
				saversaver.restore(sess, pretrained_model)

			prev_loss = -1
			curr_loss = 0
			start_t = time.time()

			for e in range(self.n_epochs):
				print "epoch {}".format(e)
				next_batch = self.data.next_batch_generator()
				next_batch = ["/ais/gobi4/basketball/olga_ethan_features/IKTBGSIwA_o/clip_1"] * batch_size 
				i = 0
				while next_batch:
					i += 1
					frame_features_batch = np.zeros([batch_size, self.ctx_shape[0], self.ctx_shape[1]], dtype='float32')
					player_features_batch = np.zeros([batch_size, self.ctx_shape[0], self.N, self.player_feature_shape[2]], dtype='float32')
					labels_batch = np.zeros([batch_size, 11], dtype='float32')
					seq_len_batch = 20*np.ones([batch_size])
					for i, clip_dir in enumerate(next_batch):
						new_frame_features = np.load(os.path.join(clip_dir, 'frame_features.npy'))
						new_player_features = np.swapaxis(np.load(os.path.join(clip_dir, 'player_features.npy')),0,1)
						new_event_label = np.load(os.path.join(clip_dir, 'label.npy'))
						new_seq_len = new_frame_features.shape[0]
						frame_features_batch[i,:new_frame_features.shape[0],:] = new_frame_features
						player_features_batch[i,:new_player_features[0],:np.min((new_player_features[1],10)),:] = new_player_features
						labels_batch[i,:] = new_event_label
						seq_len_batch[i] = new_seq_len
					
					# How to use feed_dict !!!!!!!!!
					feed_dict = {self.model.frame_features: frame_features_batch,\
					 self.model.player_features: player_features_batch, \
					 self.model.labels: labels_batch, self.sequence_lengths: seq_len_batch}
					_, l = sess.run([train_op, loss], feed_dict)
					curr_loss += l

					if i % 10 == 0:
						summary = sess.run(summary_op, feed_dict)
						summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

					if (i+1) % self.print_every == 0:
						print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)


				print "Previous epoch loss: ", prev_loss
				print "Current epoch loss: ", curr_loss
				print "Elapsed time: ", time.time() - start_t

				if (e+1) % self.save_every == 0:
					saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
					print "model-%s saved." %(e+1)
			
				prev_loss = curr_loss
				curr_loss = 0

# Debug
if __name__ == '__main__':
	cell = Video_Event_dectection()
	cell.build_model()

