#-*- coding: utf-8 -*-
from __future__ import division
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle
from keras.preprocessing import sequence

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
		self.player_feature_shape = [None, 10, dim_embed]
		self.dim_ctx = dim_ctx
		self.dim_hidden = dim_hidden
		self.n_lstm_steps = n_lstm_steps

		self.weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
		self.const_initializer = tf.constant_initializer(0.0)
		self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
		self.reuse = -1

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
	
			w = tf.get_variable('w_p', [self.ctx_shape[1], self.dim_embed], initializer=self.emb_initializer)
			b = tf.Variable(tf.constant(0.0, shape=[self.dim_embed]))
			inputs = tf.reshape(inputs, (-1, self.ctx_shape[1]))
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
					_, (self.c2, self.h2) = blstm_fw_cell(reversed_features[:,inx,:], state=(self.c2, self.h2))
					
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

# Debug
if __name__ == '__main__':
	cell = Video_Event_dectection()
	cell.build_model()

