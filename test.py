#-*- coding: utf-8 -*-
from __future__ import division
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle
from keras.preprocessing import sequence

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
	def __init__(self, word_to_idx, dim_ctx=2048, dim_embed=256, dim_hidden=256,\
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

		ctx_shape = [20,2048]
		# self.n_words = n_words
		self.dim_embed = dim_embed
		self.dim_ctx = dim_ctx
		self.dim_hidden = dim_hidden
		self.ctx_shape = ctx_shape
		self.n_lstm_steps = n_lstm_steps
		#self.batch_size = batch_size

		#self.weight_initializer = tf.contrib.layers.xavier_initializer()
		self.weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
		self.const_initializer = tf.constant_initializer(0.0)
		self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

		# Place holder for features and captions
		#self.features = tf.placeholder(tf.float32, [self.batch_size, self.L, self.D])
		#self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

	def _get_initial_lstm(self, features):
		with tf.variable_scope('initial_lstm'):
			features_mean = tf.reduce_mean(features, 1)
            # change self.D to self.M
			w_h = tf.get_variable('w_h', [self.dim_ctx, self.dim_hidden], initializer=self.weight_initializer)
			b_h = tf.get_variable('b_h', [self.dim_hidden], initializer=self.const_initializer)
			h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

			w_c = tf.get_variable('w_c', [self.dim_ctx, self.dim_hidden], initializer=self.weight_initializer)
			b_c = tf.get_variable('b_c', [self.dim_hidden], initializer=self.const_initializer)
			c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
			return c, h

	def _word_embedding(self, inputs, reuse=False):
		with tf.variable_scope('word_embedding', reuse=reuse):
			# change self.V to 1
			w = tf.get_variable('w', [20, self.dim_embed], initializer=self.emb_initializer)
			x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
			return x

	def _attention_layer(self, features, features_proj, h, reuse=False):
	with tf.variable_scope('attention_layer', reuse=reuse):
		w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
		b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
		w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

		h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
		out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
		alpha = tf.nn.softmax(out_att)  
		context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
		return context, alpha

	def _attention_layer(self, features, features_proj, h, reuse=False):
	with tf.variable_scope('attention_layer', reuse=reuse):
		w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
		b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
		w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

		h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
		out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
		alpha = tf.nn.softmax(out_att)  
		context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
		return context, alpha

	def build_model(self):
		batch_size = tf.shape(features)[0]
		features = tf.placeholder("float32", [batch_size, self.ctx_shape[0], self.ctx_shape[1]])
		mask = tf.placeholder("float32", [self.batch_size, self.n_lstm_steps])

		features = self._word_embedding(features)
		c, h = self._get_initial_lstm(features=features)
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_hidden)
		loss = 0.0
		for inx in range(self.n_lstm_steps):
			with tf.variable_scope('lstm', reuse=(t!=0)):
				output, _, _ = tf.nn.bidirectional_rnn(lstm_cell,lstm_cell,features)

