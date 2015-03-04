#!/usr/bin/env python
""" RNN + LSTM
"""
from convertData import TwoWayDict
from inputParser import get_parser
from collections import defaultdict
import sys
import gzip
import numpy as np
import theano
import theano.tensor as T

import logging
import time
import os
import datetime
import cPickle as pickle
import update_list

from utils import case_insensitive_import

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
mode = theano.Mode(linker='cvm')

class LSTM(object):
	"""	   LSTM  class
	"""
	def __init__(self, input0, input1, input2, n_in, n_hidden, n_out, ylen0 = 1, ylen1 = 1, ylen2 = 2, activation=T.tanh,
				 output_type='real', use_symbolic_softmax=False, prange = 0.01, lstm_flag = False):

		self.lstm_flag = lstm_flag
		self.output_type = output_type

		self.input0 = input0
		self.input1 = input1
		self.input2 = input2

		self.activation = activation
		self.n_out = n_out
		self.n_hidden = n_hidden

		if lstm_flag:
			# hidden to hidden connections
			W_hh_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
												  low=-.01, high=.01),
												  dtype=theano.config.floatX)
			self.W_hh = theano.shared(value=W_hh_init, name='Whh', borrow = True)

			W_hz_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
												  low=-.01, high=.01),
												  dtype=theano.config.floatX)
			self.W_hz = theano.shared(value=W_hz_init, name='Whz', borrow = True)

			W_hr_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
												  low=-.01, high=.01),
												  dtype=theano.config.floatX)
			self.W_hr = theano.shared(value=W_hr_init, name='Whr', borrow = True)


			# input to hidden layer weights
			W_xh_init = np.asarray(np.random.uniform(size=(n_in, n_hidden),
													 low=-.01, high=.01),
													 dtype=theano.config.floatX)
			self.W_xh = theano.shared(value=W_xh_init, name='Wxh', borrow = True)

			W_xz_init = np.asarray(np.random.uniform(size=(n_in, n_hidden),
													 low=-.01, high=.01),
													 dtype=theano.config.floatX)
			self.W_xz = theano.shared(value=W_xz_init, name='Wxz', borrow = True)

			W_xr_init = np.asarray(np.random.uniform(size=(n_in, n_hidden),
													 low=-.01, high=.01),
													 dtype=theano.config.floatX)
			self.W_xr = theano.shared(value=W_xr_init, name='Wxr', borrow = True)

			h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
			self.h0 = theano.shared(value=h0_init, name='h0', borrow = True)

			# biases
			bhh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
			self.bhh = theano.shared(value=bhh_init, name='bhh', borrow = True)

			bhz_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
			self.bhz = theano.shared(value=bhz_init, name='bhz', borrow = True)

			bhr_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
			self.bhr = theano.shared(value=bhr_init, name='bhr', borrow = True)

			bxh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
			self.bxh = theano.shared(value=bxh_init, name='bxh', borrow = True)

			bxz_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
			self.bxz = theano.shared(value=bxz_init, name='bxz', borrow = True)

			bxr_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
			self.bxr = theano.shared(value=bxr_init, name='bxr', borrow = True)

			# softmax layer
			W_log_init = np.asarray(np.random.uniform(size=( 3*n_hidden, n_out),
													  low=-.01, high=.01),
													  dtype=theano.config.floatX)
			self.W_log = theano.shared(value=W_log_init, name='W_log', borrow = True)

			blog_init = np.zeros((n_out,), dtype=theano.config.floatX)
			self.blog = theano.shared(value=blog_init, name='blog', borrow = True)

			# parameters
			self.params = [self.W_hh, self.W_hz, self.W_hr,  self.W_xh, self.W_xz, self.W_xr, self.W_log, self.h0, self.bhh , self.bhz, self.bhr, self.bxh, self.bxz, self.bxr, self.blog]

			def step(x_t, h_tm1):
				z = self.activation( T.dot(x_t, self.W_xz) + self.bxz + T.dot(h_tm1, self.W_hz) + self.bhz )
				r = self.activation( T.dot(x_t, self.W_xr) + self.bxr + T.dot(h_tm1, self.W_hr) + self.bhr)
				htilda = self.activation( T.dot(x_t, self.W_xh) + self.bxh + T.dot(r * h_tm1, self.W_hh) + self.bhh)
				h_t = z*h_tm1 + (1-z)*htilda
				return h_t

			self.s0h, _ = theano.scan(step, sequences=self.input0, outputs_info=[self.h0])
			self.s1h, _ = theano.scan(step, sequences=self.input1, outputs_info=[self.h0])
			self.s2h, _ = theano.scan(step, sequences=self.input2, outputs_info=[self.h0])

			self.L1 = abs(self.W_xh.sum()) + abs(self.W_xz.sum()) + abs(self.W_xr.sum()) + abs(self.W_hh.sum()) + abs(self.W_hz.sum()) + abs(self.W_hr.sum())
			self.L2_sqr = (self.W_xh ** 2).sum() + (self.W_xz ** 2).sum() + (self.W_xr ** 2).sum() + (self.W_hh ** 2).sum() + (self.W_hz ** 2).sum() + (self.W_hr ** 2).sum()

		else:
			W_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
												  low=-.01, high=.01),
												  dtype=theano.config.floatX)
			self.W = theano.shared(value=W_init, name='W', borrow = True)
			# input to hidden layer weights
			W_in_init = np.asarray(np.random.uniform(size=(n_in, n_hidden),
													 low=-.01, high=.01),
													 dtype=theano.config.floatX)
			self.W_in = theano.shared(value=W_in_init, name='W_in', borrow = True)

			h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
			self.h0 = theano.shared(value=h0_init, name='h0', borrow = True)

			bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
			self.bh = theano.shared(value=bh_init, name='bh', borrow = True)

			W_log_init = np.asarray(np.random.uniform(size=( 3*n_hidden, n_out),
													  low=-.01, high=.01),
													  dtype=theano.config.floatX)
			self.W_log = theano.shared(value=W_log_init, name='W_log', borrow = True)

			blog_init = np.zeros((n_out,), dtype=theano.config.floatX)
			self.blog = theano.shared(value=blog_init, name='blog', borrow = True)

			self.params = [self.W, self.W_in, self.W_log, self.h0, self.bh, self.blog]

			def step0(x_t, h_tm1):
				h_t = self.activation(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W) + self.bh)
				return h_t

			def step1(x_t, h_tm1):
				h_t = self.activation(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W) + self.bh)
				return h_t

			def step2(x_t, h_tm1):
				h_t = self.activation(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W) + self.bh)
				return h_t

			self.s0h, _ = theano.scan(step0, sequences=self.input0, outputs_info=[self.h0])
			self.s1h, _ = theano.scan(step1, sequences=self.input1, outputs_info=[self.h0])
			self.s2h, _ = theano.scan(step2, sequences=self.input2, outputs_info=[self.h0])

			self.L1 = abs(self.W.sum()) + abs(self.W_in.sum())
			self.L2_sqr = (self.W ** 2).sum() + (self.W_in ** 2).sum()

		self.L1 +=  abs(self.W_log.sum())
		self.L2_sqr += (self.W_log ** 2).sum()

		self.final_layer =  T.concatenate([ self.s0h[ylen0 - 1,:], self.s1h[ylen1 - 1,:], self.s2h[ylen2 - 1,:] ], axis=0)
		self.p_pred = T.nnet.softmax( T.dot( self.final_layer, self.W_log ) + self.blog)
		self.y_sentence = T.argmax( self.p_pred , axis = 1)

		self.updates = {}
		for param in self.params:
			init = np.zeros(param.get_value(borrow=True).shape,
							dtype=theano.config.floatX)
			self.updates[param] = theano.shared(init)

		self.loss = lambda y, ylen0, ylen1, ylen2 : self.nll_multiclass_l(y, ylen0, ylen1, ylen2)

	def nll_multiclass_l(self, y, ylen0, ylen1, ylen2):
		pred_loss = -T.log(self.p_pred[0,y]).sum()
		return  pred_loss

	def debug_final_layer(self, y, ylen0, ylen1, ylen2):
		return (self.p_pred,y,self.final_layer.shape)

	def errors_l(self, y, ylen0, ylen1, ylen2):
		return T.neq(y,self.y_sentence)

	def debug_len(self,ylen0):
		return ylen0

	def debug_lnorm(self):
		return self.L1 + self.L2_sqr

	def predict_instance(self,y, ylen0, ylen1, ylen2):
		return self.y_sentence

class MetaRNN(object):
	def __init__(self, n_in = 7, n_hidden = 50, n_out = 7, learning_rate = 0.01,
				 n_epochs = 100, L1_reg = 0.00001, L2_reg = 0.00001, learning_rate_decay = 1,
				 activation = 'tanh', output_type='softmax',
				 final_momentum = 0.9, initial_momentum = 0.5,
				 momentum_switchover = 5,
				 use_symbolic_softmax = False,
				 lstm_flag = False):
		self.n_in = int(n_in)
		self.n_hidden = int(n_hidden)
		self.n_out = int(n_out)
		self.learning_rate = float(learning_rate)
		self.learning_rate_decay = float(learning_rate_decay)
		self.n_epochs = int(n_epochs)
		self.L1_reg = float(L1_reg)
		self.L2_reg = float(L2_reg)
		self.activation = activation
		self.output_type = output_type
		self.initial_momentum = float(initial_momentum)
		self.final_momentum = float(final_momentum)
		self.momentum_switchover = int(momentum_switchover)
		self.use_symbolic_softmax = use_symbolic_softmax
		self.lstm_flag = lstm_flag
		self.ready()

	def ready(self):
		# input (where first dimension is time)
		self.x0 = T.matrix()
		self.x1 = T.matrix()
		self.x2 = T.matrix()

		self.y = T.scalar(name='y', dtype='int32')

		self.m = T.vector(name='mask-y', dtype='int32')
		# initial hidden state of the RNN
		self.h0 = T.vector()
		# learning rate
		self.lr = T.scalar()

		self.l0 = T.scalar(name='length-y0', dtype='int32')
		self.l1 = T.scalar(name='length-y1', dtype='int32')
		self.l2 = T.scalar(name='length-y2', dtype='int32')

		if self.activation == 'tanh':
			activation = T.tanh
		elif self.activation == 'sigmoid':
			activation = lambda x: 1./(1. + T.exp(-3.75 * x))
		elif self.activation == 'relu':
			activation = lambda x: x * (x > 0)
		elif self.activation == 'cappedrelu':
			activation = lambda x: T.minimum(x * (x > 0), 6)
		else:
			raise NotImplementedError

		self.rnn = LSTM(input0 = self.x0,input1 = self.x1,input2 = self.x2, n_in=self.n_in,
						n_hidden=self.n_hidden, n_out = self.n_out, ylen0 = self.l0, ylen1 = self.l1, ylen2 = self.l2,activation = activation, output_type = self.output_type, lstm_flag = self.lstm_flag)

	def onehot(self,x,numclasses=None):
		""" Convert integer encoding for class-labels (starting with 0 !)
		to one-hot encoding.

        If numclasses (the number of classes) is not provided, it is assumed 
        to be equal to the largest class index occuring in the labels-array + 1.
        The output is an array who's shape is the shape of the input array plus
        an extra dimension, containing the 'one-hot'-encoded labels. 
		"""
		if x.shape==():
			x = x[None]
		if numclasses is None:
			numclasses = x.max() + 1
		result = np.zeros(list(x.shape) + [numclasses], dtype='int64')
		z = np.zeros(x.shape)
		for c in range(int(numclasses)):
			z *= 0
			z[np.where(x==c)] = 1
			result[...,c] += z
		return result

	def shared_dataset(self, data_xy):
		""" Load the dataset into shared variables """

		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x,
											dtype=theano.config.floatX),borrow = True)

		shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow = True)

		if self.output_type in ('binary', 'softmax'):
			return shared_x, T.cast(shared_y, 'int32')
		else:
			return shared_x, shared_y

	def compute_gradients(self,loss):
		"""Compute gradients of the parameters, given a loss variable"""
		grads = T.grad(loss, self.rnn.params)
		return zip(self.rnn.params, grads)

	def compute_update_rules(self,gradients, lr, momentum):
		"""Compute param updates with momentum"""
		updates = []
		for i in range(len(gradients)):
			w = gradients[i][0]	 # Weight
			g = gradients[i][1]	 # Weight gradient
			g0 = self.rnn.lstm_prev_updates[i] # Previous weight update

			updates.append((g0, momentum * g0 - lr * g))
			updates.append((w, w + g0))
		return updates

	def get_updates(self,cost,l_r,mom):
		if self.lstm_flag :
			updates = self.compute_update_rules(self.compute_gradients(cost), l_r, mom)
		else:
			gparams = []
			for param in self.rnn.params:
				gparam = T.grad(cost, param)
				gparams.append(gparam)

			updates = {}
			for param, gparam in zip(self.rnn.params, gparams):
				weight_update = self.rnn.updates[param]
				upd = mom * weight_update - l_r * gparam
				updates[weight_update] = upd
				updates[param] = param + upd
		return updates

	def debug_grad_norm(self,loss):
		grads = T.grad(loss, self.rnn.params)
		g_total = 0
		for g in grads:
			g_total += abs(g.sum())
		return g_total

	def fit(self, dataset, test_dataset , tag_set, out_file, validation_frequency=100):
		""" Fit model
		"""

		TRAIN,VALIDATION,TEST = dataset
		(X_train,Y_train,M_train_x,M_train_y,L_train) = TRAIN
		(X_val,Y_val,M_val_x,M_val_y,L_val) = VALIDATION
		(X_test,Y_test,M_test_x,M_test_y,L_test) = TEST

		self.interactive = True

		train_s0_x, train_s0_y = self.shared_dataset((X_train, Y_train))
		train_s0_l = theano.shared(np.asarray(L_train, dtype= 'int32'),borrow = True)

		train_s1_x, train_s1_y = self.shared_dataset((X_val, Y_val))
		train_s1_l = theano.shared(np.asarray(L_val, dtype= 'int32'),borrow = True)

		train_s2_x, train_s2_y = self.shared_dataset((X_test, Y_test))
		train_s2_l = theano.shared(np.asarray(L_test, dtype= 'int32'),borrow = True)

		n_train = train_s0_x.get_value(borrow=True).shape[0]

		TRAIN,VALIDATION,TEST = test_dataset
		(X_train,Y_train,M_train_x,M_train_y,L_train) = TRAIN
		(X_val,Y_val,M_val_x,M_val_y,L_val) = VALIDATION
		(X_test,Y_test,M_test_x,M_test_y,L_test) = TEST

		val_s0_x, val_s0_y = self.shared_dataset((X_train, Y_train))
		val_s0_l = theano.shared(np.asarray(L_train, dtype= 'int32'),borrow = True)

		val_s1_x, val_s1_y = self.shared_dataset((X_val, Y_val))
		val_s1_l = theano.shared(np.asarray(L_val, dtype= 'int32'),borrow = True)

		val_s2_x, val_s2_y = self.shared_dataset((X_test, Y_test))
		val_s2_l = theano.shared(np.asarray(L_test, dtype= 'int32'),borrow = True)

		n_val = val_s0_x.get_value(borrow=True).shape[0]

		######################
		# BUILD ACTUAL MODEL #
		######################
		logger.info('... building the model')

		index = T.lscalar('index')	  # index to a case

		cost = self.rnn.loss(self.y, self.l0, self.l1, self.l2) + self.L1_reg * self.rnn.L1 + self.L2_reg * self.rnn.L2_sqr
		gradient = self.debug_grad_norm(cost)

		compute_train_error = theano.function(inputs=[index, ],
											  outputs=self.rnn.errors_l(self.y, self.l0, self.l1, self.l2),
											  givens={
												  self.x0: train_s0_x[index],
												  self.x1: train_s1_x[index],
												  self.x2: train_s2_x[index],
												  self.y : train_s0_y[index][0],
												  self.l0 : train_s0_l[index],
												  self.l1 : train_s1_l[index],
												  self.l2 : train_s2_l[index]},
											  mode=mode)

		compute_val_error = theano.function(inputs=[index, ],
											  outputs=self.rnn.errors_l(self.y, self.l0, self.l1, self.l2),
											  givens={
												  self.x0: val_s0_x[index],
												  self.x1: val_s1_x[index],
												  self.x2: val_s2_x[index],
												  self.y : val_s0_y[index][0],
												  self.l0 : val_s0_l[index],
												  self.l1 : val_s1_l[index],
												  self.l2 : val_s2_l[index]},
											  mode=mode)


		l_norm = theano.function(inputs=[],outputs=self.rnn.debug_lnorm(),givens={},mode=mode)


		get_final_layer = theano.function(inputs=[index,],
									   outputs=self.rnn.debug_final_layer(self.y, self.l0, self.l1, self.l2),
									   givens={
										   self.x0: train_s0_x[index],
										   self.x1: train_s1_x[index],
										   self.x2: train_s2_x[index],
										   self.y : train_s0_y[index][0],
										   self.l0 : train_s0_l[index],
										   self.l1 : train_s1_l[index],
										   self.l2 : train_s2_l[index]},
									   mode=mode)
		###! here
		predict_test = theano.function(inputs=[index,],
									   outputs=self.rnn.predict_instance(self.y, self.l0, self.l1, self.l2),
									   givens={
										   self.x0: val_s0_x[index],
										   self.x1: val_s1_x[index],
										   self.x2: val_s2_x[index],
										   self.y : val_s0_y[index][0],
										   self.l0 : val_s0_l[index],
										   self.l1 : val_s1_l[index],
										   self.l2 : val_s2_l[index]},
									   mode=mode)

		self.updater = update_list.Adadelta(lr = self.learning_rate)
		updates = self.updater.get_updates(self.rnn.params, cost)

		###! here
		train_model = theano.function(inputs=[index,],
									  outputs=[cost,gradient],
									  updates=updates,
									  givens={
										  self.x0: train_s0_x[index],
										  self.x1: train_s1_x[index],
										  self.x2: train_s2_x[index],
										  self.y : train_s0_y[index][0],
										  self.l0 : train_s0_l[index],
										  self.l1 : train_s1_l[index],
										  self.l2 : train_s2_l[index]},
									  mode=mode)

		###############
		# TRAIN MODEL #
		###############
		logger.info('... training')
		epoch = 0

		while (epoch < self.n_epochs):
			epoch = epoch + 1

			epoch_gradient = np.array([])
			batch_loss = []
			id_dict = defaultdict(int)
			for idx in xrange(n_train):
				example_cost,example_gradient = train_model(idx)
				example_final_layer = get_final_layer(idx)
				_,y,_= example_final_layer

				y = int(y.astype(int))
				id_dict [y] += 1

				batch_loss.append(example_cost)
				epoch_gradient = np.append(epoch_gradient,[example_gradient])
				example_final_layer = get_final_layer(0)
				iter = (epoch - 1) * n_train + idx + 1

				if iter % validation_frequency == 0:

					train_losses = [compute_train_error(i) for i in xrange(n_train)]
					this_train_loss = np.mean(train_losses)

#					val_losses = [compute_val_error(i) for i in xrange(n_val)]
					this_val_loss = 0 #np.mean(val_losses)

					logger.info('epoch %i/%i, seq %i/%i, tr error %.3f val error %.3f loss : %.3f lnorm %.3f gradnorm %.3f ' % \
						(epoch, self.n_epochs, idx + 1, n_train, this_train_loss, this_val_loss, np.mean(batch_loss), l_norm(), epoch_gradient.mean()))

		print >> sys.stderr, "predicting test file"
		out = open(out_file,'w')
		for i in xrange(n_val):
			pred = predict_test(i)
			print >> out, tag_set[pred[0]]

def runData(data_file):
	try:
		(dim,num_tag,tag_set,dataset) = pickle.load(gzip.open(p.benchmark))
	except:
		print >> sys.stderr, "data file",p.benchmark ,"cannot be read"
		quit(1)

	try:
		(_,_,_,test_dataset) = pickle.load(gzip.open(p.test_benchmark))
	except:
		print >> sys.stderr, "test file",p.test_benchmark,"cannot be read"
		quit(1)

	model = MetaRNN(n_in= dim, n_hidden= p.n_hidden, n_out= num_tag,
					learning_rate= p.learning_rate, learning_rate_decay= p.learning_rate_decay,
					n_epochs= p.n_epochs, activation= p.activation,
					L1_reg = p.L1_reg, L2_reg = p.L2_reg,
					lstm_flag = p.lstm_flag)
	out_file = p.out_file
	if out_file == "":
		lstm = 'out'
		if p.lstm_flag:
			lstm = ''
		out_file = 'test.'+'_'.join(['b-'+p.benchmark.split('.')[0],'e-'+str(p.n_epochs),'l-'+str(p.learning_rate),'h-'+str(p.n_hidden),'a-'+p.activation,'with'+lstm+'lstm'])

	model.fit(dataset, test_dataset, tag_set, out_file, validation_frequency= p.validation_frequency)

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	np.random.seed(seed = 0)
	parser = get_parser()
	p = parser.parse_args()
	runData(p)
