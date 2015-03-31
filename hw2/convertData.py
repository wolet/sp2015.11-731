#!/usr/bin/env python
"""
Convert a sequence labeling data for RNN

@author Volkan Cirik
"""
import sys
import theano
import numpy as np
import pickle
import cPickle
import gzip

UNK="*unknown*"
class TwoWayDict(dict):
	def __setitem__(self, key, value):
		# Remove any previous connections with these values
		if key in self:
			del self[key]
		if value in self:
			del self[value]
		dict.__setitem__(self, key, value)
		dict.__setitem__(self, value, key)

	def __delitem__(self, key):
		dict.__delitem__(self, self[key])
		dict.__delitem__(self, key)

	def __len__(self):
		"""Returns the number of connections"""
		return dict.__len__(self) // 2
def convertData(tr_,val_,te_,vector_file,max_length,num_seq,dim,weight):

	try:
		tr_file = open(tr_)
	except:
		print >> sys.stderr, "seqence file",tr_,"cannot be read"
		quit(1)
	try:
		val_file = open(val_)
	except:
		print >> sys.stderr, "seqence file",val_,"cannot be read"
		quit(1)
	try:
		te_file = open(te_)
	except:
		print >> sys.stderr, "seqence file",te_,"cannot be read"
		quit(1)
	try:
		v_map = cPickle.load(gzip.open(vector_file, "rb"))
		print >> sys.stderr, "loaded word vectors .."
	except:
		print >> sys.stderr, "word embedding file",vector_file,"cannot be read."
		quit(1)

	print >> sys.stderr,"______________"
	print >> sys.stderr,"parameters are:"
	print >> sys.stderr,tr_,val_,te_,vector_file,max_length,num_seq,dim
	print >> sys.stderr,"______________"

	X = np.zeros((num_seq,max_length, dim), dtype=theano.config.floatX)
	Y = np.zeros((num_seq,max_length), dtype=theano.config.floatX)
	Mx = np.zeros((num_seq,max_length, dim), dtype=theano.config.floatX)
	My = np.zeros((num_seq,max_length), dtype=theano.config.floatX)
	L = np.zeros((num_seq),dtype=theano.config.floatX)

	tag_set = TwoWayDict()
	tag_id = 0
	s = []
	s_id = 0

	indexes = []
	deleted = 0
	for infile in [tr_file,val_file,te_file]:
		ntok = 0.0
		unk = 0.0
		for line in infile:
			l = line.split()
			if len(l) != 2:
				if len(s) > max_length or len(s) < 1:
					s = []
					deleted += 1
					continue
				for j,(v,t,w) in enumerate(s):
					X[s_id,j,:] = v
					Mx[s_id,j,:] = 1

					Y[s_id,j] = t
					My[s_id,j] = w

				L[s_id] = len(s)
				s = []
				s_id += 1
				continue

			token = l[0]
			tag = l[1]

			ntok +=1
			try:
				v = v_map[token]
			except:
				unk += 1
				v = v_map[UNK]
				pass

			if tag not in tag_set:
				tag_set[tag] = tag_id
				tag_id += 1
			if tag_set[tag] == 1:
				w = weight
#				print >> sys.stderr, "+",
			else:
				w = 1
			t = tag_set[tag]
			s.append((v,t,w))


		if len(s) <= max_length and len(s) >= 1:
			for j,(v,t,w) in enumerate(s):
				X[s_id,j,:] = v
				Mx[s_id,j,:] = 1

				Y[s_id,j] = t
				My[s_id,j] = w
			L[s_id] = len(s)
			s = []
			s_id += 1
		indexes.append(s_id)
		print >> sys.stderr
		print >> sys.stderr, "completed a file..."
		print >> sys.stderr, "UNKNOWN rate...",unk/ntok

	print >> sys.stderr, "Dim:",dim,"Tag:",tag_id,"TR:",indexes[0],"VAL:",indexes[1],"TE:",indexes[2]-deleted

	print >> sys.stderr,"A Sample data"
	print >> sys.stderr, L[0]
	print >> sys.stderr, X[0,:,:]
	print >> sys.stderr, My[0,:]

	print >> sys.stderr,"______________"

	TR = indexes[0]
	VAL = indexes[1]
	TE = indexes[2] - deleted

	tr_X = X[0:TR,:,:]
	tr_Y = Y[0:TR,:]
	tr_Mx = Mx[0:TR,:,:]
	tr_My = My[0:TR,:]
	tr_L = L[0:TR]

	val_X = X[TR:VAL,:,:]
	val_Y = Y[TR:VAL,:]
	val_Mx = Mx[TR:VAL,:,:]
	val_My = My[TR:VAL,:]
	val_L = L[TR:VAL]

	te_X = X[VAL:TE,:,:]
	te_Y = Y[VAL:TE,:]
	te_Mx = Mx[VAL:TE,:,:]
	te_My = My[VAL:TE,:]
	te_L = L[VAL:TE]

	TRAIN = (tr_X,tr_Y,tr_Mx,tr_My,tr_L)
	VALIDATION = (val_X,val_Y,val_Mx,val_My,val_L)
	TEST = (te_X,te_Y,te_Mx,te_My,te_L)

	dataset = (TRAIN,VALIDATION,TEST)
	out_data = (dim,tag_id,tag_set,dataset)

	print "DEBUG tagset:",tag_set
	return out_data

def dumpDataset(out_data,outfile, zipped = False):

	if zipped:
		out = gzip.open(out_file,'wb')
	else:
		out = open(out_file,'wb')

	pickle.dump(out_data,out)
	out.close()
usage="""
python convertData.py tac-kbp.coarse.train tac-kbp.coarse.dev tac-kbp.coarse.test w2vec-300-lowercase.pkl.gz 142 2217 300 tac-kbp.coarse.full.w2vec.pkl.gz 1 
"""
if __name__ == "__main__":
	if len(sys.argv) < 9:
		print usage
		quit(1)
	tr_file = sys.argv[1]
	val_file = sys.argv[2]
	te_file = sys.argv[3]

	vector_file = sys.argv[4]
	max_length = int(sys.argv[5])
	num_seq = int(sys.argv[6])

	dim = int(sys.argv[7])
	out_file = sys.argv[8]
	weight = float(sys.argv[9])
	dataset = convertData(tr_file,val_file,te_file,vector_file,max_length,num_seq,dim,weight)
	dumpDataset(dataset,out_file)

#python ../src/convertData.py tac-coarse-compressed.train tac-coarse-compressed.dev tac-coarse-compressed.test ../src/glove6B-50-lowercase.pkl.gz 107 2217 50 tac-coarse-compressed.glove6B-50.pkl.gz 1 
# python convertData.py data/sent_1.conll data/sent_2.conll data/sent_3.conll glove6B.pkl.gz 141 78624 50 mt_eval-full-glove50new.pkl 1
