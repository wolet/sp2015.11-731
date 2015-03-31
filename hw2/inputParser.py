import argparse
import theano.tensor as T

def get_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--benchmark', action='store', dest='benchmark',help='benchmark pkl.gz file',default = 'mt_eval-2000-glove50.pkl.gz')

	parser.add_argument('--val-benchmark', action='store', dest='val_benchmark',help=' val benchmark pkl.gz file',default = '')

	parser.add_argument('--test-benchmark', action='store', dest='test_benchmark',help=' test benchmark pkl.gz file',default = '')

	parser.add_argument('--outfile', action='store', dest='out_file',help='output file name, default test.$benchmark_$epoch_$lrate_$hidden_$activation_$with{|out}lstm',default = '')

	parser.add_argument('--lrate', action='store', dest='learning_rate',help='Learning Rate, default = 0.5',type=float,default = 0.5)

	parser.add_argument('--lr-decay', action='store', dest='learning_rate_decay',help='Learning Rate decay, default = 0.998',type=float,default = 0.998)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 100',type=int,default = 100)

	parser.add_argument('--val-freq', action='store', dest='validation_frequency',help='validation frequency, default = 1000',type=int,default = 1000)

	parser.add_argument('--hidden', action='store', dest='n_hidden',help='hidden size of softmax layer, default = 100',type=int,default = 100)

	parser.add_argument('--activation', action='store', dest='activation',help='activation function {tanh,relu,sigmoid,cappedrelu}, default = sigmoid',default = 'sigmoid')

	parser.add_argument('--with-lstm', action='store_true', dest='lstm_flag',help='train with lstm units')
	parser.add_argument('--l1-reg', action='store', dest='L1_reg',help='L1 reg coefficient, default = 0.0001',type=float,default = 0.0001)

	parser.add_argument('--l2-reg', action='store', dest='L2_reg',help='L2 reg coefficient, default = 0.0001',type=float,default = 0.0001)

	parser.set_defaults(lstm_flag = False)

	return parser
