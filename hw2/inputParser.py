import argparse
import theano.tensor as T

def get_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--benchmark', action='store', dest='benchmark',help='benchmark pkl.gz file',default = 'mt_eval-2000-glove50.pkl.gz')

	parser.add_argument('--test-benchmark', action='store', dest='test_benchmark',help=' test benchmark pkl.gz file',default = '')

	parser.add_argument('--outfile', action='store', dest='out_file',help='output file name, default test.$benchmark_$epoch_$lrate_$hidden_$activation_$with{|out}lstm',default = '')

	parser.add_argument('--lrate', action='store', dest='learning_rate',help='Learning Rate, default = 0.01',type=float,default = 0.01)

	parser.add_argument('--lr-decay', action='store', dest='learning_rate_decay',help='Learning Rate decay, default = 0.998',type=float,default = 0.998)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 100',type=int,default = 100)

	parser.add_argument('--val-freq', action='store', dest='validation_frequency',help='validation frequency, default = 1000',type=int,default = 1000)

	parser.add_argument('--hidden', action='store', dest='n_hidden',help='hidden size of softmax layer, default = 100',type=int,default = 100)

	parser.add_argument('--activation', action='store', dest='activation',help='activation function {tanh,relu,sigmoid,cappedrelu}, default = sigmoid',default = 'sigmoid')

	parser.add_argument('--with-lstm', action='store_true', dest='lstm_flag',help='train with lstm units,')

	parser.add_argument('--l1-reg', action='store', dest='L1_reg',help='L1 reg coefficient, default = 0.0001',type=float,default = 0.0001)

	parser.add_argument('--l2-reg', action='store', dest='L2_reg',help='L2 reg coefficient, default = 0.0001',type=float,default = 0.0001)

	parser.add_argument('--lm-mode', action='store_true', dest='lm_mode',help='use LM mode, default : false')

	parser.add_argument('--train-file', action='store', dest='tr_file',help='tr file',default = '')
	parser.add_argument('--val-file', action='store', dest='val_file',help='val file',default = '')
	parser.add_argument('--test-file', action='store', dest='te_file',help='te file',default = '')
	parser.add_argument('--vector-file', action='store', dest='vector_file',help='vector file',default = '')
	parser.add_argument('--max-length', action='store', dest='max_length',help='max length',type=int,default = 0)
	parser.add_argument('--num-seq', action='store', dest='num_seq',help='num seq',type=int,default = 0)
	parser.add_argument('--dim', action='store', dest='dim',help='dim',type=int,default = 0)

	parser.set_defaults(lm_mode = False)
	parser.set_defaults(lstm_flag = False)

	return parser
