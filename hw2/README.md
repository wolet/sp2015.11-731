### Neural Models for MT Evaluation

Here's the solution:

- A vanilla RNN and a RNN with Gated Recurrent Unit (GRU) is trained to get 3 sentence representations
- These representations are fed to a softmax layer to predict {-1,0,1}

Other details:

- sentences are tokenized with ptb tokenizer script written in sed
- you need a theano installation to run the code
- In small dataset(2K sentences) it achieves ~ %90 accuracy(= able to overfit).
- Since I do not use GPU and mini-batch training it is *very* slow.
- the code is mutated from RNN/LSTM tagger very quickly, that's why it's impossible to read.
- you have to prepare a benchmark pkl.gz to run the code. I provide a [link](https://www.dropbox.com/s/epqfjru9kwbbcps/mt_eval-half.pkl.gz) to the half of the training set I use. To run prediction on whole training set you need to use *prepare_benchmark.sh*. Then run:

     python rnn_mt_eval_benchmark.py --benchmark mt_eval-half.pkl.gz --test-benchmark mt_eval-full-glove50.pkl.gz --lrate 0.5 --epoch 25 --with-lstm --val-freq 26208

- The following command line parameters *do not* work: [dim,num-seq,max-length,vector-file,train-file,lm-mode,lr-decay]

### Below is Original Readme

There are three Python programs here (`-h` for usage):

 - `./evaluate` evaluates pairs of MT output hypotheses relative to a reference translation using counts of matched words
 - `./check` checks that the output file is correctly formatted
 - `./grade` computes the accuracy

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./evaluate | ./check | ./grade


The `data/` directory contains the following two files:

 - `data/train-test.hyp1-hyp2-ref` is a file containing tuples of two translation hypotheses and a human (gold standard) translation. The first 26208 tuples are training data. The remaining 24131 tuples are test data.

 - `data/train.gold` contains gold standard human judgements indicating whether the first hypothesis (hyp1) or the second hypothesis (hyp2) is better or equally good/bad for training data.

Until the deadline the scores shown on the leaderboard will be accuracy on the training set. After the deadline, scores on the blind test set will be revealed and used for final grading of the assignment.
