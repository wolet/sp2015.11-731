#!/usr/bin/sh

k=$1
embeddings=$2
dim=$3
name=$4
head -$k sent_1.tok > sent_1.tok.$k
head -$k sent_2.tok > sent_2.tok.$k
head -$k sent_3.tok > sent_3.tok.$k

head -$k sent.labels > sent.labels.$k
python prepare_conll.py  sent.labels.$k sent_1.tok.$k > sent_1.conll.$k
python prepare_conll.py  sent.labels.$k sent_2.tok.$k > sent_2.conll.$k
python prepare_conll.py  sent.labels.$k sent_3.tok.$k > sent_3.conll.$k

mlen=`cat sent_*.tok.$k | awk 'BEGIN{mi=100;ma=0}{if(NF<mi)mi = NF;if(NF>ma)ma = NF;}END{print ma}'`
echo -e "$mlen"
k3=$(($k*3))
echo -e "3k = $k3"
echo -e "python prepare_data.py sent_1.conll.$k sent_2.conll.$k sent_3.conll.$k /ai/home/vcirik/temp/rnn-lstm/EMBEDDINGS/glove6B-50-lowercase.pkl.gz $mlen $k3 50 mt_eval.$mlen.glove6B-50.pkl.gz 1"
python prepare_data.py sent_1.conll.$k sent_2.conll.$k sent_3.conll.$k $embeddings $mlen $k3 $dim $name.pkl.gz 1
