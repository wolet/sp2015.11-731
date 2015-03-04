#!/usr/bin/env python
"""
Convert a conll like data 

@author Volkan Cirik
"""
import sys

labels =[line.strip() for line in open(sys.argv[1])]
sent = [ line.strip().split() for line in open(sys.argv[2])]
for i,l in enumerate(labels):
	for tok in sent[i]:
		print tok.lower(),l
	print

