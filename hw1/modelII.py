#! /usr/bin/python

__author__="Volkan Cirik <volkan.cirik@gmail.com>"
__date__ ="Feb, 2015 (copied from my crappy code written in May,2013)"

import sys
from collections import defaultdict as dd

tokens = {}
t = {}
esptokens = set()
q = {}
EPOCH=20

def initialize(in_file_name):
	in_lines = [line.strip() for line in open(in_file_name)]

	ger_lines = []
	en_lines = []
	for line in in_lines:
		l1 = line.split('|||')[1].strip().split()
		l2 = line.split('|||')[0].strip().split()

		ger_lines.append(l2)
		en_lines.append(l1)
		for teng in l1:
			if teng not in tokens:
				tokens.setdefault(teng,{})
				t.setdefault(teng,{})
			for tesp in l2:
				esptokens.add(tesp)
				if tesp not in tokens[teng]:
					tokens[teng].setdefault(tesp,0)
					t[teng].setdefault(tesp,0)
				tokens[teng][tesp] += 1

	t.setdefault("NULL",{})
	for tesp in esptokens:
		t["NULL"].setdefault(tesp,1.00/len(esptokens))
	for eng in tokens:
		for esp in tokens[eng]:
			t[eng][esp] = 1.00/len(tokens[eng])

	return ger_lines,en_lines
def debug(delta,c):

	print "DELTA"
	for k in delta:
		for i in delta[k]:
			for j in delta[k][i]:
				print k,i,j,delta[k][i][j]
	print "C PARAMETERS"
	for teng in c:
		for tesp in c[teng]:
			print teng,tesp,c[teng][tesp]
	print "T PARAMETERS"
	for teng in t:
		for tesp in t[teng]:
			print teng,tesp,t[teng][tesp]

def EM1(ger_lines, en_lines):
	for e in xrange(EPOCH):
		delta = {}
		c = {}
		c["NULL"] = {}
		k = 0
		for l1,l2 in zip(en_lines,ger_lines):
			k += 1

			if k%5000 == 0:
				print >> sys.stderr, k,"sentence is done."
			delta[k] = {}
			for I,tesp in enumerate(l2):

				i = I+1
				sumj = 0
				delta[k].setdefault(i,{})
				for J,teng in enumerate(l1):

					j = J+1
					delta[k][i].setdefault(j,0)
					sumj += t[teng][tesp]
				delta[k][i].setdefault(0,0) 
				sumj += t["NULL"][tesp]

				for J,teng in enumerate(l1):
					j = J+1

					delta[k][i][j] = t[teng][tesp]/sumj
					if teng not in c:
						c.setdefault(teng,{})
					if tesp not in c[teng]:
						c[teng].setdefault(tesp,0)
						c[teng].setdefault("NULL",0)
					c[teng][tesp] += delta[k][i][j]
					c[teng]["NULL"] += delta[k][i][j]

				delta[k][i][0] = t["NULL"][tesp]/sumj
				if tesp not in c["NULL"]:
					c["NULL"].setdefault(tesp,0)
					c["NULL"].setdefault("NULL",0)
				c["NULL"][tesp] += delta[k][i][0]
				c["NULL"]["NULL"] += delta[k][i][0]
		for teng in c:
			for tesp in c[teng]:
				t[teng][tesp] = c[teng][tesp]/c[teng]["NULL"]
		print >> sys.stderr, e, "EPOCH ended"

def printResult(ger_lines,en_lines):

	for l1,l2 in zip(en_lines,ger_lines):
		for i,tesp in enumerate(l2):
			maxv = t["NULL"][tesp]
			maxindex = 0
			for j,teng in enumerate(l1):
				if t[teng][tesp] > maxv:
					maxv = t[teng][tesp]
					maxindex = j
			print "-".join([str(i),str(maxindex)]),
		print

def debug2(dic,st):
	print st,"PARAMETERS"
	for M in dic:
		for L in dic[M]:
			for I in dic[M][L]:
				for J in dic[M][L][I]:
					print J,I,L,M,dic[M][L][I][J]
def EM2(ger_lines,en_lines):
	for e in xrange(EPOCH):
		delta = {}
		c = {}
		c2 = {}
		c["NULL"] = {}
		k = 0
		for l1,l2 in zip(en_lines,ger_lines):
			k += 1

			if k%5000 == 0:
				print >> sys.stderr, k,"sentence is done."

			l = len(l1)
			m = len(l2)

			if m not in c2:
				c2.setdefault(m,{})
			if l not in c2[m]:
				c2[m].setdefault(l,{})

			delta[k] = {}
			for I,tesp in enumerate(l2):

				i = I+1
				if i not in c2[m][l]:
					c2[m][l].setdefault(i,{})

				sumj = 0
				delta[k].setdefault(i,{})
				for J,teng in enumerate(l1):

					j = J+1

					delta[k][i].setdefault(j,0)
					sumj += t[teng][tesp]*q[m][l][i][j]

				delta[k][i].setdefault(0,0)
				sumj += t["NULL"][tesp]*q[m][l][i][0]


				for J,teng in enumerate(l1):
					j = J+1

					delta[k][i][j] = q[m][l][i][j]*t[teng][tesp]/sumj

					if teng not in c:
						c.setdefault(teng,{})
					if tesp not in c[teng]:
						c[teng].setdefault(tesp,0)
						c[teng].setdefault("NULL",0)
					c[teng][tesp] += delta[k][i][j]
					c[teng]["NULL"] += delta[k][i][j]

					if j not in c2[m][l][i]:
						c2[m][l][i].setdefault(j,0)
						c2[m][l][i].setdefault(-1,0)
					c2[m][l][i][j] += delta[k][i][j]
					c2[m][l][i][-1] += delta[k][i][j]


				delta[k][i][0] = t["NULL"][tesp]*q[m][l][i][0]/sumj
				if tesp not in c["NULL"]:
					c["NULL"].setdefault(tesp,0)
					c["NULL"].setdefault("NULL",0)
				c["NULL"][tesp] += delta[k][i][0]
				c["NULL"]["NULL"] += delta[k][i][0]

				if 0 not in c2[m][l][i]:
					c2[m][l][i].setdefault(0,0)
				c2[m][l][i][0] += delta[k][i][0]

				if -1 not in c2[m][l][i]:
					c2[m][l][i].setdefault(-1,0)
				c2[m][l][i][-1] += delta[k][i][0]

		for teng in c:
			for tesp in c[teng]:
				t[teng][tesp] = c[teng][tesp]/c[teng]["NULL"]
		for M in q:
			for L in q[M]:
				for I in q[M][L]:
					for J in q[M][L][I]:
						q[M][L][I][J] = c2[M][L][I][J]/c2[M][L][I][-1]
		print >> sys.stderr, e, "EPOCH ended"
def initializeAlignment(ger_lines,en_lines):
	for l1,l2 in zip(en_lines,ger_lines):

		l = len(l1)
		m = len(l2)
		if m not in q:
			q.setdefault(m,{})
		if l not in q[m]:
			q[m].setdefault(l,{})
		for I in xrange(m):
			i = I+1
			if i not in q[m][l]:
				q[m][l].setdefault(i,{})
			for j in xrange(-1,l+1):
				if j not in q[m][l][i]:
					q[m][l][i].setdefault(j,1.00/(l+1))

def printResult2(ger_lines,en_lines):

	for l1,l2 in zip(en_lines,ger_lines):

		l = len(l1)
		m = len(l2)

		for I,tesp in enumerate(l2):
			i = I + 1
			maxv = t["NULL"][tesp]*q[m][l][i][0]
			maxindex = 0
			for J,teng in enumerate(l1):
				j = J + 1
				if q[m][l][i][j]*t[teng][tesp] > maxv:
					maxv = q[m][l][i][j]*t[teng][tesp]
					maxindex = J
			print "-".join([str(I),str(maxindex)]),
		print
ger_lines, en_lines = initialize(sys.argv[1])
EM1(ger_lines,en_lines)
initializeAlignment(ger_lines,en_lines)
EM2(ger_lines,en_lines)
printResult2(ger_lines,en_lines)
