#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,re,os,argparse

def main():
	parser=argparse.ArgumentParser(description='''
NT-AcPredictor: predicting N-terminal acetylated sequences by decision tree

This program predicts whether N-terminal acetylation occurs for proteins 
whose initiator methionine residues are removed by methionine amino peptidase.
The program was produced and would be maintained by Kazunori D Yamada 
(kyamada@ecei.tohoku.ac.jp) and Masaru Miyagi (mxm356@case.edu).
'''
	,formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument('-i',dest="input",type=str,required=True,help='Input sequence(s) fasta.')
	parser.add_argument('-f',dest="first",type=str,choices=["truncated","exist"],required=True,help='First residue condition.')
	args=parser.parse_args()
	
	input=args.input.rstrip()
	liname,liseq=readfasta(input)
	likmername,likmerseq,licutcheck=seqcheck(liname,liseq,args.first)
	lianswer=predict(likmerseq)
	output(liname,likmerseq,lianswer,licutcheck)
	
def output(liname,likmerseq,lianswer,licutcheck):
	print("In the Acetylation column, Ac and unAc represents Acetylated and Unacetylated respectively.\n\n",sep="",end="")
	if "not" in licutcheck:
		print("Since the sequence of which the amino acid in the first site (without iMet) is not A, C, G, P, S, T, V\n",sep="",end="")
		print("may not be processed by NatA or B, the prediction result for the sequence may not be accurate.\n\n",sep="",end="")
	dipred={0:"unAc",1:"Ac"}
	print("*************** Prediction results ***************")
	print("ID","\t","Sequence","\t","Acetylation","\t","Supplement","\n",sep="",end="")
	for i in range(len(liname)):
		print(liname[i],"\t",likmerseq[i],"\t",dipred[lianswer[likmerseq[i]]],"\t",sep="",end="")
		if licutcheck[i]=="not":
			print("The prediction may not be accurate. The sequence may be processed by not NatA or D but NatB, C E or F.\n",sep="",end="")
		else:
			print("\n",sep="",end="")

encode={"A":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		"R":[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		"N":[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		"D":[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		"C":[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		"Q":[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		"E":[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
		"G":[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
		"H":[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
		"I":[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
		"L":[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
		"K":[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
		"M":[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
		"F":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
		"P":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
		"S":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
		"T":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
		"W":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
		"Y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
		"V":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
}

def predict(likmerseq):
	lianswer={}
	for kmerseq in likmerseq:
		s=[]
		for aa in kmerseq:
			s+=encode[aa]
		if s[15]<0.5:
			if s[0]<0.5:
				if s[23]<0.5:
					lianswer[kmerseq]=0
				else:
					if s[19]<0.5:
						lianswer[kmerseq]=1
					else:
						lianswer[kmerseq]=0
			else:
				if s[34]<0.5:
					if s[21]<0.5:
						lianswer[kmerseq]=1
					else:
						lianswer[kmerseq]=0
				else:
					lianswer[kmerseq]=0
		else:
			if s[154]<0.5:
				if s[61]<0.5:
					if s[94]<0.5:
						lianswer[kmerseq]=1
					else:
						lianswer[kmerseq]=0
				else:
					lianswer[kmerseq]=0
			else:
				lianswer[kmerseq]=0
	return lianswer

def seqcheck(liname,liseq,first):
	badlen,badchar=0,0
	truncateaa=['A','C','G','P','S','T','V']
	likmername,likmerseq,licutcheck,libadlen,libadchar=[],[],[],[],[]
	if len(liname)!=len(liseq):
		print("Input file is not of standard FASTA format.")
		sys.exit()
	else:
		for i in range(len(liseq)):
			truncatedseq=liseq[i]
			# print("1",truncatedseq)
			# print("---------")
			if first=="exist": truncatedseq=re.sub("^.","",truncatedseq)
			# print("2",truncatedseq)
			newseq=""
			for j in range(len(truncatedseq)):
				if j==kmer: break
				newseq+=truncatedseq[j]
			if len(newseq)<kmer or re.search("[^ARNDCQEGHILKMFPSTWYV]",newseq):
				if len(newseq)<kmer:
					badlen=1
					libadlen.append(liname[i])
				if re.search("[^ARNDCQEGHILKMFPSTWYV]",newseq):
					badchar=1
					libadchar.append(liname[i])
			else:
				likmerseq.append(newseq)
				likmername.append(liname[i])
				cutcheck="not" # If "not", the sequence may be processed by NatB, C E or F.
				if newseq[0] in truncateaa: cutcheck="ok"
				licutcheck.append(cutcheck)
	
	if badlen==1 or badchar==1:
		if badlen==1:
			print("The following sequence(s) without iMet is shorter than 10 mer.")
			print("\n".join(libadlen))
		if badchar==1:
			print("The following sequence(s) contains non-standard character(s).")
			print("\n".join(libadchar))
		sys.exit()
	else:
		return [likmername,likmerseq,licutcheck]

def readfasta(file):
	liname,liseq=[],[]
	fin=open(file,"r")
	tmpline=""
	for line in fin:
		line=line.rstrip()
		if line.startswith(">"):
			line=line.replace(">","")
			liname.append(line)
			if tmpline:
				if len(tmpline): liseq.append(tmpline)
				tmpline=""
		else:
			line=line.replace(".","-")
			tmpline+=line
	if len(tmpline): liseq.append(tmpline)
	fin.close()
	return [liname,liseq]

if __name__ == '__main__':
	kmer=10
	main()
