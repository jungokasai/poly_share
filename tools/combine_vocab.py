import sys
from collections import defaultdict
import argparse
import glob
import IPython as ipy


parser = argparse.ArgumentParser(description='conllu to horizontal')
parser.add_argument('--ud')
parser.add_argument('--mt')
parser.add_argument('--cutoff', type=int, default=100000)
args = parser.parse_args()

def combine_vocab(ud_file, mt_file, cutoff):
    words = []
    #lines = open(ud_file).readlines()
    #for line in lines:
    #    #for w in line.split():
    #    w = line.split()[0]
    #    words.append(w)
    lines = open(mt_file).readlines()
    for line in lines:
        w = line.split()[0]
        if w not in words:
            words.append(w)
        print(len(words))
        if len(words) == cutoff:
            break
    special_words = ["eng:@start@", "deu:@start@", "@end@"]


    with open("tokens.txt",'w') as f:
        f.write("@@UNKNOWN@@\n")
        for w in special_words:
            f.write("{}\n".format(w))
        for w in words:
            f.write("{}\n".format(w))

if __name__=="__main__":
    combine_vocab(args.ud, args.mt, args.cutoff)
        
