import sys
from collections import defaultdict
import argparse
import glob
import IPython as ipy


parser = argparse.ArgumentParser(description='conllu to horizontal')
parser.add_argument('--vocab')
parser.add_argument('--embeddings')
parser.add_argument('--outfile')
args = parser.parse_args()

def filter_vocab(vocab_file, embeddings, outfile):
    vocab = []
    with open(vocab_file) as fin:
        for line in fin:
            vocab.append(line.strip())
    filtered_lines = []
    with open(embeddings) as fin:
        for line in fin:
            tokens = line.strip().split(' ')
            if len(tokens) > 0:
                token = tokens[0]
                if token in vocab:
                    filtered_lines.append(line)
    output(filtered_lines, outfile)

def output(filtered_lines, outfile):
    with open(outfile, 'wt') as fout:
        for line in filtered_lines:
            fout.write(line)
    
if __name__=="__main__":
    filter_vocab(args.vocab, args.embeddings, args.outfile)
        
