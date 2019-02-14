import sys
from collections import defaultdict
import argparse
import IPython as ipy
import glob
import sentencepiece as spm


parser = argparse.ArgumentParser(description='Use sentence piece to get BPE')
parser.add_argument('--infile') #,'--list', nargs='+', help='input files', required=True)
parser.add_argument('--outfile') #,'--list', nargs='+', help='input files', required=True)
parser.add_argument('--model') #,'--list', nargs='+', help='input files', required=True)
args = parser.parse_args()

def run(infile, outfile, model):
    sp = spm.SentencePieceProcessor()
    sp.Load(model)
    with open(infile) as fin:
        with open(outfile, 'wt') as fout:
            for line in fin:
                line = line.strip()
                line = sp.EncodeAsPieces(line)
                fout.write(' '.join(line))
                fout.write('\n')
    
if __name__=="__main__":
    run(args.infile, args.outfile, args.model) 
