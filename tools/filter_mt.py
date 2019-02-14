import sys
from collections import defaultdict
import argparse
import glob
import IPython as ipy
import json, jsonlines


parser = argparse.ArgumentParser(description='conllu to horizontal')
parser.add_argument('--infile')
parser.add_argument('--outfile')
parser.add_argument('--cutoff', type=int)
args = parser.parse_args()

def filter_mt(infile, outfile, cutoff):
    filtered = []
    with open(infile) as fin:
        for line in fin:
            data = json.loads(line)
            if len(data['lang1']['text'].split()) <= cutoff and len(data['lang2']['text'].split()) <= cutoff:
                filtered.append(data)
    with jsonlines.open(outfile, mode='w') as fout:
        for data in filtered:
            fout.write(data)

def output(filtered_lines, outfile):
    with open(outfile, 'wt') as fout:
        for line in filtered_lines:
            fout.write(line)
    
if __name__=="__main__":
    filter_mt(args.infile, args.outfile, args.cutoff)
        
