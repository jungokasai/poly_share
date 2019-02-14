## We have to combine two conllu files with token range information because bitext can contain two sentences concatenated. We have to revert the parallel structure again using token ranges. Please let me know if there's a better solution. Jungo Kasai.
import argparse

parser = argparse.ArgumentParser(description='conllu to horizontal')
parser.add_argument('--conllu')
parser.add_argument('--outfile')
parser.add_argument('--idx', default=1, type=int)
args = parser.parse_args()
"""
word: 1
UPOS: 3
XPOS: 4
"""
def main():
    sents = []
    sent = []
    idx = args.idx
    with open(args.conllu) as fin:
        for line in fin:
            tokens = line.split()
            if len(tokens) > 0:
                if tokens[0] != '#':
                    if '-' in tokens[0]:
                        continue
                    sent.append(tokens[idx])
            else:
                sents.append(sent)
                sent = []
    output_sents(sents, args.outfile)

def output_sents(sents, outfile):
    with open(outfile, 'w') as fout:
        for sent in sents:
            fout.write(' '.join(sent))
            fout.write('\n')
    
                            
def get_sent_ranges(infile):
    lengths = []
    with open(infile) as fin:
        for line in fin:
            line = line.strip()
            lengths.append(len(line))
    return lengths
            
if __name__ == '__main__':
    main()

