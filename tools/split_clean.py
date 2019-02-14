## We have to combine two conllu files with token range information because bitext can contain two sentences concatenated. We have to revert the parallel structure again using token ranges. Please let me know if there's a better solution. Jungo Kasai.
import argparse

parser = argparse.ArgumentParser(description='conllu to horizontal')
parser.add_argument('--lang1')
parser.add_argument('--lang2')
args = parser.parse_args()
"""
word: 1
UPOS: 3
XPOS: 4
"""

batch_size = 200000
batch_idx = 0
sent_idx = 0
sents_1 = []
sents_2 = []
with open(args.lang1) as fin_1:
    with open(args.lang2) as fin_2:
        for line_1, line_2 in zip(fin_1, fin_2):
            line_1 = line_1.strip()
            line_2 = line_2.strip()
            if line_1 == '' or line_2 == '':
                continue
            sents_1.append(line_1)
            sents_2.append(line_2)
nb_batches = len(sents_1)//batch_size + 1
for batch_idx in range(nb_batches):
    with open(args.lang1 + str(batch_idx+1), 'wt') as fout:
        for line in sents_1[batch_idx*batch_size:(batch_idx+1)*batch_size]:
            fout.write(line)
            fout.write('\n')
for batch_idx in range(nb_batches):
    with open(args.lang2 + str(batch_idx+1), 'wt') as fout:
        for line in sents_2[batch_idx*batch_size:(batch_idx+1)*batch_size]:
            fout.write(line)
            fout.write('\n')

