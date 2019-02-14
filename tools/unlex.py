import sys, os
sys.path.append(os.getcwd())
from allennlp.data import conllu_parse
input_file = sys.argv[1]
with open(input_file) as fin:
    output_data = fin.read()
data = conllu_parse(output_data)
outfile = open('new.conll', 'wt')
for sent in data:
    for token in sent:
        token['form'] = token['upostag']
    outfile.write(sent.serialize())
outfile.close()
