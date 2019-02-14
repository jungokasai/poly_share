import sys
from collections import defaultdict
import argparse
import glob
import IPython as ipy


parser = argparse.ArgumentParser(description='conllu to horizontal')
parser.add_argument('--infile')
parser.add_argument('--tokenized', action='store_true')
args = parser.parse_args()

def build_vocab(filepatterns, tokenized=True):
    char_counts = defaultdict(int)
    word_counts = defaultdict(int)
    filenames = []
    if tokenized:
        cutoff = 5
    else:
        cutoff = 0
    for filepattern in filepatterns:
        filenames += glob.glob(filepattern)
    for filename in filenames:
        path_components = filename.split('/')
        if tokenized:
            if 'german' in path_components:
                lang = 'ger'
            elif 'english' in path_components:
                lang = 'eng'
            else:
                print('other language?')
        lines = open(filename).readlines()
        for line in lines:
            for w in line.split():
                if tokenized:
                    w = lang + ':' + w
                word_counts[w] += 1
                for ch in w:
                    char_counts[ch] += 1
    charcounts = list(char_counts.items())
    charcounts.sort(key=lambda x: x[1], reverse=True)
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: (x[1], x[0]), reverse=True)
    special_chars =  ["<",">","/","S"]
    special_words = ["eng:@start@", "ger:@start@", "@end@"]
    #["<S>", "</S>"]

#    with open("char_vocab.txt",'w') as f:
#        for i, chcount in enumerate(charcounts):
#            ch = chcount[0]
#            count = chcount[1]
#            if count >= 5:
#                f.write("{}\t{}\n".format(ch, i))
#        for ch in special_chars:
#            if ch not in char_counts.keys():
#                i += 1
#                f.write("{}\t{}\n".format(ch, i))

    with open("tokens_freq.txt",'w') as f:
        f.write("@@UNKNOWN@@\n")
        for w in special_words:
            if w not in word_counts.keys():
                f.write("{}\n".format(w))
        for i, wcount in enumerate(wcounts):
            w = wcount[0]
            count = wcount[1]
            f.write("{}\t{}\n".format(w, count))

    with open("tokens.txt",'w') as f:
        f.write("@@UNKNOWN@@\n")
        for w in special_words:
            if w not in word_counts.keys():
                f.write("{}\n".format(w))
        for i, wcount in enumerate(wcounts):
            w = wcount[0]
            count = wcount[1]
            if count >= cutoff:
                f.write("{}\n".format(w))

if __name__=="__main__":
    build_vocab([args.infile], args.tokenized)
        
