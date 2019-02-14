import sys
def combine_vecs(filenames, langs):
    with open('combined_vecs.txt', 'w') as outfile:
        for fname, lang in zip(filenames, langs):
            with open(fname) as infile:
                for i, line in enumerate(infile):
                    if i != 0:
                        ## skip headers
                        outfile.write(lang + ':' + line)

if __name__ == '__main__':
    filenames = sys.argv[1:3]
    langs = sys.argv[3:5]
    assert len(filenames) == len(langs)
    combine_vecs(filenames, langs)
