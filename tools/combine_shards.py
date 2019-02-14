from hanziconv import HanziConv
import argparse, jsonlines

parser = argparse.ArgumentParser(description='Create json for auxiliary machine translation')
args = parser.parse_args()
bases = ['/homes/gws/jkasai/data/europarl/de-en/tokenized/data/english/europarl-v7.de-en.en{}.tokenized']
bases.append('/homes/gws/jkasai/data/europarl/de-en/tokenized/data/german/europarl-v7.de-en.de{}.tokenized')
def main():
    with open('combined.tokenized', 'wt') as fout:
        for basis in bases:
            for i in range(1, 8):
                with open(basis.format(i)) as fin:
                    for line in fin:
                        fout.write(line)
            
if __name__ == '__main__':
    main()

