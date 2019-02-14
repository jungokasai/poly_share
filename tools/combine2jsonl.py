from hanziconv import HanziConv
import argparse, jsonlines

parser = argparse.ArgumentParser(description='Create json for auxiliary machine translation')
parser.add_argument('--lang1')
parser.add_argument('--lang1upos')
parser.add_argument('--lang1xpos')
parser.add_argument('--lang1piece')
parser.add_argument('--lang2')
parser.add_argument('--lang2upos')
parser.add_argument('--lang2xpos')
parser.add_argument('--lang2piece')
parser.add_argument('--outfile')
parser.add_argument('--src')
parser.add_argument('--tgt')
args = parser.parse_args()
def main():
    count = 0
    with jsonlines.open(args.outfile, mode='w') as writer: 
        with open(args.lang1) as fin1:
            with open(args.lang2) as fin2:
                with open(args.lang1upos) as fin1upos:
                    with open(args.lang2upos) as fin2upos:
                        with open(args.lang1xpos) as fin1xpos:
                            with open(args.lang2xpos) as fin2xpos:
                                with open(args.lang1piece) as fin1piece:
                                    with open(args.lang2piece) as fin2piece:
                                        for line1, line2, line1upos, line2upos, line1xpos, line2xpos, line1piece, line2piece in zip(fin1, fin2, fin1upos, fin2upos, fin1xpos, fin2xpos, fin1piece, fin2piece):
                                            line1 = line1.strip()
                                            line2 = line2.strip()
                                            #if len(line1.split()) > 50 or len(line2.split()) > 50:
                                            #    continue
                                            line1upos = line1upos.strip()
                                            line2upos = line2upos.strip()
                                            line1xpos = line1xpos.strip()
                                            line2xpos = line2xpos.strip()
                                            line1piece = line1piece.strip()
                                            line2piece = line2piece.strip()
                                            line = {}
                                            line['lang1'] = {'text': line1, 'lang': args.src, 'upos': line1upos, 'xpos': line1xpos, 'piece': line1piece}
                                            line['lang2'] = {'text': line2, 'lang': args.tgt, 'upos': line2upos, 'xpos': line2xpos, 'piece': line2piece}
                                            writer.write(line)
                                            assert len(line1.split()) == len(line1upos.split())
                                            assert len(line1.split()) == len(line1xpos.split())
                                            assert len(line2.split()) == len(line2upos.split())
                                            assert len(line2.split()) == len(line2xpos.split())
                                            count += 1
                    #if count == 5000:
                    #    break
if __name__ == '__main__':
    main()

