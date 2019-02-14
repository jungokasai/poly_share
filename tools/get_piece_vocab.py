with open('data/test7/vocabulary/pieces.txt') as fin:
    with open('data/test7/vocabulary/pieces_new.txt', 'wt') as fout:
        fout.write('deu:@start@')
        fout.write('eng:@start@')
        fout.write('@end@')
        fout.write('\n')
        for line in fin:
            piece = line.strip().split()[0]
            fout.write(piece)
            fout.write('\n')
