import sys

def main(old, new):
    left = 0
    flag = False
    with open(old) as fin:
        with open(new, 'wt') as fout:
            for i, line in enumerate(fin):    
                tokens = line.strip().split()
                if len(tokens) == 2:
                    left = 0
                    for char in tokens[1]:
                        if char == '(':
                            left += 1
                    if left == 2:
                        flag = True
                    if flag and tokens[1] == '*)':
                        tokens[1] = '*))'
                        flag = False
                    fout.write(tokens[0].ljust(15))
                    fout.write(tokens[1].rjust(15))
                    fout.write('\n')
                else:
                    fout.write('\n')
                        
if __name__ == '__main__':
    old = sys.argv[1]
    new = sys.argv[2]
    main(old, new)
    
