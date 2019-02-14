import sys

filename = sys.argv[1]
lang = sys.argv[2]
if len(sys.argv) > 3:
    outf = open(sys.argv[3],'w')
else:
    outf = sys.stdout

for line in open(filename,'r'):
    if len(line.split()) < 3:
        continue
    print(lang+':'+line.strip(),file=outf)
