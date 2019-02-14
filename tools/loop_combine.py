import subprocess

def main(dir_name, lang1, lang2):
    lang1_full = get_full(lang1)
    lang2_full = get_full(lang2)
    for i in range(1, 11):
        command = 'python tools/combine2jsonl.py'
        command += ' --lang1 ~/data/europarl/{}/tokenized/data/{}/europarl-v7.{}.{}.tokenized'.format(dir_name, lang1_full, dir_name, dir_name.split('-')[1]+str(i))
        command += ' --lang1upos ~/data/europarl/{}/upos/data/{}/europarl-v7.{}.{}.upos'.format(dir_name, lang1_full, dir_name, dir_name.split('-')[1]+str(i))
        command += ' --lang1xpos ~/data/europarl/{}/xpos/data/{}/europarl-v7.{}.{}.xpos'.format(dir_name, lang1_full, dir_name, dir_name.split('-')[1]+str(i))
        command += ' --lang1piece ~/data/europarl/{}/piece/data/{}/europarl-v7.{}.{}.piece'.format(dir_name, lang1_full, dir_name, dir_name.split('-')[1]+str(i))
        command += ' --lang2 ~/data/europarl/{}/tokenized/data/{}/europarl-v7.{}.{}.tokenized'.format(dir_name, lang2_full, dir_name, dir_name.split('-')[0]+str(i))
        command += ' --lang2upos ~/data/europarl/{}/upos/data/{}/europarl-v7.{}.{}.upos'.format(dir_name, lang2_full, dir_name, dir_name.split('-')[0]+str(i))
        command += ' --lang2xpos ~/data/europarl/{}/xpos/data/{}/europarl-v7.{}.{}.xpos'.format(dir_name, lang2_full, dir_name, dir_name.split('-')[0]+str(i))
        command += ' --lang2piece ~/data/europarl/{}/piece/data/{}/europarl-v7.{}.{}.piece'.format(dir_name, lang2_full, dir_name, dir_name.split('-')[0]+str(i))
        command += ' --outfile ~/data/europarl/{}/europarl-v7.{}-{}.jsonl'.format(dir_name, dir_name, str(i))
        command += ' --src {} --tgt {}'.format(lang1, lang2)
        print(command)
        subprocess.check_call(command, shell=True)

def get_full(lang):
    if lang == 'eng':
        return 'english'
    elif lang == 'deu':
        return 'german'
    elif lang == 'cmnt':
        return 'chinese'

if __name__ == '__main__':
    dir_name = 'de-en'
    lang1 = 'eng'
    lang2 = 'deu'
    main(dir_name, lang1, lang2)
