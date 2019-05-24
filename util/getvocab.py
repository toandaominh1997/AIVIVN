import os
from glob import glob 
# import transformer.Constants as Constants

def getvocab(dir_name):
    vocab = list()
    if(os.path.isdir(dir_name)):
        for filename in glob('{}/*.*'.format(dir_name)):
            vocab += list(open(filename, encoding='utf-8'))
    else:
        vocab += list(open(dir_name, encoding='utf-8'))
    
    alphabets = ''.join(sorted(set(''.join(vocab))))
    return alphabets.replace('\n', '').replace('\t', '')
