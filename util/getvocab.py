import transformer.Constants as Constants
def getvocab(filename):
    vocab = list(open(filename, encoding='latin-1'))
    alphabets = ''.join(sorted(set(''.join(vocab))))
    return alphabets