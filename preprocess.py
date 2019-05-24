''' Handling the data io '''
import argparse
import torch
import os 
import transformer.Constants as Constants
from util import generate_mistake
from glob import glob

from sklearn.model_selection import train_test_split

def read_instances(dir_name, max_sent_len=100, keep_case=False, alphabet=None, training=False):
    if(os.path.isdir(dir_name)):
        for filename in glob('{}/*.*'.format(dir_name)):
            inst_file = list(open(filename, encoding='utf-8'))
            for sent in inst_file:
                if(not keep_case):
                    sent = sent.lower()
                sent = sent.replace('\n', '').replace('\t', '')
                if(training):
                    words = list(sent)
                    if(len(words) > max_sent_len):
                        trimmed_sent_count +=1
                    word_inst = words[:max_sent_len]
                    if(word_inst):
                        train_word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                        target_word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                        if(alphabet is not None):
                            aug_words = generate_mistake.generate_mistakes(words, alphabet)
                            if(len(aug_words) > max_sent_len):
                                trimmed_sent_count +=1
                            aug_word_inst = aug_words[:max_sent_len] 
                            train_word_insts += [[Constants.BOS_WORD] + aug_word_inst + [Constants.EOS_WORD]]
                            target_word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                else:
                    train_words, target_words = sent.split('|')
                    if(len(train_words)>max_sent_len):
                        trimmed_sent_count +=1
                    if(len(target_words)>max_sent_len):
                        trimmed_sent_count +=1
                    train_words_inst, target_words_inst = list(train_words), list(target_words)
                    train_word_insts += [[Constants.BOS_WORD] + train_words_inst + [Constants.EOS_WORD]]
                    target_word_insts += [[Constants.BOS_WORD] + target_words_inst + [Constants.EOS_WORD]]
    print('[Info] Get {} instances from {}'.format(len(train_word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return train_word_insts, target_word_insts


def read_instances_from_file(inst_file, max_sent_len, keep_case, vocab=None, training=False):
    ''' Convert file into word seq lists and vocab '''

    # word_insts = []
    train_word_insts = []
    target_word_insts = []
    trimmed_sent_count = 0
    with open(inst_file, encoding='latin-1') as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            # words = sent.split()
            sent = sent.replace('\n', '').replace('\\', '')
            sent = sent.replace('\t', '')
            
            if(training):
                words = list(sent)
                if len(words) > max_sent_len:
                    trimmed_sent_count += 1
                word_inst = words[:max_sent_len]
                if word_inst:
                    # word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                    train_word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                    target_word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                    if(vocab!=None):
                        aug_word = generate_mistake.generate_mistakes(words, vocab)
                        aug_word_inst = aug_word[:max_sent_len] 
                        train_word_insts += [[Constants.BOS_WORD] + aug_word_inst + [Constants.EOS_WORD]]
                        target_word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                train_words, target_words = sent.split('|')
                if(len(train_words)>max_sent_len):
                    trimmed_sent_count +=1
                if(len(target_words)>max_sent_len):
                    trimmed_sent_count +=1
                train_words_inst, target_words_inst = list(train_words), list(target_words)
                train_word_insts += [[Constants.BOS_WORD] + train_words_inst + [Constants.EOS_WORD]]
                target_word_insts += [[Constants.BOS_WORD] + target_words_inst + [Constants.EOS_WORD]]



    print('[Info] Get {} instances from {}'.format(len(train_word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return train_word_insts, target_word_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def getvocab(dir_name):
    vocab = list()
    if(os.path.isdir(dir_name)):
        for filename in glob('{}/*.*'.format(dir_name)):
            vocab += list(open(filename, encoding='utf-8'))
    else:
        vocab += list(open(dir_name, encoding='utf-8'))
    
    alphabets = ''.join(sorted(set(''.join(vocab))))
    return alphabets.replace('\n', '').replace('\t', '')


def preprocess(train_src, valid_src, max_word_seq_len, min_word_count, keep_case):
    max_token_seq_len = max_word_seq_len + 2

    # Training set
    alphabets = getvocab(train_src)
    train_src_word_insts, train_tgt_word_insts = read_instances(dir_name=train_src, max_sent_len=max_token_seq_len, keep_case=False, alphabet=alphabets, training=True)
    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]
    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts, valid_tgt_word_insts = read_instances(dir_name=valid_src, max_sent_len=max_token_seq_len, keep_case=False, alphabet=alphabets, training=False)
    
    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))
    
    print('[Info] Build vocabulary for source.')
    src_word2idx = build_vocab_idx(train_src_word_insts, min_word_count)
    print('[Info] Build vocabulary for target.')
    tgt_word2idx = build_vocab_idx(train_tgt_word_insts, min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}
    
    return data


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_src', required=True)
    parser.add_argument('--valid_src', required=True)
    parser.add_argument('--save_data', required=True)
    parser.add_argument('--max_word_seq_len', type=int, default=100)
    parser.add_argument('--min_word_count', type=int, default=5)
    parser.add_argument('--keep_case', action='store_true')

    parser.add_argument('--share_vocab', action='store_true')
    parser.add_argument('--vocab', default=None)

    opt = parser.parse_args()
    print(opt)
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>


    alphabets = getvocab.getvocab(opt.train_src)

    # Training set
    train_src_word_insts = []
    train_tgt_word_insts = []
    for filename in glob.glob('{}/*.*'.format(opt.train_src)):
        src_word_insts, tgt_word_insts = read_instances_from_file(filename, opt.max_word_seq_len, opt.keep_case, alphabets, training=True)
        train_src_word_insts +=src_word_insts
        train_tgt_word_insts +=tgt_word_insts
    for filename in glob.glob('{}/*.*'.format(opt.target_src)):
        src_word_insts, tgt_word_insts = read_instances_from_file(filename, opt.max_word_seq_len, opt.keep_case, alphabets, training=False)
        valid_src_word_insts +=src_word_insts
        valid_tgt_word_insts +=tgt_word_insts
    
    # train_tgt_word_insts = read_instances_from_file(
    #     opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # # Validation set
    # valid_src_word_insts = read_instances_from_file(
    #     opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    # valid_tgt_word_insts = read_instances_from_file(
    #     opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    # if len(valid_src_word_insts) != len(valid_tgt_word_insts):
    #     print('[Warning] The validation instance count is not equal.')
    #     min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
    #     valid_src_word_insts = valid_src_word_insts[:min_inst_count]
    #     valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    # #- Remove empty instances
    # valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
    #     (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))
    

    print('[Info] Build vocabulary for source.')
    src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
    print('[Info] Build vocabulary for target.')
    tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
