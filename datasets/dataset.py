from torch.utils.data import Dataset
import transformer.Constants as Constants
from glob import glob
import os
import numpy as np 
import torch
import torch.utils.data
from transformer import Constants
import unicodedata
import string
import re

class TranslationDataset(Dataset):
    def __init__(self, dir_name, max_word_seq_len=100, min_word_count=5, keep_case=False, src_word2idx=None, tgt_word2idx=None):
        src_insts, tgt_insts, src_word2idx, tgt_word2idx = self.preprocess(dir_name=dir_name, max_word_seq_len=max_word_seq_len, min_word_count=min_word_count, keep_case=keep_case, src_word2idx=src_word2idx, tgt_word2idx=tgt_word2idx)
        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )



    def normalizeString(self, s):
        s = re.sub(r"([.,!?])", r"", s)
        s = s.replace('\n', '').replace('\t', '')
        return s
    def read_instances(self, dir_name, max_sent_len, keep_case):
        src_word_insts = []
        tgt_word_insts = []
        trimmed_sent_count = 0
        if(os.path.isdir(dir_name)):
            for filename in glob('{}/*.*'.format(dir_name)):
                inst_file = list(open(filename, encoding='utf-8'))
                for sent in inst_file:
                    if(not keep_case):
                        sent = sent.lower()
                    sent = sent.replace('\n', '').replace('\t', '')
                    src_words = list(self.normalizeString(self.unicodeToAscii(sent)))
                    tgt_words = list(self.normalizeString(sent))
                    if(len(src_words) > max_sent_len):
                        trimmed_sent_count +=1
                    src_word_inst = src_words[:max_sent_len]
                    tgt_word_inst = tgt_words[:max_sent_len]
                    if(src_word_inst):
                        src_word_insts += [[Constants.BOS_WORD] + src_word_inst + [Constants.EOS_WORD]]
                        tgt_word_insts += [[Constants.BOS_WORD] + tgt_word_inst + [Constants.EOS_WORD]]
        print('[Info] Get {} instances'.format(len(src_word_insts)))

        if trimmed_sent_count > 0:
            print('[Warning] {} instances are trimmed to the max sentence length {}.'.format(trimmed_sent_count, max_sent_len))

        return src_word_insts, tgt_word_insts
    def build_vocab_idx(self, word_insts, min_word_count):
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
    def convert_instance_to_idx_seq(self, word_insts, word2idx):
        ''' Mapping words to idx sequence. '''
        return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]
    def preprocess(self, dir_name, max_word_seq_len, min_word_count, keep_case, src_word2idx=None, tgt_word2idx=None):
        max_token_seq_len = max_word_seq_len + 2

        # Training set
        src_word_insts, tgt_word_insts = self.read_instances(dir_name=dir_name, max_sent_len=max_token_seq_len, keep_case=False)
        if len(src_word_insts) != len(tgt_word_insts):
            print('[Warning] The training instance count is not equal.')
            min_inst_count = min(len(src_word_insts), len(tgt_word_insts))
            src_word_insts = src_word_insts[:min_inst_count]
            tgt_word_insts = tgt_word_insts[:min_inst_count]
        #- Remove empty instances
        src_word_insts, tgt_word_insts = list(zip(*[
            (s, t) for s, t in zip(src_word_insts, tgt_word_insts) if s and t]))

        if(src_word2idx is None or tgt_word2idx is None):
            print('[Info] Build vocabulary for source.')
            src_word2idx = self.build_vocab_idx(src_word_insts, min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = self.build_vocab_idx(tgt_word_insts, min_word_count)

        # word to index
        print('[Info] Convert source word instances into sequences of word index.')
        src_insts = self.convert_instance_to_idx_seq(src_word_insts, src_word2idx)
        tgt_insts = self.convert_instance_to_idx_seq(tgt_word_insts, tgt_word2idx)
        
        return src_insts, tgt_insts, src_word2idx, tgt_word2idx

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word
    def generate_mistakes(self, name, word2idx):
        used = []
        num_mists = np.random.choice([1, 2, 3], p=[0.4, 0.3, 0.3])
        if(len(name)<4):
            return name
        if(len(name)<6):
            num_mists=1
        num = 0
        while(num<num_mists):
            mist_type = np.random.choice(['insert', 'change', 'swap', 'delete_letter'], p=[0.4, 0.3, 0.1, 0.2])
            while(True):
                idx = np.random.randint(low=1, high=len(name)-1)
                if(idx not in used):
                    break
            num+=1
            used.append(idx)
            if(mist_type=='insert'):
                new_letter = np.random.randint(low=4, high=len(word2idx))
                if(np.random.rand()<0.5):
                    name.insert(idx, new_letter)
                else:
                    name.insert(idx+1, new_letter)
                 
            if(mist_type=='change'):
                new_letter = np.random.randint(low=4, high=len(word2idx))
                name[idx] = new_letter
            if(mist_type=='swap'):
                if(idx==0):
                    name[idx], name[idx+1] = name[idx+1], name[idx]
                if(idx==len(name)-2):
                    name[idx], name[idx-1] = name[idx-1], name[idx]
                else:
                    if(np.random.rand() < 0.5):
                        name[idx], name[idx+1] = name[idx+1], name[idx]
                    else:
                        name[idx], name[idx-1] = name[idx-1], name[idx]
            if(mist_type=='delete_letter'):
                name.pop(idx) 
        return name
    def __len__(self):
        return self.n_insts
    
    def __getitem__(self, index):
        if(self._tgt_insts):
            return self._src_insts[index], self._tgt_insts[index]
        return self._src_insts[index]


def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos
