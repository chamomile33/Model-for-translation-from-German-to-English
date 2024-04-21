import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import io   
from torch.nn.utils.rnn import pad_sequence

#https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html

def sample_lines(input_de,input_en,output_de,output_en,n, seed):
  np.random.seed(seed)
  with open(input_en, 'r', encoding='utf-8') as file:
        lines_en = file.readlines()
  with open(input_de, 'r', encoding='utf-8') as file:
        lines_de = file.readlines()
  if n == -1:
        n = len(lines_en)
  sampled_lines = np.random.choice(range(0,len(lines_en)),size = n, replace = False)
  with open(output_en, 'w', encoding='utf-8') as file:
        file.writelines(np.array(lines_en)[sampled_lines])
  with open(output_de, 'w', encoding='utf-8') as file:
        file.writelines(np.array(lines_de)[sampled_lines])

def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      counter.update(tokenizer(string_))
  return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'],min_freq = 15)

def prepare_vocabs(filepaths):
    de_tokenizer = get_tokenizer(None, language='de')
    en_tokenizer = get_tokenizer(None, language='en')
    
    de_vocab = build_vocab(filepaths[0], de_tokenizer)
    en_vocab = build_vocab(filepaths[1], en_tokenizer)

    de_vocab.set_default_index(de_vocab['<unk>'])
    en_vocab.set_default_index(en_vocab['<unk>'])
    return de_vocab,en_vocab,de_tokenizer,en_tokenizer

def data_process(filepaths, de_vocab, en_vocab, de_tokenizer, en_tokenizer):
  raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)],
                            dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                            dtype=torch.long)
        data.append((de_tensor_, en_tensor_,raw_de,raw_en))
  return data


def generate_batch(data_batch, bos_idx, eos_idx,pad_idx):
    de_batch, en_batch = [], []
    de_texts, en_texts = [],[]
    for (de_item, en_item,de_text, en_text) in data_batch:
        de_batch.append(torch.cat([torch.tensor([bos_idx]), de_item, torch.tensor([eos_idx])], dim=0))
        en_batch.append(torch.cat([torch.tensor([bos_idx]), en_item, torch.tensor([eos_idx])], dim=0))
        de_texts.append(de_text)
        en_texts.append(en_text)
    de_batch = pad_sequence(de_batch, padding_value=pad_idx)
    en_batch = pad_sequence(en_batch, padding_value=pad_idx)
    return de_batch, en_batch, de_texts, en_texts

