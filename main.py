import torch
from dataset import sample_lines,prepare_vocabs,data_process, generate_batch
from train import train
from model import Seq2SeqTransformer, translate, remove_unk
import random
import numpy as np
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from model import make_translation_dict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

def main(): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_filepaths = ['train_short.de','train_short.en']   
    valid_filepaths = ['valid_short_de','valid_short_en']
    
    sample_lines('train.de-en.de','train.de-en.en',train_filepaths[0],train_filepaths[1], -1, 122333)
    sample_lines('val.de-en.de','val.de-en.en',valid_filepaths[0],valid_filepaths[1], 500, 122333)
    
    de_vocab,en_vocab,de_tokenizer, en_tokenizer = prepare_vocabs(train_filepaths)
    
    train_data = data_process(train_filepaths, de_vocab, en_vocab,de_tokenizer,en_tokenizer)
    val_data = data_process(valid_filepaths, de_vocab, en_vocab,de_tokenizer,en_tokenizer)
    
    BATCH_SIZE = 256 #учила так но может быть cuda out of memory тогда надо уменьшить 
    PAD_IDX = de_vocab['<pad>']
    BOS_IDX = de_vocab['<bos>']
    EOS_IDX = de_vocab['<eos>']
    UNK_IDX = de_vocab['<unk>']
    
    rand_gen = torch.Generator()
    rand_gen.manual_seed(122333)
    
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=lambda x: generate_batch(x,BOS_IDX,EOS_IDX,PAD_IDX),generator = rand_gen)
    valid_iter = DataLoader(val_data, batch_size=1, collate_fn=lambda x: generate_batch(x,BOS_IDX,EOS_IDX,PAD_IDX))
    
    fix_seed(122333)
    SRC_VOCAB_SIZE = len(de_vocab)
    TGT_VOCAB_SIZE = len(en_vocab)
    EMB_SIZE = 384
    NHEAD = 6
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    NUM_EPOCH = 43
    MASK_UNK = False
    LABEL_SMOOTHING = 0.1

    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in model.parameters():
         if p.dim() > 1:
             nn.init.xavier_uniform_(p)

    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing = LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'max',patience = 4,factor = 0.3, min_lr = 1e-7)
    train(model,optimizer,loss_fn,NUM_EPOCH,train_iter,valid_iter,PAD_IDX,UNK_IDX,
          BOS_IDX,EOS_IDX,en_vocab,device, MASK_UNK, scheduler = scheduler)
    
    translations_vocab = make_translation_dict(model,train_iter,PAD_IDX,device)
    
    test_filepaths = ['test1.de-en.de','test1.de-en.de']
    test_data = data_process(test_filepaths,de_vocab, en_vocab,de_tokenizer,en_tokenizer)
    test_iter = DataLoader(test_data, batch_size=1,shuffle=False, collate_fn=lambda x: generate_batch(x,BOS_IDX,EOS_IDX,PAD_IDX))
    
    beam_search = True
    k = 5

    model.eval()
    with torch.no_grad():
        with open('test1.de-en.en', 'w') as f:
            for (src, trg,de_text,en_text) in tqdm(test_iter):
                src, trg = src.to(device), trg.to(device)
                
                pred_text,att = translate(model,src, BOS_IDX, UNK_IDX, EOS_IDX, device, en_vocab,
                                          beam_search = beam_search, k = k, with_weights = True)
                
                seq1 = pred_text.split(' ') + ['<eos>']
                pred_seq_remove_unk = remove_unk(model,src,seq1.copy(),BOS_IDX,UNK_IDX,en_vocab,device)
                seq2 = ['<bos>'] + de_text[0].split(' ') + ['<eos>']

                for row_index, row in enumerate(att.detach().cpu()):
                    if seq1[row_index] == '<unk>':
                        top_for_row,top_index = torch.sort(row,descending = True)[:2]
                        top_for_col1= att[:,top_index[0]].max()
                        top_for_col2 = att[:,top_index[1]].max()
                        score_col1 = (top_for_row[0] - (top_for_col1 - top_for_row[0])).item()
                        score_col2 = (top_for_row[1] - (top_for_col2 - top_for_row[1])).item()
                        if score_col1 >= score_col2:
                            top_index = top_index[0].item()
                        else:
                            top_index = top_index[1].item()
                        if seq2[top_index] in translations_vocab.keys():
                            seq1[row_index] = translations_vocab[seq2[top_index]]
                        else:
                            seq1[row_index] = pred_seq_remove_unk[row_index]

                pred_text = ' '.join(seq1[:-1])
                f.write(pred_text + '\n')
                
if __name__ == '__main__':     
    main()
            
