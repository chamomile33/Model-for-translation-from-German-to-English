from torch.utils.data import DataLoader
from model import create_mask
from tqdm import tqdm
from torch import nn
from model import translate
import sacrebleu
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader

def train_epoch(model,
          iterator,
          optimizer,
          criterion,pad_idx, unk_idx, device, mask_unk = False):
    model.train()
    losses = 0

    for (src,tgt,_,_) in tqdm(iterator):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input,pad_idx, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]

        if mask_unk:
          tgt_out = tgt_out.reshape(-1)
          mask = ((tgt_out != unk_idx)&(tgt_out != pad_idx)).float()
          loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_out, reduction = 'none')
          loss = loss * mask
          loss = loss.sum()/mask.sum()
        else:
          loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(iterator)


def evaluate(model,
             iterator,
             criterion,pad_idx,unk_idx,bos_idx, eos_idx, en_vocab, device, mask_unk = False,without_unk = False,
             left_unk_for_gen = False, beam_search = False, k = 0):
    model.eval()
    losses = 0

    en_texts_all = []
    pred_texts = []
    for (src, tgt,de_text,en_text) in tqdm(iterator):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input,pad_idx, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]

        if mask_unk:
          tgt_out = tgt_out.reshape(-1)
          mask = ((tgt_out != unk_idx)&(tgt_out != pad_idx)).float()
          loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_out, reduction = 'none')
          loss = loss * mask
          loss = loss.sum()/mask.sum()
        else:
          loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        losses += loss.item()

        pred_text = translate(model,src,bos_idx,unk_idx,eos_idx,device,en_vocab,without_unk, left_unk_for_gen, beam_search, k)
        pred_texts.append(pred_text)
        en_texts_all += en_text
    bleu = BLEU()
    bleu_score = bleu.corpus_score(pred_texts,[en_texts_all]).score

    return losses / len(iterator), bleu_score, pred_texts[:3], en_texts_all[:3]

               
def train(model,optimizer,criterion,n_epochs, train_iter, valid_iter,pad_idx, unk_idx,bos_idx, eos_idx, en_vocab, device,mask_unk = False, scheduler = None,step = 0):
      train_losses,valid_losses,valid_bleu = [],[],[]

      for epoch in range(n_epochs):
          train_loss = train_epoch(model, train_iter, optimizer, criterion, pad_idx, unk_idx, device, mask_unk)
          valid_loss, val_bleu, pred_text, real_text = evaluate(model, valid_iter, criterion, pad_idx, unk_idx,bos_idx,eos_idx,en_vocab,device, mask_unk)

          train_losses += [train_loss]
          valid_losses += [valid_loss]
          valid_bleu += [val_bleu]
          
          if scheduler is not None:
            scheduler.step(valid_bleu[-1])
            
