#https://pytorch.org/tutorials/beginner/translation_transformer.html

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import torch.nn.modules.transformer as t

def _mha_block_with_weights(layer, x, mem,
                   attn_mask, key_padding_mask, is_causal) -> Tensor:
        x,att = layer.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=True,average_attn_weights = True)
        return layer.dropout2(x),att


def forward_with_return_att(layer,
        tgt,
        memory,
        tgt_mask = None,
        memory_mask = None,
        tgt_key_padding_mask = None,
        memory_key_padding_mask = None,
        tgt_is_causal = False,
        memory_is_causal = False,
    ) -> Tensor:

        x = tgt
        if layer.norm_first:
            x = x + layer._sa_block(layer.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            c,att = _mha_block_with_weights(layer,layer.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + c
            x = x + layer._ff_block(layer.norm3(x))
        else:
            x = layer.norm1(x + layer._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            c,att = _mha_block_with_weights(layer,x, memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = layer.norm2(x + c)
            x = layer.norm3(x + layer._ff_block(x))

        return x,att


def forward_with_return_attention_from_last(decoder, tgt, memory, tgt_mask= None,
                memory_mask = None, tgt_key_padding_mask = None,
                memory_key_padding_mask = None, tgt_is_causal = None,
                memory_is_causal = False) -> Tensor:
        output = tgt
        seq_len = t._get_seq_len(tgt, decoder.layers[0].self_attn.batch_first)
        tgt_is_causal = t._detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for i in range(len(decoder.layers)):
            mod = decoder.layers[i]
            if i == len(decoder.layers) - 1:
              output,att = forward_with_return_att(mod,output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)
            else:
              output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)

        if decoder.norm is not None:
            output = decoder.norm(output)

        return output, att


def transformer_forward_with_weights(model,src, tgt, src_mask = None, tgt_mask = None,
                memory_mask= None, src_key_padding_mask= None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None,
                src_is_causal = None, tgt_is_causal = None,
                memory_is_causal = False):
        is_batched = src.dim() == 3
        if not model.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif model.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        
        if src.size(-1) != model.d_model or tgt.size(-1) != model.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = model.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
                              is_causal=src_is_causal)
        out,att = forward_with_return_attention_from_last(model.decoder,tgt,memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
        return out,att

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 500):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor,
                with_weights = False):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        if with_weights:
            outs,att = transformer_forward_with_weights(self.transformer,src_emb,tgt_emb,src_mask,tgt_mask,None,src_padding_mask,tgt_padding_mask,memory_key_padding_mask)
            return self.generator(outs),att
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, with_weights = False):
        if with_weights:
          return forward_with_return_attention_from_last(self.transformer.decoder,self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
        
def generate_square_subsequent_mask(sz,device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def create_mask(src, tgt,pad_idx,device):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len,device)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

        src_padding_mask = (src == pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
 

def greedy_decode(model, src, src_mask, max_len, start_symbol,device,unk_idx,eos_idx,without_unk = False, left_unk_for_gen = False, with_weights = False):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    decoded = ys
    for i in range(max_len-1):
        att = None
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0),device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask,with_weights)
        if with_weights:
          out, att = out[0], out[1]
          att = att.squeeze(0)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        change = 0
        if without_unk:
            if next_word == unk_idx:
                _,top_tokens = torch.topk(prob,k = 2,dim = 1)
                next_word = top_tokens[0][1].item()
                change = 1
        decoded = torch.cat([decoded,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        
        if left_unk_for_gen and change:
            ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(unk_idx)], dim=0)
        else:
            ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break
    if with_weights:
      return decoded,att
    return decoded


def beam_search_decode(model,src,src_mask,max_len,start_symbol, k, device,unk_idx,eos_idx, without_unk = False, with_weights = False):
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1,1).fill_(start_symbol).type(torch.long).to(device)

    att = None
    beam = [(ys.squeeze(1),0,att)]
    for i in range(max_len):
        candidates = []
        tgt_mask = (generate_square_subsequent_mask(i+1,device).type(torch.bool)).to(device)
        for sequence, p,a in beam:
            if sequence[-1] == eos_idx:
                candidates.append((sequence,p,a))
                continue
            sequence = sequence.to(device)
            out = model.decode(sequence.unsqueeze(1), memory, tgt_mask,with_weights)
            if with_weights:
              out, att = out[0], out[1]
              att = att.squeeze(0)

            out = out.transpose(0, 1)
            logits = model.generator(out[:, -1])
            log_probs = logits.log_softmax(dim=-1)

            if without_unk:
                top_log_probs,top_inds = log_probs[0].topk(k + 1)
                top_log_probs = top_log_probs[top_inds != unk_idx][:k]
                top_inds = top_inds[top_inds != unk_idx][:k]
            else:
                top_log_probs,top_inds = log_probs[0].topk(k)

            for token,log_prob in zip(top_inds, top_log_probs):
                new_sequence = torch.cat((sequence,token.unsqueeze(0)))
                new_p = p + log_prob.item()
                candidates.append((new_sequence,new_p,att))
        beam = sorted(candidates,key=lambda x: x[1], reverse=True)[:k]
    if with_weights:
      return sorted(beam,key=lambda x: x[1], reverse=True)[0][0],sorted(beam,key=lambda x: x[1], reverse=True)[0][2]
    return sorted(beam,key=lambda x: x[1], reverse=True)[0][0]


def translate(model, src, bos_idx,unk_idx,eos_idx, device,en_vocab, without_unk = False, left_unk_for_gen = False, beam_search = False, k = 0, with_weights = False):
    model.eval()
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    if beam_search:
        tgt_tokens = beam_search_decode(model,  src, src_mask, num_tokens + 10, bos_idx, k, device, unk_idx, eos_idx, without_unk = without_unk,with_weights = with_weights)
        if with_weights:
          tgt_tokens,atts = tgt_tokens[0].flatten(),tgt_tokens[1]
    else:
        tgt_tokens = greedy_decode(
        model,  src, src_mask, num_tokens + 10, bos_idx,device, unk_idx, eos_idx, without_unk = without_unk, left_unk_for_gen=left_unk_for_gen, with_weights = with_weights)
        if with_weights:
          tgt_tokens,atts = tgt_tokens[0].flatten(),tgt_tokens[1]
    if with_weights:
      return ' '.join(en_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).removeprefix('<bos> ').removesuffix(' <eos>'), atts
    return ' '.join(en_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).removeprefix('<bos> ').removesuffix(' <eos>')


def make_translation_dict(model,train_iter, pad_idx,device):
    translation_vocab = defaultdict(float)
    for (src, tgt,de_text,en_text) in tqdm(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input,pad_idx,device)
        with torch.no_grad():
            logits,att = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask, with_weights = True)

        for i in range(len(en_text)):
            en = en_text[i]
            de = de_text[i]

            en_seq = en.split(' ') + ['<eos>']
            de_seq = ['<bos>'] + de.split(' ') + ['<eos>']
            curr_att = att[i,:len(en_seq),:len(de_seq)]

            probs,top_indices = curr_att.sort(descending = True,dim = 0)
            for j in range(curr_att.shape[1]):
                translation_vocab[(de_seq[j], en_seq[top_indices[0][j]])] += probs[0][j]
                translation_vocab[(de_seq[j], en_seq[top_indices[1][j]])] += probs[1][j]

            probs,top_indices = curr_att.sort(descending = True,dim = -1)
            for j in range(curr_att.shape[0]):
                translation_vocab[(de_seq[top_indices[j][0]], en_seq[j])] += probs[j][0]
                translation_vocab[(de_seq[top_indices[j][1]], en_seq[j])] += probs[j][1]
    translation_vocab = {k:translation_vocab[k].item() for k in translation_vocab.keys()}
    
    df = pd.DataFrame([(k[0], k[1], v) for k, v in translation_vocab.items()], columns=['de', 'en', 'cum_prob'])
    df = df[~df['en'].str.contains('\n')]
    clear_df = df.loc[df.groupby('de').agg({'cum_prob':'idxmax'})['cum_prob'].values]
    clear_vocab = clear_df.set_index('de')['en'].to_dict()
    return clear_vocab

def remove_unk(model, src, decoded,bos_idx,unk_idx,en_vocab,device):
    model.eval()
    src = src.to(device)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(bos_idx).type(torch.long).to(device)

    for i in range(len(decoded)):
        att = None
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0),device)
                    .type(torch.bool)).to(device)
        
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(en_vocab.lookup_indices([decoded[i]])[0])], dim=0)
        if en_vocab.lookup_indices([decoded[i]])[0] == unk_idx:
                _,top_tokens = torch.topk(prob,k = 2,dim = 1)
                next_word = top_tokens[0][0].item()
                if next_word == unk_idx:
                    next_word = top_tokens[0][1].item()
                decoded[i] = en_vocab.lookup_token(next_word)
    return decoded