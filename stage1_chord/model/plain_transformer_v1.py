import sys

import torch
from torch import nn
import torch.nn.functional as F

from .optimus_txl_decoder import OptimusTXLDecoder

from .transformer_helpers import (
  WordEmbedding,
  weights_init
)

class PlainTransformer(nn.Module):
  def __init__(self, d_word_embed, vocab_size,
               dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, dec_mem_len, dec_tgt_len,
               dec_dropout=0.1, dec_activation='relu', 
               pad_index=None, pre_lnorm=False,
               global_attention_heads=2, local_attention_window=4
  ):
    super(PlainTransformer, self).__init__()

    self.d_word_embed = d_word_embed
    self.vocab_size = vocab_size

    self.dec_n_layer = dec_n_layer
    self.dec_n_head = dec_n_head
    self.dec_d_model = dec_d_model
    self.dec_d_ff = dec_d_ff
    self.dec_dropout = dec_dropout
    self.dec_activation = dec_activation
    self.dec_mem_len = dec_mem_len
    self.dec_tgt_len = dec_tgt_len

    self.word_emb = WordEmbedding(vocab_size, d_word_embed, dec_d_model)
    self.emb_dropout = nn.Dropout(dec_dropout)
    if pad_index is None:
      self.pad_index = self.vocab_size - 1
    else:
      self.pad_index = pad_index

    self.decoder = OptimusTXLDecoder(
                    dec_n_layer, dec_n_head, dec_d_model, dec_d_model // dec_n_head, dec_d_ff,
                    None, dec_dropout, dec_dropout,
                    tgt_len=dec_tgt_len, mem_len=dec_mem_len, ext_len=0,
                    pre_lnorm=pre_lnorm, use_segment_emb=False
                  )
    self.dec_out_proj = nn.Linear(dec_d_model, vocab_size)

    # 全局注意力和局部注意力参数
    self.global_attention_heads = global_attention_heads
    self.local_attention_window = local_attention_window

    self.apply(weights_init)

  def generate(self, dec_input, dec_mems):
    dec_word_emb = self.word_emb(dec_input)
    dec_input = self.emb_dropout(dec_word_emb)
    dec_out = self.decoder(dec_input, None, *dec_mems)
    dec_logits = self.dec_out_proj(dec_out[0])[-1, 0, :]
    new_dec_mems = dec_out[1:]

    return dec_logits, new_dec_mems

  def forward(self, dec_input, dec_mems, dec_seg_len=None, return_avg_attn=False):
    dec_word_emb = self.word_emb(dec_input)
    dec_input = self.emb_dropout(dec_word_emb)
    # print ('[debug] in model forward()')

    if not return_avg_attn:
      dec_out = self.decoder(dec_input, None, *dec_mems, dec_seg_len=dec_seg_len)
      dec_logits = self.dec_out_proj(dec_out[0])
      new_dec_mems = dec_out[1:]
      return dec_logits, new_dec_mems

    else:
      dec_out, avg_attn_probs = self.decoder(
                                  dec_input, None, *dec_mems, 
                                  dec_seg_len=dec_seg_len,
                                  return_avg_attn=True
                                )
      dec_logits = self.dec_out_proj(dec_out[0])
      new_dec_mems = dec_out[1:]

      return dec_logits, new_dec_mems, avg_attn_probs

  def compute_loss(self, dec_logits, dec_tgt, reduction='mean'):
    ce_loss = F.cross_entropy(
                    dec_logits.view(-1, dec_logits.size(-1)),
                    dec_tgt.contiguous().view(-1),
                    ignore_index=self.pad_index,
                    reduction=reduction
                  )

    return {
      'ce_loss': ce_loss,
      'total_loss': ce_loss
    }

  def forward_with_attention(self, dec_input, dec_mems, dec_seg_len=None):
    dec_word_emb = self.word_emb(dec_input)
    dec_input = self.emb_dropout(dec_word_emb)

    # 全局注意力和局部注意力的掩码
    global_attn_mask = torch.zeros(dec_input.size(0), dec_input.size(1), dec_input.size(1), dtype=torch.bool, device=dec_input.device)
    local_attn_mask = torch.zeros(dec_input.size(0), dec_input.size(1), dec_input.size(1), dtype=torch.bool, device=dec_input.device)

    # 设置全局注意力的掩码
    for i in range(dec_input.size(0)):
      global_attn_mask[i, :, :] = True

    # 设置局部注意力的掩码
    for i in range(dec_input.size(0)):
      for j in range(dec_input.size(1)):
        start = max(0, j - self.local_attention_window)
        end = min(dec_input.size(1), j + self.local_attention_window + 1)
        local_attn_mask[i, j, start:end] = True

    # 解码器前向传播
    dec_out = self.decoder(dec_input, None, *dec_mems, dec_seg_len=dec_seg_len, global_attn_mask=global_attn_mask, local_attn_mask=local_attn_mask)
    dec_logits = self.dec_out_proj(dec_out[0])
    new_dec_mems = dec_out[1:]

    return dec_logits, new_dec_mems