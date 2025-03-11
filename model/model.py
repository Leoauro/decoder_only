import torch
from torch import nn

from config import config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = Embedding()
        self.decoder = DecoderOnly()
        self.lm_head = nn.Linear(config.D_MODEL, config.VOCAB_SIZE)

    def forward(self, seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(seq)
        out = self.decoder(embedded, seq_mask)
        out = self.lm_head(out)
        return out


class DecoderOnly(nn.Module):
    def __init__(self):
        super(DecoderOnly, self).__init__()
        # 使用transformer中的EncoderLayer实现decoder-only ，
        # transformer中的TransformerDecoder依赖memory参数，无法实现decoder-only模式
        decoder_layer = nn.TransformerEncoderLayer(config.D_MODEL, config.N_HEAD, dim_feedforward=1024,
                                                   batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.DECODER_LAYERS,
                                             norm=nn.LayerNorm(config.D_MODEL))

    def forward(self, seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        causal_mask = self.generate_square_subsequent_mask(seq.shape[-2]).to(seq.device)
        out = self.decoder(src=seq, mask=causal_mask, src_key_padding_mask=seq_mask)
        return out

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """因果掩码（下三角矩阵）"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.pos_ebd = nn.Embedding(config.SEQ_TOKEN_LEN, config.D_MODEL)
        self.pos_t = torch.arange(0, config.SEQ_TOKEN_LEN).reshape(1, config.SEQ_TOKEN_LEN)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        ebd = self.embedding(seq)
        position = self.pos_t[:, :seq.shape[-1]].to(seq.device)
        position = self.pos_ebd(position)
        return ebd + position
