from typing import Tuple, List

import datasets
import numpy as np
import torch
from torch.utils import data

from decoder_only.config import config
from decoder_only.tokenizer.tokenizer import CustomTokenizer


class DecoderOnlyDataset(data.Dataset):
    def __init__(self, csv_path: str) -> None:
        super().__init__()
        self.tokenizer = CustomTokenizer()
        self.cls_token = self.tokenizer.tokenize("[CLS]")
        self.sep_token = self.tokenizer.tokenize("[SEP]")
        self.mask_token = 1
        self.pad_token = self.tokenizer.tokenize("[PAD]")
        self.dataset = datasets.Dataset.from_csv(csv_path)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset.__getitem__(index)
        ret = self.to_token(item["text1"])
        return ret

    def to_token(self, content: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer.tokenize(content)
        return self.pad_to_max_length(tokens)

    def pad_to_max_length(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_mask = np.array([False] * config.SEQ_TOKEN_LEN)
        if len(tokens) >= config.SEQ_TOKEN_LEN:
            tokens = tokens[:config.SEQ_TOKEN_LEN]
            target_tokens = tokens[1:] + self.sep_token
            return (torch.tensor(tokens, dtype=torch.long),
                    torch.tensor(tokens_mask, dtype=torch.bool),
                    torch.tensor(target_tokens, dtype=torch.long))

        pad_len = config.SEQ_TOKEN_LEN - len(tokens)
        target_tokens = tokens[1:] + self.sep_token
        tokens += self.pad_token * pad_len
        target_tokens += self.pad_token * pad_len
        tokens_mask[-pad_len:] = True
        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(tokens_mask, dtype=torch.bool),
                torch.tensor(target_tokens, dtype=torch.long))
