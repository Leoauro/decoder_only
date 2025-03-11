import os

from tokenizers import Tokenizer

from tokenizer.tokenizer_train import data_dir


class CustomTokenizer:
    def __init__(self):
        tokenizer_file = os.path.join(data_dir(), "data", "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

    def tokenize(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def token_num(self) -> int:
        size = self.tokenizer.get_vocab_size()
        return size

    def detokenize(self, tokens: list[int]) -> str:
        decoded = ""
        for item in tokens:
            c = self.tokenizer.decode([item])
            decoded += c
        return decoded
