import torch

from decoder_only.config import config
from decoder_only.model.model import Model
from decoder_only.tokenizer.tokenizer import CustomTokenizer


def predict(decoder_only_model: Model, prompt: str):
    tokenizer = CustomTokenizer()
    sep_token = tokenizer.tokenize("[SEP]")
    tokens = tokenizer.tokenize(prompt)
    tokens_mask = [False] * len(tokens)
    target_id_lst = []
    while len(tokens) < config.SEQ_TOKEN_LEN:
        input_token = torch.tensor(tokens, dtype=torch.long).cuda().unsqueeze(0)
        input_mask = torch.tensor(tokens_mask, dtype=torch.bool).cuda().unsqueeze(0)
        output = decoder_only_model(input_token, input_mask)
        word_id = torch.argmax(output[0][-1])
        target_id_lst.append(word_id.item())
        tokens.append(word_id.item())
        tokens_mask.append(False)
        if word_id.item() == sep_token[0]:
            break
    result = tokenizer.detokenize(target_id_lst)

    return prompt + result


if __name__ == "__main__":
    model = Model().cuda()
    model.load_state_dict(torch.load("./train/model.pth"))
    print(predict(model, "黄河"))
