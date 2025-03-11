import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from decoder_only.config import config
from decoder_only.model.model import Model
from decoder_only.train.dataset import DecoderOnlyDataset

model = Model().cuda()
# 获取数据集
dataset = DecoderOnlyDataset("../data/train.csv")
# 装载到 DataLoader，
dataloader = DataLoader(dataset, config.BATCH_SIZE, shuffle=True)
# 损失函数 CrossEntropyLoss 已经隐式包含 softmax
loss_func = nn.CrossEntropyLoss(ignore_index=3).cuda()
# 优化器
trainer = AdamW(params=model.parameters(), lr=config.LEARNING_RATE)

for epoch in range(config.EPOCHS):
    t = tqdm(dataloader)
    for input_id, input_m, output_id in t:
        # 数据放入到 GPU上,进行模型训练,
        output = model(input_id.cuda(), input_m.cuda())
        # output = output.permute(0, 2, 1)
        loss = loss_func(output.reshape(-1, config.VOCAB_SIZE), output_id.reshape(-1).cuda())
        loss.backward()
        # 梯度裁剪,计算梯度的总范数（如L2范数），若超过阈值max_norm，则按比例缩放梯度，使其范数等于阈值。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        trainer.step()
        trainer.zero_grad()
        t.set_description(str(loss.item()))

torch.save(model.state_dict(), "model.pth")
