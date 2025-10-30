# pretrain_dataset.py
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    samples.append(line)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = self.samples[index]  # 原始表达式，如 "12+34=46"

        # ✅ 直接交给 tokenizer 处理，它会自动加 <s> 和 </s>
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True  # ← 关键：启用自动添加
        )

        input_ids = encoding.input_ids.squeeze()  # [max_length]
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  # 注意：用 .pad_id

        # 构造自回归任务
        X = input_ids[:-1].clone().detach().long()
        Y = input_ids[1:].clone().detach().long()
        loss_mask = loss_mask[1:].clone().detach().long()

        return X, Y, loss_mask