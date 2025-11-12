#MSE Loss model_ultils/pretrain_dataset.py
# model_utils/pretrain_dataset.py

import torch
from torch.utils.data import Dataset

class MSEPretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)  # → List[(expr_str: str, target: float)]


    def load_data(self, path):
        """从 txt 加载并解析为 (expr_str, float_target) 列表
        输入格式：expr = answer（如 "3 + 4 * 2 = 11"）
        内部转换为：<compute>expr</compute> → float(answer)
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                if '=' not in line:
                    print(f"⚠️ Warning: Line {idx} skipped (no '=' found): '{line}'")
                    continue
                try:
                    # 分割一次 '='，防止含多个 '=' 的异常数据
                    left, right = line.split('=', 1)
                    expr = left.strip()
                    ans_str = right.strip()
                    if not expr or not ans_str:
                        print(f"⚠️ Warning: Line {idx} has empty expr or answer: '{line}'")
                        continue

                    # 🔹 关键：仅用 float() 解析答案，禁用 eval（安全第一！）
                    try:
                        target = float(ans_str)
                    except ValueError:
                        print(f"⚠️ Warning: Line {idx} answer not numeric: '{ans_str}' in '{line}'")
                        continue

                    # ✅ 构造任务化表达式：<compute>expr</compute>
                    # 注意：expr 中的 = 已被剥离，不会干扰
                    task_expr = f"<compute>{expr}</compute>"
                    samples.append((task_expr, target))

                except Exception as e:
                    print(f"⚠️ Warning: Line {idx} failed to parse: '{line}' | Error: {e}")
                    continue
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        expr_with_tags, target = self.samples[index]
        bos_token = getattr(self.tokenizer, 'bos_token', '')
        eos_token = getattr(self.tokenizer, 'eos_token', '')
        full_text = bos_token + expr_with_tags + eos_token
    
        enc = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        input_ids = enc.input_ids.squeeze(0)
    
        if input_ids.size(0) < self.max_length:
            pad_len = self.max_length - input_ids.size(0)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
        else:
            input_ids = input_ids[:self.max_length]
        target_tensor = torch.tensor([target], dtype=torch.float32)  # shape: [1]
        return input_ids, target_tensor





# #CE Loss model_ultils/pretrain_dataset.py

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)  # → List[(prompt, answer)]

    def load_data(self, path):
        """从 txt 加载并解析为 (prompt, answer) 列表"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                if '=' not in line:
                    print(f"⚠️ Warning: Line {idx} skipped (no '=' found): '{line}'")
                    continue
                try:
                    # 分割一次 '='，防止含多个 '=' 的异常数据
                    left, right = line.split('=', 1)
                    left = left.strip()
                    right = right.strip()
                    if left and right:  # 确保非空
                        samples.append((left + "=", right))  # prompt 以 '=' 结尾
                except Exception as e:
                    print(f"⚠️ Warning: Line {idx} failed to parse: '{line}' | Error: {e}")
                    continue
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        prompt, answer = self.samples[index]  # e.g., ("12+34=", "46")
        bos_token = getattr(self.tokenizer, 'bos_token', '<s>')
        prompt_with_bos = bos_token +prompt
        # 🔹 Step 1: 编码 prompt（带 BOS）
        prompt_enc = self.tokenizer(
            prompt_with_bos,
            add_special_tokens=False,      # ← adds <s>
            return_tensors='pt',
            truncation=False,
        )
        prompt_ids = prompt_enc.input_ids.squeeze(0)  # [L_p]

        # 🔹 Step 2: 编码 answer + EOS（不加 BOS！避免重复）
        # 获取 EOS token ID（适配你的 tokenizer：可能是 2 或 3）
        eos_token = getattr(self.tokenizer, 'eos_token', '</s>')
        # 手动拼接 answer + EOS
        answer_with_eos = answer + eos_token
        answer_enc = self.tokenizer(
            answer_with_eos,
            add_special_tokens=False,     # ← critical! no extra <s>
            return_tensors='pt',
            truncation=False,
        )
        answer_ids = answer_enc.input_ids.squeeze(0)  # [L_a]

        # 🔹 Step 3: 拼接 + padding/truncation
        full_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
        else:
            pad_len = self.max_length - len(full_ids)
            full_ids = torch.cat([
                full_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])

        # 🔹 Step 4: 构造 labels（仅 answer+EOS 区域有效）
        labels = torch.full_like(full_ids, -100)  # -100 = ignore in CrossEntropy
        start_ans = len(prompt_ids)
        end_ans = min(start_ans + len(answer_ids), self.max_length)
        labels[start_ans:end_ans] = full_ids[start_ans:end_ans]

        # 🔹 Step 5: 构造自回归输入/目标
        X = full_ids[:-1].clone().detach().long()   # input for model
        Y = labels[1:].clone().detach().long()      # target (with -100)
        loss_mask = (Y != -100).long()              # where to compute loss

        return X, Y, loss_mask