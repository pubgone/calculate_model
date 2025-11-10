#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 PretrainDataset 是否正确实现 left-to-right completion 模式
依赖：
  - my_tokenizers.hf_math_tokenizer.HFMathTokenizer
  - dataset.pretrain_dataset.PretrainDataset
"""

import os
import sys
import torch

# 添加项目根目录到 sys.path（适配根目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ✅ 导入你已实现的模块（不重复定义类！）
from my_tokenizers.hf_math_tokenizer import HFMathTokenizer
from model_ultils.pretrain_dataset import PretrainDataset


def create_test_data():
    """生成临时测试数据文件"""
    test_data = [
        "1+1=2",
        "5-2=3",
        "1.5+2.5=4.0",
        "100-99=1",
        "",               # 空行（应跳过）
        "invalid_expr",   # 无效行（应跳过）
        "a+b=c",          # 无效（你的 tokenizer 可能不支持，但格式合法 → 会尝试处理）
    ]
    test_path = os.path.join(project_root, "data", "test_sample.txt")
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    with open(test_path, 'w', encoding='utf-8') as f:
        for line in test_data:
            f.write(line + "\n")
    return test_path


def test_pretrain_dataset():
    print("🧪 正在测试 PretrainDataset...")

    # 1. 创建测试数据
    test_path = create_test_data()
    print(f"✅ 生成测试数据: {test_path}")

    # 2. 初始化 tokenizer 和 dataset
    tokenizer = HFMathTokenizer()
    dataset = PretrainDataset(
        data_path=test_path,
        tokenizer=tokenizer,
        max_length=11
    )

    print(f"📊 Dataset size: {len(dataset)} (expected: 5 valid lines)")

    # 3. 测试第 0 个样本 ("12+34=46" → prompt="12+34=", answer="46")
    print("\n🔍 测试样本 0: '1+1=2'")
    X, Y, loss_mask = dataset[0]

    # 解码 X（模型输入）
    decoded_input = tokenizer.decode(X.tolist(), skip_special_tokens=False)
    print(f"   Input (X)  : '{decoded_input}'")

    # 找出 Y 中参与 loss 的位置（Y != -100）
    loss_positions = torch.where(Y != -100)[0].tolist()
    loss_tokens = [Y[i].item() for i in loss_positions]
    decoded_loss_targets = tokenizer.decode(loss_tokens, skip_special_tokens=False)
    print(f"   Loss tokens: {loss_tokens} → '{decoded_loss_targets}'")
    print(f"   Loss mask  : {loss_mask.tolist()}")

    # ✅ 预期：loss 应覆盖 "46</s>" 的 token IDs（例如 [?, ?, 2]）
    eos_id = getattr(tokenizer, 'eos_token_id', tokenizer.convert_tokens_to_ids(['</s>'])[0])
    print(f"   EOS token ID: {eos_id}")

    # 断言检查（关键！）
    try:
        assert len(dataset) == 5, f"Expected 5 valid samples, got {len(dataset)}"
        assert "=" in decoded_input, "Input should contain '='"
        assert len(loss_positions) > 0, "Should have at least one loss position"
        assert eos_id in loss_tokens, f"EOS ({eos_id}) should be in loss targets"
        print("\n✅ 所有断言通过！PretrainDataset 工作正常。")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)

    # 4. 可选：打印 tokenizer 关键属性（帮你验证一致性）
    print("\nℹ️ Tokenizer info:")
    print(f"   vocab_size   : {tokenizer.vocab_size}")
    print(f"   pad_token_id : {getattr(tokenizer, 'pad_token_id', 'N/A')}")
    print(f"   bos_token_id : {getattr(tokenizer, 'bos_token_id', 'N/A')}")
    print(f"   eos_token_id : {eos_id}")
    print(f"   '4' token ID : {tokenizer.convert_tokens_to_ids(['4'])}")

    # 清理
    os.remove(test_path)
    print(f"\n🧹 临时文件已清理。")


if __name__ == "__main__":
    test_pretrain_dataset()
