#MSELoss pretrain_dataset.test
"""
🧪 测试 PretrainDataset（MSE 回归版）是否正确实现：
   - 输入格式：<compute>expr</compute> 封装
   - 输出：input_ids [L], target: float
   - 不使用 mask，适配 MSE loss

依赖：
  - my_tokenizers.hf_math_tokenizer.HFMathTokenizer
  - model_utils.pretrain_dataset.PretrainDataset
"""

import os
import sys
import torch

# 添加项目根目录到 sys.path（适配从 tests/ 或根目录运行）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from my_tokenizers.hf_math_tokenizer import HFMathTokenizer
from model_ultils.pretrain_dataset import PretrainDataset


def create_test_data():
    """生成临时测试数据文件"""
    test_data = [
        "1+1=2",
        "5-2=3",
        "1.5+2.5=4.0",
        "-3*2=-6",
        "100-99=1",
        "",               # 空行（应跳过）
        "no_eq_sign",     # 无效（无 '='）
        "a+b=c",          # 无效（答案非数字 → 应跳过）
        "2^3=8",          # 合法（若 tokenizer 支持 '^'）
    ]
    test_dir = os.path.join(project_root, "data")
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, "test_mse_dataset.txt")
    with open(test_path, 'w', encoding='utf-8') as f:
        for line in test_data:
            f.write(line + "\n")
    return test_path


def test_pretrain_dataset_mse():
    print("🧪 正在测试 PretrainDataset（MSE 回归模式）...")

    # 1. 创建测试数据
    test_path = create_test_data()
    print(f"✅ 生成测试数据: {test_path}")

    # 2. 初始化 tokenizer 和 dataset
    tokenizer = HFMathTokenizer()
    # 确保 tokenizer 有必要属性（兼容你的实现）
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<pad>"
        print("⚠️ Warning: pad_token not set; using '<pad>'")

    dataset = PretrainDataset(
        data_path=test_path,
        tokenizer=tokenizer,
        max_length=32  # 足够容纳 <s><compute>1+1</compute></s>
    )

    print(f"📊 Dataset size: {len(dataset)} (expected: 6 valid lines)")

    # 3. 测试第 0 个样本 ("1+1=2" → expr="<compute>1+1</compute>", target=2.0)
    print("\n🔍 测试样本 0: '1+1=2'")
    input_ids, target = dataset[0]

    # 解码 input_ids（查看实际输入文本）
    decoded_input = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    print(f"   Input (input_ids) : {input_ids.tolist()}")
    print(f"   Decoded input     : '{decoded_input}'")
    print(f"   Target (float)    : {target.item():.1f} (dtype={target.dtype})")
        # ✅ 关键修复：提前计算 non_pad_count
    pad_id = tokenizer.pad_token_id
    non_pad_count = (input_ids != pad_id).sum().item()
    print(f"   Non-pad tokens    : {input_ids[:non_pad_count].tolist()}")
    
    # 可选：打印每个 token（极有助于 debug）
    print(f"   Tokens (decoded):")
    for i, tid in enumerate(input_ids[:non_pad_count]):
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        print(f"     [{i:2d}] {tid:5d} → '{tok}'")
    # 4. 关键属性检查
    bos_token = getattr(tokenizer, 'bos_token', '<s>')
    eos_token = getattr(tokenizer, 'eos_token', '</s>')
    bos_task_token = getattr(tokenizer, 'bos_task_token', '<compute>')
    eos_task_token = getattr(tokenizer, 'eos_task_token', '</compute>')

    # 验证 decoded_input 是否包含完整结构
    expected_substrings = [bos_token, bos_task_token, "1+1", eos_task_token, eos_token]
    missing = [s for s in expected_substrings if s not in decoded_input]

    # 5. 断言检查（核心验证）
    try:
        # (a) 样本数量正确（跳过 3 行：空行 + no_eq + a+b=c）
        assert len(dataset) == 6, f"Expected 6 valid samples, got {len(dataset)}"

        # (b) input_ids 是 LongTensor，长度 == max_length
        assert isinstance(input_ids, torch.LongTensor), "input_ids must be LongTensor"
        assert input_ids.size(0) == 32, f"input_ids length must be max_length=32, got {input_ids.size(0)}"

        # (c) target 是 float32 标量
        assert isinstance(target, torch.FloatTensor), f"target must be FloatTensor, got {type(target)}"
        assert target.ndim == 0, f"target must be scalar (0-dim), got {target.ndim}-dim"

        # (d) target 值正确
        assert abs(target.item() - 2.0) < 1e-5, f"target should be 2.0, got {target.item()}"

        # (e) padding 正确：末尾应为 pad_token_id
        pad_id = tokenizer.pad_token_id
        last_token = input_ids[-1].item()
        assert last_token == pad_id, f"Last token should be pad_id={pad_id}, got {last_token}"

        # (f) 至少有一个非-pad token（防止全 pad）
        non_pad_count = (input_ids != pad_id).sum().item()
        assert non_pad_count > 5, f"Too few non-pad tokens: {non_pad_count}"

        print("\n✅ 所有断言通过！PretrainDataset（MSE版）工作正常。")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)

    # 6. 打印 tokenizer 关键 token IDs（诊断用）
    print("\nℹ️ Tokenizer token IDs:")
    for token in ['<s>', '</s>', '<pad>', '<compute>', '</compute>', '1', '+']:
        try:
            tid = tokenizer.convert_tokens_to_ids([token])[0]
            print(f"   '{token}': {tid}")
        except Exception:
            print(f"   '{token}': N/A")

    # 7. 额外：测试边界样本（负数 & 小数）
    print("\n🔍 测试样本 2: '1.5+2.5=4.0'")
    _, target2 = dataset[2]
    print(f"   Target: {target2.item():.1f}")
    assert abs(target2.item() - 4.0) < 1e-5, "Sample 2 target mismatch"

    print("🔍 测试样本 3: '-3*2=-6'")
    _, target3 = dataset[3]
    print(f"   Target: {target3.item():.1f}")
    assert abs(target3.item() + 6.0) < 1e-5, "Sample 3 target mismatch"

    # 清理
    os.remove(test_path)
    print(f"\n🧹 临时文件 '{test_path}' 已清理。")
    print("🎉 测试完成！")


if __name__ == "__main__":
    test_pretrain_dataset_mse()
#CE Loss pretrain_dataset.test
# # -*- coding: utf-8 -*-
# """
# 测试 PretrainDataset 是否正确实现 left-to-right completion 模式
# 依赖：
#   - my_tokenizers.hf_math_tokenizer.HFMathTokenizer
#   - dataset.pretrain_dataset.PretrainDataset
# """

# import os
# import sys
# import torch

# # 添加项目根目录到 sys.path（适配根目录运行）
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

# # ✅ 导入你已实现的模块（不重复定义类！）
# from my_tokenizers.hf_math_tokenizer import HFMathTokenizer
# from model_ultils.pretrain_dataset import PretrainDataset


# def create_test_data():
#     """生成临时测试数据文件"""
#     test_data = [
#         "1+1=2",
#         "5-2=3",
#         "1.5+2.5=4.0",
#         "100-99=1",
#         "",               # 空行（应跳过）
#         "invalid_expr",   # 无效行（应跳过）
#         "a+b=c",          # 无效（你的 tokenizer 可能不支持，但格式合法 → 会尝试处理）
#     ]
#     test_path = os.path.join(project_root, "data", "test_sample.txt")
#     os.makedirs(os.path.dirname(test_path), exist_ok=True)
#     with open(test_path, 'w', encoding='utf-8') as f:
#         for line in test_data:
#             f.write(line + "\n")
#     return test_path


# def test_pretrain_dataset():
#     print("🧪 正在测试 PretrainDataset...")

#     # 1. 创建测试数据
#     test_path = create_test_data()
#     print(f"✅ 生成测试数据: {test_path}")
    
#     # 2. 初始化 tokenizer 和 dataset
#     tokenizer = HFMathTokenizer()
#     dataset = PretrainDataset(
#         data_path=test_path,
#         tokenizer=tokenizer,
#         max_length=11
#     )

#     print(f"📊 Dataset size: {len(dataset)} (expected: 5 valid lines)")

#     # 3. 测试第 0 个样本 ("12+34=46" → prompt="12+34=", answer="46")
#     print("\n🔍 测试样本 0: '1+1=2'")
#     X, Y, loss_mask = dataset[0]

#     # 解码 X（模型输入）
#     decoded_input = tokenizer.decode(X.tolist(), skip_special_tokens=False)
#     print(f"   Input (X)  : '{decoded_input}'")

#     # 找出 Y 中参与 loss 的位置（Y != -100）
#     loss_positions = torch.where(Y != -100)[0].tolist()
#     loss_tokens = [Y[i].item() for i in loss_positions]
#     decoded_loss_targets = tokenizer.decode(loss_tokens, skip_special_tokens=False)
#     print(f"   Loss tokens: {loss_tokens} → '{decoded_loss_targets}'")
#     print(f"   Loss mask  : {loss_mask.tolist()}")

#     # ✅ 预期：loss 应覆盖 "46</s>" 的 token IDs（例如 [?, ?, 2]）
#     eos_id = getattr(tokenizer, 'eos_token_id', tokenizer.convert_tokens_to_ids(['</s>'])[0])
#     print(f"   EOS token ID: {eos_id}")

#     # 断言检查（关键！）
#     try:
#         assert len(dataset) == 5, f"Expected 5 valid samples, got {len(dataset)}"
#         assert "=" in decoded_input, "Input should contain '='"
#         assert len(loss_positions) > 0, "Should have at least one loss position"
#         assert eos_id in loss_tokens, f"EOS ({eos_id}) should be in loss targets"
#         print("\n✅ 所有断言通过！PretrainDataset 工作正常。")
#     except AssertionError as e:
#         print(f"\n❌ 测试失败: {e}")
#         sys.exit(1)

#     # 4. 可选：打印 tokenizer 关键属性（帮你验证一致性）
#     print("\nℹ️ Tokenizer info:")
#     print(f"   vocab_size   : {tokenizer.vocab_size}")
#     print(f"   pad_token_id : {getattr(tokenizer, 'pad_token_id', 'N/A')}")
#     print(f"   bos_token_id : {getattr(tokenizer, 'bos_token_id', 'N/A')}")
#     print(f"   eos_token_id : {eos_id}")
#     print(f"   '4' token ID : {tokenizer.convert_tokens_to_ids(['4'])}")

#     # 清理
#     os.remove(test_path)
#     print(f"\n🧹 临时文件已清理。")


# if __name__ == "__main__":
#     test_pretrain_dataset()
