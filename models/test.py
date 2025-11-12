#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🧪 测试 MiniMindForRegression 回归头（适配已配置好的 HFMathTokenizer）
"""

import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from my_tokenizers.hf_math_tokenizer import HFMathTokenizer
    from models.model_minimind import MiniMindConfig, MiniMindForRegression
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def validate_tokenizer(tokenizer):
    """验证 tokenizer 已正确配置 special tokens（只读检查，不修改）"""
    required_attrs = ['pad_token_id', 'bos_token_id', 'eos_token_id']
    for attr in required_attrs:
        assert hasattr(tokenizer, attr), f"tokenizer missing {attr}"
        val = getattr(tokenizer, attr)
        assert isinstance(val, int) and val >= 0, f"{attr}={val} invalid"

    # 检查 <compute> 是否在 vocab 中
    compute_id = tokenizer.convert_tokens_to_ids(['<compute>'])
    close_compute_id = tokenizer.convert_tokens_to_ids(['</compute>'])
    assert compute_id[0] != tokenizer.unk_token_id, "<compute> not in vocab"
    assert close_compute_id[0] != tokenizer.unk_token_id, "</compute> not in vocab"

    print("✅ Tokenizer 验证通过：special tokens 已正确配置")


def test_regression_head():
    print("🧪 正在测试回归头...")

    # 1. 加载 tokenizer（不修改！）
    tokenizer = HFMathTokenizer()
    validate_tokenizer(tokenizer)

    # 2. 构造 config（IDs 从 tokenizer 读取，确保一致）
    config = MiniMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,      # 小模型加速测试
        num_hidden_layers=2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 3. 初始化模型
    model = MiniMindForRegression(config)
    model.eval()
    print(f"✅ 模型初始化成功 | Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}")

    # 4. 构造测试样本（使用 tokenizer 原生 encode）
    test_expr = "<s><compute>2+3</compute></s>"
    input_ids = tokenizer(test_expr, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"   Input: '{tokenizer.decode(input_ids[0], skip_special_tokens=False)}'")
    print(f"   Shape: {input_ids.shape} | IDs: {input_ids.tolist()}")

    # 5. 前向测试（含 profiling）
    with torch.no_grad():
        outputs = model(input_ids, profiling=True)

    # 6. 关键断言
    assert "prediction" in outputs
    pred = outputs["prediction"]
    assert pred.shape == (1,), f"Prediction shape mismatch: {pred.shape}"
    assert not torch.isnan(pred).any(), "Prediction is NaN!"
    assert not torch.isinf(pred).any(), "Prediction is Inf!"

    # 7. Profiling 检查
    assert "profiling_logs" in outputs, "Missing profiling_logs"
    logs = outputs["profiling_logs"]
    expected_events = config.num_hidden_layers * 4  # attn_in, out, mlp_in, out
    assert len(logs) >= expected_events, f"Too few logs: {len(logs)} < {expected_events}"

    print(f"✅ 预测值: {pred.item():.3f}")
    print(f"✅ Profiling logs: {len(logs)} events collected")
    print("🎉 回归头测试通过！")


if __name__ == "__main__":
    test_regression_head()
# # # 加载最终模型（推荐用于推理）
# # from models.model_minimind import MiniMindForCausalLM
# # from my_tokenizers.hf_math_tokenizer import HFMathTokenizer

# # model = MiniMindForCausalLM.from_pretrained("training/2025.10.30/minimind-math-h512-l8")
# # tokenizer = HFMathTokenizer.from_pretrained("training/2025.10.30/minimind-math-h512-l8")
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# from models.model_minimind import MiniMindForCausalLM
# from my_tokenizers.hf_math_tokenizer import HFMathTokenizer

# # === 配置路径 ===
# # 请根据你的实际保存路径修改这里
# SAVE_DIR = "training/2025.10.30"  # 你的 args.out_dir
# MODEL_NAME = "minimind-math-h512-l8"  # 根据你的配置调整，比如 h512-l8 或 h512-l8-moe

# MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME)
# TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer")  # 或直接用 MODEL_PATH（如果你用了最终完整模型）

# def test_load():
#     print("🔍 正在尝试加载模型和分词器...")

#     # 1. 加载分词器
#     try:
#         tokenizer = HFMathTokenizer.from_pretrained(TOKENIZER_PATH)
#         print(f"✅ 分词器加载成功！vocab_size = {tokenizer.vocab_size}")
#     except Exception as e:
#         print(f"❌ 分词器加载失败: {e}")
#         return

#     # 2. 加载模型
#     try:
#         model = MiniMindForCausalLM.from_pretrained(MODEL_PATH)
#         print(f"✅ 模型加载成功！模型类型: {type(model).__name__}")
#         print(f"   模型 vocab_size: {model.config.vocab_size}")
#         print(f"   模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
#     except Exception as e:
#         print(f"❌ 模型加载失败: {e}")
#         return

#     # 3. 验证一致性
#     if tokenizer.vocab_size != model.config.vocab_size:
#         print("⚠️ 警告：tokenizer 与 model 的 vocab_size 不一致！")
#         print(f"   Tokenizer: {tokenizer.vocab_size}, Model: {model.config.vocab_size}")
#     else:
#         print("✅ Tokenizer 与 Model vocab_size 一致")

#     # 4. 简单推理测试
#     test_text = "12+34=46"
#     try:
#         inputs = tokenizer(test_text, return_tensors="pt", add_special_tokens=True)
#         print(f"✅ 编码测试: '{test_text}' → {inputs.input_ids.tolist()}")
        
#         model.eval()
#         with torch.no_grad():
#             outputs = model(**inputs)
#         print(f"✅ 前向传播成功！logits shape: {outputs.logits.shape}")
        
#         # 解码预测
#         pred_id = outputs.logits.argmax(dim=-1)[0, -1].item()
#         pred_token = tokenizer.decode([pred_id])
#         print(f"✅ 预测下一个 token: '{pred_token}'")
        
#     except Exception as e:
#         print(f"❌ 推理测试失败: {e}")
#         return

#     print("\n🎉 所有测试通过！模型和分词器可以正常使用。")

# if __name__ == "__main__":
#     import torch
#     test_load()