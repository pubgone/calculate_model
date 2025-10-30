# # 加载最终模型（推荐用于推理）
# from models.model_minimind import MiniMindForCausalLM
# from my_tokenizers.hf_math_tokenizer import HFMathTokenizer

# model = MiniMindForCausalLM.from_pretrained("training/2025.10.30/minimind-math-h512-l8")
# tokenizer = HFMathTokenizer.from_pretrained("training/2025.10.30/minimind-math-h512-l8")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from models.model_minimind import MiniMindForCausalLM
from my_tokenizers.hf_math_tokenizer import HFMathTokenizer

# === 配置路径 ===
# 请根据你的实际保存路径修改这里
SAVE_DIR = "training/2025.10.30"  # 你的 args.out_dir
MODEL_NAME = "minimind-math-h512-l8"  # 根据你的配置调整，比如 h512-l8 或 h512-l8-moe

MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME)
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer")  # 或直接用 MODEL_PATH（如果你用了最终完整模型）

def test_load():
    print("🔍 正在尝试加载模型和分词器...")

    # 1. 加载分词器
    try:
        tokenizer = HFMathTokenizer.from_pretrained(TOKENIZER_PATH)
        print(f"✅ 分词器加载成功！vocab_size = {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        return

    # 2. 加载模型
    try:
        model = MiniMindForCausalLM.from_pretrained(MODEL_PATH)
        print(f"✅ 模型加载成功！模型类型: {type(model).__name__}")
        print(f"   模型 vocab_size: {model.config.vocab_size}")
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 验证一致性
    if tokenizer.vocab_size != model.config.vocab_size:
        print("⚠️ 警告：tokenizer 与 model 的 vocab_size 不一致！")
        print(f"   Tokenizer: {tokenizer.vocab_size}, Model: {model.config.vocab_size}")
    else:
        print("✅ Tokenizer 与 Model vocab_size 一致")

    # 4. 简单推理测试
    test_text = "12+34=46"
    try:
        inputs = tokenizer(test_text, return_tensors="pt", add_special_tokens=True)
        print(f"✅ 编码测试: '{test_text}' → {inputs.input_ids.tolist()}")
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✅ 前向传播成功！logits shape: {outputs.logits.shape}")
        
        # 解码预测
        pred_id = outputs.logits.argmax(dim=-1)[0, -1].item()
        pred_token = tokenizer.decode([pred_id])
        print(f"✅ 预测下一个 token: '{pred_token}'")
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return

    print("\n🎉 所有测试通过！模型和分词器可以正常使用。")

if __name__ == "__main__":
    import torch
    test_load()