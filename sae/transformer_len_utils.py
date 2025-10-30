# load_minimind_hooked.py
import os
import sys
import torch
sys.path.append(os.path.abspath('.'))

from transformer_lens import HookedTransformer, HookedTransformerConfig
from my_tokenizers.hf_math_tokenizer import HFMathTokenizer
from models.model_minimind import MiniMindForCausalLM


def convert_minimind_weights_to_hooked(hf_state_dict, cfg, num_key_value_heads):
    new_sd = {}
    n_heads = cfg.n_heads
    n_kv_heads = num_key_value_heads
    d_model = cfg.d_model
    d_head = cfg.d_head
    n_rep = n_heads // n_kv_heads

    new_sd["embed.W_E"] = hf_state_dict["model.embed_tokens.weight"]
    new_sd["ln_final.w"] = hf_state_dict["model.norm.weight"]
    new_sd["unembed.W_U"] = hf_state_dict["lm_head.weight"].T

    for i in range(cfg.n_layers):
        hf_p = f"model.layers.{i}."
        tl_p = f"blocks.{i}."

        # Q
        q = hf_state_dict[hf_p + "self_attn.q_proj.weight"]
        q = q.T.reshape(n_heads, d_head, d_model).permute(0, 2, 1)
        new_sd[tl_p + "attn.W_Q"] = q

        # K
        k = hf_state_dict[hf_p + "self_attn.k_proj.weight"]
        k = k.T.reshape(n_kv_heads, d_head, d_model)
        k = k.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(n_heads, d_head, d_model)
        k = k.permute(0, 2, 1)
        new_sd[tl_p + "attn.W_K"] = k

        # V
        v = hf_state_dict[hf_p + "self_attn.v_proj.weight"]
        v = v.T.reshape(n_kv_heads, d_head, d_model)
        v = v.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(n_heads, d_head, d_model)
        v = v.permute(0, 2, 1)
        new_sd[tl_p + "attn.W_V"] = v

        # O
        o = hf_state_dict[hf_p + "self_attn.o_proj.weight"]
        o = o.T.reshape(n_heads, d_head, d_model)
        new_sd[tl_p + "attn.W_O"] = o

        new_sd[tl_p + "ln1.w"] = hf_state_dict[hf_p + "input_layernorm.weight"]
        new_sd[tl_p + "ln2.w"] = hf_state_dict[hf_p + "post_attention_layernorm.weight"]

        # SwiGLU
        new_sd[tl_p + "mlp.W_gate"] = hf_state_dict[hf_p + "mlp.gate_proj.weight"].T
        new_sd[tl_p + "mlp.W_up"] = hf_state_dict[hf_p + "mlp.up_proj.weight"].T
        new_sd[tl_p + "mlp.W_out"] = hf_state_dict[hf_p + "mlp.down_proj.weight"].T

    return new_sd


def load_minimind_as_hooked(model_path: str, device="cpu"):
    # 直接加载 tokenizer（不经过 AutoTokenizer）
    tokenizer = HFMathTokenizer.from_pretrained(model_path)
    hf_model = MiniMindForCausalLM.from_pretrained(model_path)
    hf_config = hf_model.config
    hf_state_dict = hf_model.state_dict()

    cfg = HookedTransformerConfig(
        d_model=hf_config.hidden_size,
        d_head=hf_config.hidden_size // hf_config.num_attention_heads,
        n_heads=hf_config.num_attention_heads,
        n_layers=hf_config.num_hidden_layers,
        n_ctx=hf_config.max_position_embeddings,
        d_vocab=hf_config.vocab_size,
        d_mlp=hf_config.intermediate_size,
        act_fn="silu",  # 注意：新版 transformer-lens 用 "silu" 即可，内部自动处理 gate+up
        normalization_type="RMS",
        positional_embedding_type="rotary",
        device=device,
        use_attn_result=False,
    )

    converted_sd = convert_minimind_weights_to_hooked(
        hf_state_dict, cfg, num_key_value_heads=hf_config.num_key_value_heads
    )

    hooked_model = HookedTransformer(cfg)
    missing, unexpected = hooked_model.load_state_dict(converted_sd, strict=False)

    # === 关键：绕过 set_tokenizer，手动绑定 tokenizer 和 to_tokens ===
    hooked_model.tokenizer = tokenizer

    # 重写 to_tokens 方法
    def to_tokens(text, prepend_bos=True, move_to_device=True):
        # 使用你的 tokenizer 编码
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True  # 确保包含 BOS/EOS
        )
        tokens = encoded.input_ids
        if move_to_device:
            tokens = tokens.to(device)
        return tokens

    hooked_model.to_tokens = to_tokens
    # =================================================================

    return hooked_model


# ========================
# Test
# ========================
if __name__ == "__main__":
    MODEL_PATH = "training/2025.10.30/minimind-math-h512-l8"
    DEVICE = "cpu"

    print("🔍 Loading MiniMind as HookedTransformer...")
    model = load_minimind_as_hooked(MODEL_PATH, device=DEVICE)

    print("✅ Success!")
    print(f"  - d_model: {model.cfg.d_model}")
    print(f"  - n_heads: {model.cfg.n_heads}")
    print(f"  - d_mlp: {model.cfg.d_mlp}")
    print(f"  - act_fn: {model.cfg.act_fn}")

    # Test hooking
    def hook_fn(act, hook):
        print(f"  🔥 {hook.name}: {act.shape}")
        return act

    tokens = model.to_tokens("12+34=")
    print("Tokens:", tokens.tolist())
    logits = model.run_with_hooks(tokens, fwd_hooks=[("blocks.0.hook_resid_post", hook_fn)])
    print("\n🎉 Hooking works!")