# math_tokenizer_with_functions.py
import json
from typing import List, Dict, Optional

class MathTokenizer:
    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        special_tokens: Optional[List[str]] = None,
        operators: Optional[List[str]] = None,
        functions: Optional[List[str]] = None,
        constants: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
    ):
        # 默认特殊 token
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        # 默认运算符
        if operators is None:
            operators = ["+", "-", "*", "/", "^", "=", "(", ")", ".", ",", " "]
        # 默认初等函数
        if functions is None:
            functions = ["sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs"]
        # 默认常量
        if constants is None:
            constants = ["pi", "e"]
        # 默认变量
        if variables is None:
            variables = ["x", "y", "z", "a", "b", "c"]

        # 构建词表：**multi-char tokens 放前面**（最长匹配优先）
        vocab_list = special_tokens + functions + constants + variables + operators
        digits = [str(i) for i in range(10)]
        vocab_list += digits

        # 去重并保持顺序
        seen = set()
        vocab_clean = []
        for t in vocab_list:
            if t not in seen:
                vocab_clean.append(t)
                seen.add(t)

        self.vocab = vocab_clean
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}

        # 特殊 token ID
        self.pad_id = self.token_to_id["<pad>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.bos_id = self.token_to_id.get("<s>", None)
        self.eos_id = self.token_to_id.get("</s>", None)

        # 为最长匹配准备：按长度降序排序
        self.sorted_tokens = sorted(self.token_to_id.keys(), key=lambda x: -len(x))

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = []
        if add_special_tokens and self.bos_id is not None:
            ids.append(self.bos_id)

        i = 0
        while i < len(text):
            matched = False
            for token in self.sorted_tokens:
                if text.startswith(token, i):
                    ids.append(self.token_to_id[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                # 未知字符 → <unk>
                ids.append(self.unk_id)
                i += 1

        if add_special_tokens and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.pad_id, self.unk_id}
        if self.bos_id is not None:
            special_ids.add(self.bos_id)
        if self.eos_id is not None:
            special_ids.add(self.eos_id)

        tokens = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            token = self.id_to_token.get(i, "<unk>")
            if not (skip_special_tokens and token in ["<pad>", "<unk>", "<s>", "</s>"]):
                tokens.append(token)
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save_vocab(self, file_path: str):
        # 保存为 {token: id} 格式，类似 Qwen 风格
        token_to_id_dict = {token: idx for token, idx in self.token_to_id.items()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(token_to_id_dict, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            token_to_id_dict = json.load(f)
        vocab = sorted(token_to_id_dict, key=token_to_id_dict.get)  # 按 ID 排序还原列表
        return cls(vocab=vocab)

    def print_vocab_with_ids(self):
        print("Vocabulary (token: id):")
        for token, idx in self.token_to_id.items():
            print(f'"{token}": {idx}')

    def __repr__(self):
        return f"MathTokenizer(vocab_size={self.vocab_size})"


# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    tokenizer = MathTokenizer()

    # 测试复杂表达式
    test_expr = "sin(pi * x) + log(e) = 1\nsqrt(16) = 4, abs(-5) = 5"

    print("原始表达式:")
    print(repr(test_expr))
    print("\n编码 IDs:")
    ids = tokenizer.encode(test_expr)
    print(ids)
    print("\n解码结果:")
    decoded = tokenizer.decode(ids)
    print(repr(decoded))

    print(f"\n词表大小: {tokenizer.vocab_size}")
    tokenizer.print_vocab_with_ids()

    # 保存词表（格式如 "*": 9）
    tokenizer.save_vocab("math_tokenizer_vocab.json")
    print("\n词表已保存为 'math_tokenizer_vocab.json'")

    # 从文件加载
    loaded = MathTokenizer.from_file("math_tokenizer_vocab.json")
    print(f"重新加载的 tokenizer 词表大小: {loaded.vocab_size}")