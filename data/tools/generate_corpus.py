import json
import os
from pathlib import Path

# ==============================
# 在这里指定你的输入和输出根路径
# ==============================
INPUT_ROOT = r"D:\calculate_model\data\raw_data\mixed_steps"      # ←←← 修改这一行
OUTPUT_ROOT = r"D:\calculate_model\data\corpus\mixed_steps"       # ←←← 修改这一行
# ==============================

def clean_question(context: str) -> str:
    """从 context 中提取不含 'Q:', '?', 'A:' 的纯表达式左侧"""
    text = context

    # 移除开头的 "\n\nQ: "（即使有额外换行也兼容）
    if text.startswith("\n\nQ: "):
        text = text[5:]
    elif text.lstrip().startswith("Q: "):
        # 兜底：如果前面有空格或换行，但本质是 Q:
        text = text.lstrip()[3:]

    # 移除问号及之后的内容（包括 "\n\nA:" 等）
    if "?" in text:
        text = text.split("?", 1)[0]

    # 清理末尾空格和可能残留的等号
    text = text.rstrip(" =")

    return text.strip()
def process_single_file(input_path: Path, output_path: Path):
    """处理单个 JSONL 文件，输出为一行一表达式的 TXT 文件"""
    samples = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"JSON error in {input_path}:{line_num} - {e}")
                    continue

                # 修复：补全条件判断
                if "context" not in data or "completion" not in data:
                    print(f"Missing fields in {input_path}:{line_num}")
                    continue

                try:
                    expr_left = clean_question(data["context"])
                    if not expr_left:
                        continue
                    completion = str(data["completion"]).lstrip()
                    full_expr = f"{expr_left} = {completion}"
                    samples.append(full_expr)
                except Exception as e:
                    print(f"Processing error in {input_path}:{line_num} - {e}")
                    continue

        # 创建输出目录并写入文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for s in samples:
                out_f.write(s + '\n')

    except Exception as e:
        print(f"Failed to process file {input_path}: {e}")

def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)

    # 查找所有 .jsonl 文件（递归）
    jsonl_files = list(input_root.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found under {input_root}")
        return

    print(f"Found {len(jsonl_files)} JSONL files. Processing...")
    for input_file in jsonl_files:
        # 保持目录结构：将 .jsonl 替换为 .txt
        rel_path = input_file.relative_to(input_root)
        output_file = output_root / rel_path.with_suffix('.txt')
        print(f"→ {input_file} → {output_file}")
        process_single_file(input_file, output_file)

    print("✅ All files processed.")

if __name__ == "__main__":
    main()