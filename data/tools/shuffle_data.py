import random
from pathlib import Path

# ==============================
# 配置区（请修改这三行）
# ==============================
CORPUS_ORIGINAL = r"D:\calculate_model\data\corpus\order\mixed_steps"        # 原始有序语料根目录
CORPUS_SHUFFLED = r"D:\calculate_model\data\corpus\random\mixed_steps"  # 新的乱序语料根目录
SHUFFLE_SEED = 42                                    # 随机种子（可复现）
# ==============================

def shuffle_and_save(input_path: Path, output_path: Path, seed: int):
    try:
        # 读取原始文件
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]

        if not lines:
            print(f"Skipping empty file: {input_path}")
            return

        # 使用独立随机状态打乱
        rng = random.Random(seed)
        rng.shuffle(lines)

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入新文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

        print(f"Shuffled: {input_path} → {output_path}")

    except Exception as e:
        print(f"Error: {input_path} - {e}")

def main():
    original_root = Path(CORPUS_ORIGINAL)
    shuffled_root = Path(CORPUS_SHUFFLED)

    # 查找所有 .txt 文件
    txt_files = list(original_root.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {original_root}")
        return

    print(f"Found {len(txt_files)} files. Shuffling with seed={SHUFFLE_SEED}...")
    for orig_file in txt_files:
        # 计算相对路径，构建新路径
        rel_path = orig_file.relative_to(original_root)
        shuffled_file = shuffled_root / rel_path
        shuffle_and_save(orig_file, shuffled_file, SHUFFLE_SEED)

    print("✅ All files shuffled and saved to new directory.")

if __name__ == "__main__":
    main()