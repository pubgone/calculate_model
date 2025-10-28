import os
import glob
from typing import Dict
from model_ultils.pretrain_dataset import PretrainDataset  # 确保路径正确

def create_dataset_per_file(
    data_dir: str,
    tokenizer,
    max_length: int = 512
) -> Dict[str, PretrainDataset]:
    """
    为 data_dir 下的每个 .txt 文件创建一个独立的 PretrainDataset。
    
    参数:
        data_dir (str): 包含多个 .txt 文件的目录路径，例如 "data/train/"
        tokenizer: 你的 MathTokenizer 实例
        max_length (int): 每个样本的最大 token 长度（建议 64~128）
    
    返回:
        dict: 键为文件名（如 "add_2digit.txt"），值为对应的 PretrainDataset 对象
    """
    datasets = {}
    txt_paths = glob.glob(os.path.join(data_dir, "*.txt"))
    
    if not txt_paths:
        raise ValueError(f"No .txt files found in {data_dir}")
    
    for file_path in sorted(txt_paths):  # 排序保证顺序一致
        filename = os.path.basename(file_path)  # 如 "add_2digit.txt"
        print(f"Creating dataset for: {filename}")
        dataset = PretrainDataset(
            data_path=file_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        datasets[filename] = dataset
    
    return datasets
# ##################################使用示例######################################
#
from my_tokenizers.math_tokenizer import MathTokenizer
# 1️⃣ 初始化 tokenizer（必须）
tokenizer = MathTokenizer()
# 2️⃣ 设置数据目录（必须）
data_dir = r"data\corpus\random\addition"  # ← 填你的实际路径，例如 "data/train/"
# 3️⃣ 设置 max_length（可选，默认 512，但数学表达式建议更小）
max_length = 64  # ← 推荐值：64 或 128
# 4️⃣ 调用函数
datasets = create_dataset_per_file(
    data_dir=data_dir,      # ← 必填：目录路径
    tokenizer=tokenizer,    # ← 必填：你的 MathTokenizer 实例
    max_length=max_length   # ← 可选：默认 512，建议设为 64
)
from torch.utils.data import DataLoader

# 只训练加法（2位数）
add_1_digit_DataLoader = DataLoader(
    datasets["1_digit_additions.txt"],
    batch_size=32,
    shuffle=True
)

