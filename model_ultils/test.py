# test.py
from model_ultils.pretrain_dataset import PretrainDataset
from my_tokenizers.math_tokenizer import MathTokenizer 

# 初始化 tokenizer
tokenizer = MathTokenizer()

# 创建数据集
dataset = PretrainDataset(
    data_path=r"D:\calculate_model\data\corpus\order\addition\1_digit_additions.txt",
    tokenizer=tokenizer,
    max_length=64  # 数学表达式通常很短
)

# 测试第一条数据
X, Y, mask = dataset[0]

print("原始文本:", dataset.samples[0])
print("解码 X:", tokenizer.decode(X.tolist(), skip_special_tokens=False))
print("解码 Y:", tokenizer.decode(Y.tolist(), skip_special_tokens=False))
print("Loss mask sum:", mask.sum().item())  # 有效 token 数量