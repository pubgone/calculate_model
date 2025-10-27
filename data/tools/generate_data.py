import json
import sys
"""
一位数运算
"""
def generate_all_one_digit_additions():
    """
    穷举所有一位数加法组合(0-9)，按顺序生成
    :return: 包含所有加法运算的列表
    """
    math_data = []
    
    # 按顺序穷举所有组合：a从0到9，b从0到9
    for a in range(10):  # 0-9
        for b in range(10):  # 0-9
            result = a + b
            # 构造符合格式的字典
            entry = {
                "context": f"\n\nQ: What is {a} + {b}?\n\nA:",
                "completion": f" {result}"
            }
            math_data.append(entry)
    
    return math_data

def generate_all_one_digit_subtraction():
    """
    穷举所有一位数加法组合(0-9)，按顺序生成
    :return: 包含所有加法运算的列表
    """
    math_data = []
    
    # 按顺序穷举所有组合：a从0到9，b从0到9
    for a in range(10):  # 0-9
        for b in range(10):  # 0-9
            result = a - b
            # 构造符合格式的字典
            entry = {
                "context": f"\n\nQ: What is {a} - {b}?\n\nA:",
                "completion": f" {result}"
            }
            math_data.append(entry)
    
    return math_data
"""
2位数运算
"""
def generate_all_two_digit_additions():
    """
    穷举所有两位数加法组合(10-99)，按顺序生成
    :return: 包含所有加法运算的列表
    """
    additions = []
    # 两位数范围：10-99
    for a in range(10, 100):
        for b in range(10, 100):
            result = a + b
            entry = {
                "context": f"\n\nQ: What is {a} + {b}?\n\nA:",
                "completion": f" {result}"
            }
            additions.append(entry)
    return additions

def generate_all_two_digit_subtractions():
    """
    穷举所有两位数减法组合(10-99)，按顺序生成
    :return: 包含所有减法运算的列表
    """
    subtractions = []
    # 两位数范围：10-99
    for a in range(10, 100):
        for b in range(10, 100):
            result = a - b
            entry = {
                "context": f"\n\nQ: What is {a} - {b}?\n\nA:",
                "completion": f" {result}"
            }
            subtractions.append(entry)
    return subtractions
"""
3位数运算
"""
def generate_all_three_digit_additions():
    """
    穷举所有三位数加法组合(100-999)，按顺序生成
    :return: 包含所有加法运算的列表
    """
    additions = []
    # 三位数范围：100-999
    for a in range(100, 1000):
        for b in range(100, 1000):
            result = a + b
            entry = {
                "context": f"\n\nQ: What is {a} plus {b}?\n\nA:",
                "completion": f" {result}"
            }
            additions.append(entry)
    return additions

def generate_all_three_digit_subtractions():
    """
    穷举所有三位数减法组合(100-999)，按顺序生成
    :return: 包含所有减法运算的列表
    """
    subtractions = []
    # 三位数范围：100-999
    for a in range(100, 1000):
        for b in range(100, 1000):
            result = a - b
            entry = {
                "context": f"\n\nQ: What is {a} minus {b}?\n\nA:",
                "completion": f" {result}"
            }
            subtractions.append(entry)
    return subtractions
"""
4位数运算
"""
def generate_four_digit_additions_batch(batch_size=100):
    """分批生成四位数加法并写入文件，避免内存溢出"""
    with open("./math_data/addition/four_digit_additions.json", 'w', encoding='utf-8') as f:
        f.write("[")  # 开始JSON数组
        first_entry = True
        
        for a in range(1000, 10000):
            if a % 100 == 0:
                print(f"生成加法: {a}/9999", end='\r')
                sys.stdout.flush()
                
            for b in range(1000, 10000):
                result = a + b
                entry = {
                    "context": f"\n\nQ: What is {a} plus {b}?\n\nA:",
                    "completion": f" {result}"
                }
                
                if not first_entry:
                    f.write(",")
                else:
                    first_entry = False
                    
                # 写入单个条目
                json.dump(entry, f, ensure_ascii=False)
                
        f.write("]")  # 结束JSON数组
    print("\n加法生成完成")

def generate_four_digit_subtractions_batch(batch_size=100):
    """分批生成四位数减法并写入文件，避免内存溢出"""
    with open("./math_data/subtraction/four_digit_subtractions.json", 'w', encoding='utf-8') as f:
        f.write("[")  # 开始JSON数组
        first_entry = True
        
        for a in range(1000, 10000):
            if a % 100 == 0:
                print(f"生成减法: {a}/9999", end='\r')
                sys.stdout.flush()
                
            for b in range(1000, 10000):
                result = a - b
                entry = {
                    "context": f"\n\nQ: What is {a} minus {b}?\n\nA:",
                    "completion": f" {result}"
                }
                
                if not first_entry:
                    f.write(",")
                else:
                    first_entry = False
                    
                # 写入单个条目
                json.dump(entry, f, ensure_ascii=False)
                
        f.write("]")  # 结束JSON数组
    print("\n减法生成完成")

def save_to_json(data, filename):
    """将数据保存为JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 执行生成和保存操作
if __name__ == "__main__":
    generate_four_digit_additions_batch()
    # 生成所有减法组合
    generate_four_digit_subtractions_batch()

    # all_additions = generate_all_one_digit_additions()
    # save_to_json(all_additions, "./math_data/addition/all_one_digit_additions.json")
    # print(f"已生成所有一位数加法组合，共 {len(all_additions)} 条数据")
    # print("结果保存至 all_one_digit_additions.json")
    