import csv
from py3dbp import Packer, Bin, Item

# 数据文件路径
dataset_path = './Dataset/Task3/task3.csv'

# 初始化 Packer 实例
packer = Packer()

# 定义容器尺寸并添加到 Packer
container_sizes = [
    (35, 23, 13), (37, 26, 13), (38, 26, 13),
    (40, 28, 16), (42, 30, 18), (42, 30, 40),
    (52, 40, 17), (54, 45, 36)
]

for idx, (length, width, height) in enumerate(container_sizes):
    max_weight = float('inf')  # 忽略重量限制
    packer.add_bin(Bin(f'bin{idx + 1}', length, width, height, max_weight))

# 读取 CSV 数据并添加物品
with open(dataset_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sta_code = row['sta_code']           # 订单编号
        sku_code = row['sku_code']           # 物品编号
        length = float(row['长(CM)'])        # 长度
        width = float(row['宽(CM)'])         # 宽度
        height = float(row['高(CM)'])        # 高度
        quantity = int(row['qty'])           # 数量

        # 根据物品数量添加多个相同的Item
        for _ in range(quantity):
            packer.add_item(Item(f"{sta_code}-{sku_code}", length, width, height))

# 执行装箱
packer.pack()

# 输出结果
for bin in packer.bins:
    print(f"Bin: {bin.name}")
    print("Fitted Items:")
    for item in bin.items:
        print(f"-> {item.name}")
    print("Unfitted Items:")
    for item in bin.unfitted_items:
        print(f"-> {item.name}\n")
