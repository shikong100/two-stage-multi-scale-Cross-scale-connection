import pandas as pd

# 示例预测结果，假设预测结果为一个包含字典的列表
predictions = [{
    "Filename",
    "ND"
}
]

# 将预测结果转换为DataFrame
df = pd.DataFrame(predictions)

# 定义CSV文件路径
csv_file = 'predictions.csv'

# 将DataFrame写入CSV文件
df.to_csv(csv_file, index=False, header=False)

a = ['1.png', '2.png']
b = [0, 1]
p = ({
    a,
    b
})
d = pd.DataFrame(p)
d.to_csv(csv_file, mode='a', header=False)
print("Predictions have been written to CSV file:", csv_file)
