import os
import pandas as pd

# 设置文件夹路径
folder_path = 'result/CNN'  # 请替换为你的CSV文件夹路径
output_file = 'pic/max_values.csv'  # 输出文件的名称

# 准备一个空的DataFrame来保存结果
results = pd.DataFrame(columns=['FileName', 'MaxValue'])

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # 找出第四列的最大值
        max_value = df.iloc[:, 3].max()

        # 创建一个新的DataFrame来保存当前文件的结果
        temp_df = pd.DataFrame({'FileName': [filename], 'MaxValue': [max_value]})

        # 使用concat来合并结果而不是append
        results = pd.concat([results, temp_df], ignore_index=True)

        # 将结果保存到新的CSV文件中
results.to_csv(output_file, index=False)