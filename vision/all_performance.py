import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置全局字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14

# 读取Excel文件
df = pd.read_excel('result.xlsx')

# 假设第一列是数据集名称，我们将其保留为索引
df.set_index('dataset', inplace=True)

# 将模型名称和准确率转换为适当的数据类型
models = df.columns
df[models] = df[models].astype(float)

palette = {
    'FCN': '#d4e7ff',       # 橙色
    'ResNet': '#b2d5ff',    # 蓝色
    'Inception': '#90c2ff', # 绿色
    'InceptionTime': '#6fb0fe', # 黄色
    'LITE': '#4d9dfe',     # 深蓝色
    'LITETime': '#2b8afe',  # 红橙色
    'ROCKET': '#016ff5',    # 紫色
    'MMM4TSC': '#0168e4'    # 黑色，MMM4TSC模型
}

# 绘制箱线图，比较所有模型的性能
plt.figure(figsize=(10, 7))  # 设置图形大小
sns.boxplot(data=df, palette=palette)  # 绘制箱线图

# 突出显示MMM4TSC模型的箱线图
mmm4tsc_accuracies = df['MMM4TSC']
sns.swarmplot(data=mmm4tsc_accuracies, color='#0168e4')  # 绘制MMM4TSC模型的散点图

plt.xlabel('Model', fontsize=16, fontweight='bold')  # 设置x轴标签
plt.ylabel('Accuracy', fontsize=16, fontweight='bold')  # 设置y轴标签
# plt.xticks(rotation=90)  # 旋转x轴标签以便阅读
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.savefig('all_performance.svg', bbox_inches='tight', dpi=300, format='svg')
# 显示图形
plt.show()