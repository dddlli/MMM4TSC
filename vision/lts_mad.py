import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置全局字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14

# 读取Excel文件
df = pd.read_csv('lts.csv')

# 计算每个模型的准确率方差
accuracy_vars = df.var(axis=0)

# 创建柱状图
plt.figure(figsize=(6, 4))

palette = {
    'FCN': '#c3deff',  # 橙色
    'ResNet': '#b2d5ff',  # 蓝色
    'Inception': '#90c2ff',  # 绿色
    'InceptionTime': '#6fb0fe',  # 黄色
    'LITE': '#4d9dfe',  # 深蓝色
    'LITETime': '#2b8afe',  # 红橙色
    'ROCKET': '#016ff5',  # 紫色
    'MMM4TSC': '#0168e4'  # 黑色，MMM4TSC模型
}

# 绘制柱状图
bar = sns.barplot(x=accuracy_vars.index, y=accuracy_vars.values, palette=palette, width=0.5)

# 在柱子上标出平均方差
for idx, var in accuracy_vars.items():
    bar.text(idx, var, f'{var:.3f}', ha='center', va='bottom', fontfamily='Times New Roman')

# 设置图形标题和坐标轴标签
plt.xlabel('Model', fontsize=16, fontweight='bold')
plt.ylabel('Mean Square Deviation', fontsize=16, fontweight='bold')

# 增加图形的网格线，使图形更加清晰
sns.despine(left=True, bottom=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.xticks(rotation=45, ha='right')

# 显示图形
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.savefig('lts_mad.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.show()
