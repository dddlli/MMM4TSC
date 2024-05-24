import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置全局字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14  # 你可以根据需要调整字体大小

# 读取Excel文件
df = pd.read_excel('result.xlsx')

# 将数据集作为index，模型和准确率作为列
df = df.set_index('dataset')

# 计算每个模型的平均准确率
accuracy_means = df.mean(axis=0)

# 创建柱状图
plt.figure(figsize=(6, 4))

palette = {
    'FCN': '#bbf8db',  # 橙色
    'ResNet': '#9cf5cb',  # 蓝色
    'Inception': '#7df2bb',  # 绿色
    'InceptionTime': '#5eefaa',  # 黄色
    'LITE': '#3fec9a',  # 深蓝色
    'LITETime': '#21e88a',  # 红橙色
    'ROCKET': '#15d279',  # 紫色
    'MMM4TSC': '#12b367'  # 黑色，MMM4TSC模型
}

# 绘制柱状图
bar = sns.barplot(x=accuracy_means.index, y=accuracy_means.values, palette=palette, width=0.5)

# 在柱子上标出平均准确率
for idx, mean_acc in accuracy_means.items():
    bar.text(idx, mean_acc, f'{mean_acc:.3f}', ha='center', va='bottom', fontfamily='Times New Roman')

# 设置图形标题和坐标轴标签
plt.xlabel('Model', fontsize=16, fontweight='bold')
plt.ylabel('Average Accuracy', fontsize=16, fontweight='bold')

plt.ylim(0.75)

# 增加图形的网格线，使图形更加清晰
sns.despine(left=True, bottom=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.xticks(rotation=45, ha='right')

# 显示图形
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.savefig('all_acc.pdf', bbox_inches='tight', dpi=300, format='pdf')
plt.show()
