import matplotlib.pyplot as plt

# Data
params = [9814, 96542, 264704, 420192, 504000]
models = ['LITE', 'MMM4TSC', 'FCN', 'Inception', 'ResNet']
accuracy = [0.8304, 0.8494, 0.7883, 0.8411, 0.8066]
colors = ['red', 'green', 'blue', 'yellow', 'purple']
bubble_sizes = [100, 300, 550, 700, 850]  # Arbitrary sizes for demonstration

# Plot
plt.figure(figsize=(10, 6))
for i in range(len(models)):
    plt.scatter(params[i], accuracy[i], s=bubble_sizes[i], c=colors[i], alpha=0.8, label=None)

ax = plt.gca()
bbox_to_anchor = (ax.get_position().x0 - 0.10, ax.get_position().y0 - 0.10)

# Legend with bubble sizes
legend_labels = [f'{models[i]} ({params[i]:,})' for i in range(len(models))]
bubble_sizes_legend = [250, 50, 100, 150, 200]  # Sizes for legend
for size, label, color in zip(bubble_sizes, legend_labels, colors):
    plt.scatter([], [], s=size, c=color, label=label, alpha=0.6)

plt.xlabel('Parameters', fontweight='bold', fontsize='large')
plt.ylabel('Accuracy', fontsize='large', fontweight='bold')
plt.title('Model Parameters vs. Accuracy', fontweight='bold', fontsize='large')
plt.ylim(0.75, 0.86)
# plt.legend(title='Models (Parameters)', loc='lower left', fontsize='small', ncol=5,
#            bbox_to_anchor=bbox_to_anchor, frameon=False)

plt.legend(loc='lower left', fontsize='small', ncol=5, bbox_to_anchor=bbox_to_anchor)

plt.grid(True)
plt.tight_layout()

plt.savefig('purppl.svg', bbox_inches='tight', dpi=300)
plt.show()