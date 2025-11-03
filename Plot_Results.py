
import matplotlib.pyplot as plt
import numpy as np

# Example-Data

a = [0, 0, 0.04, 0.05, 0, 0 ]  # Type I Error
b = [0.49, 0.07,  0.82, 0.23, 0.81, 0.39]  # Type II Error


n = len(a)

group_spacing = 1.5
bar_width = 0.35

indices = np.arange(n) * group_spacing

fig, ax = plt.subplots(figsize=(8, 4))

bars_a = ax.bar(indices - bar_width/2, a, width=bar_width, label='Type I Error', color='skyblue')
bars_b = ax.bar(indices + bar_width/2, b, width=bar_width, label='Type II Error', color='blue')

labels = ['T = 1', 'T = 0.95','T = 0.9','T = 0.85','T = 0.8']

ax.set_xticks(indices)  
ax.set_xticklabels(labels, rotation=90)  

ax.legend()
ax.set_ylabel('Error')
plt.tight_layout()
plt.show()
