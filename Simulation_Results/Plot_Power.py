import pandas as pd
import ast
path = "results_1_swap_K2_1111_50_100.csv"
path1 = "results_1_swap_K2_1511_200.csv"
df = pd.read_csv(path)
df1 = pd.read_csv(path1)

print(df.head())

df['result'] = df['result'].apply(lambda x: np.array(ast.literal_eval(x.replace('.', '.'))))
df1['result'] = df1['result'].apply(lambda x: np.array(ast.literal_eval(x.replace('.', '.'))))

df_combined = pd.concat([df1, df], ignore_index=True)
#df_combined["result"] = df_combined["result"].apply(lambda x: np.array(ast.literal_eval(x.replace('.', '.'))))
df_combined['result'] = df_combined['result'].apply(lambda x: np.sum(x))

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assume df['result'] is already numeric (sum of arrays)
output_folder = "plots_combined"
os.makedirs(output_folder, exist_ok=True)

# Get unique values
Ms = sorted(df_combined['M'].unique())
epsilons = sorted(df_combined['epsilon'].unique())
Ts = sorted(df_combined['T'].unique())

# Define colors for M
color_map = {50: "orange", 100: 'blue', 200: 'green'}

# Define markers for epsilon (expand if more than 3 epsilons)
marker_map = {eps: m for eps, m in zip(epsilons, ['s', 'D', 'v', '*'])}

# Aggregate sum(result) per (M, epsilon, T)
agg = df_combined.groupby(['M', 'epsilon', 'T'], as_index=False)['result'].sum()
agg = agg.sort_values('T')

plt.figure(figsize=(10, 6))

for Mval in Ms:
    subset_M = agg[agg['M'] == Mval]
    for eps in epsilons:
        subset = subset_M[subset_M['epsilon'] == eps].set_index('T').reindex(Ts)['result'].values
        mask = ~np.isnan(subset)
        if not mask.any():
            continue
        plt.plot(np.array(Ts)[mask], subset[mask],
                 marker=marker_map[eps],
                 linestyle='-',
                 color=color_map.get(Mval, 'gray'),
                 linewidth=2.5 if Mval in (100,200) else 1.2,
                 alpha=1.0 if Mval in (100,200) else 0.6,
                 label=f"M={Mval}, epsilon={eps}")

plt.xlabel('T', fontsize = 16)
plt.ylabel('Power', fontsize = 16)
plt.grid(True)
#plt.legend()
plt.tight_layout()
# Tick-Nummern vergrößern
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Save the combined plot
plt.savefig(os.path.join(output_folder, "combined_plot.png"))
plt.show()

