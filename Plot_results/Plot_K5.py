import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

path = "results_0_swap_K5_uniform.csv"

df_new = pd.read_csv(path)

path = "results_1_swap_K5_uniform.csv"


df = pd.read_csv(path)
df['result'] = df['result'].apply(lambda x: np.array(ast.literal_eval(x.replace('.', '.'))))
df['result'] = df['result'].apply(lambda x: np.sum(x)/500)

df_combined = pd.concat([df_new, df], ignore_index=True)
df_combined = df_combined[df_combined["M"].isin([50, 100, 200, 500, 1000])]
df_combined = df_combined[df_combined["T"].isin([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])]

#%%
df = df_combined

df = df[df["epsilon"] == 0.65]
df = df
# Get unique values
Ms = sorted(df['M'].unique())
epsilons = sorted(df['epsilon'].unique())
Ts = sorted(df['T'].unique())

# Define colors for M
# Define a color for each value
color_map = {
    50:   "blue",  
    100:  "#33a02c",  
    200:  "#a6cee3",  
    500:  "#b2df8a",  
    1000: "#08519c"   
}


# Define markers for epsilon (expand if more than 3 epsilons)
marker_map = {eps: m for eps, m in zip(epsilons, ['p', '*'])}

# Aggregate sum(result) per (M, epsilon, T)
agg = df.groupby(['M', 'epsilon', 'T'], as_index=False)['result'].sum()
agg = agg.sort_values('T')



golden_ratio = (1 + 5**0.5) / 2  # ≈ 1.618

width = 12
height = width / golden_ratio
plt.figure(figsize=(width, height))

for Mval in Ms:
    subset_M = agg[agg['M'] == Mval]
    print(subset_M)
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
                 label=f"M={Mval}")

plt.xlabel('T', fontsize = 20)
plt.ylabel('Proportion of reject', fontsize = 20)
plt.grid(True)
plt.legend(fontsize = 16)
plt.tight_layout()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(os.path.join(output_folder, "combined_plot.png"))
plt.show()
