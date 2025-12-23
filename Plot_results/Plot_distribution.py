import pandas as pd
import ast
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
path = [{'M': 50, 'T': 0.5, 'K': 2, 'epsilon': 0.65, 'result': 0.01},
        {'M': 50, 'T': 0.5, 'K': 2, 'epsilon': 0.75, 'result': 0.01}, 
        {'M': 50, 'T': 0.55, 'K': 2, 'epsilon': 0.65, 'result': 0.03}, 
        {'M': 50, 'T': 0.55, 'K': 2, 'epsilon': 0.75, 'result': 0.04}, 
        {'M': 50, 'T': 0.6, 'K': 2, 'epsilon': 0.65, 'result': 0.03}, 
        {'M': 50, 'T': 0.6, 'K': 2, 'epsilon': 0.75, 'result': 0.0},
        {'M': 50, 'T': 0.65, 'K': 2, 'epsilon': 0.65, 'result': 0.07},
        {'M': 50, 'T': 0.65, 'K': 2, 'epsilon': 0.75, 'result': 0.02},
        {'M': 50, 'T': 0.7, 'K': 2, 'epsilon': 0.75, 'result': 0.03}, 
        {'M': 50, 'T': 0.75, 'K': 2, 'epsilon': 0.75, 'result': 0.07}, 
        {'M': 100, 'T': 0.5, 'K': 2, 'epsilon': 0.65, 'result': 0.0}, 
        {'M': 100, 'T': 0.5, 'K': 2, 'epsilon': 0.75, 'result': 0.0},
        {'M': 100, 'T': 0.55, 'K': 2, 'epsilon': 0.65, 'result': 0.02},
        {'M': 100, 'T': 0.55, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 100, 'T': 0.6, 'K': 2, 'epsilon': 0.65, 'result': 0.01}, 
        {'M': 100, 'T': 0.6, 'K': 2, 'epsilon': 0.75, 'result': 0.01},
        {'M': 100, 'T': 0.65, 'K': 2, 'epsilon': 0.65, 'result': 0.07}, 
        {'M': 100, 'T': 0.65, 'K': 2, 'epsilon': 0.75, 'result': 0.01}, 
        {'M': 100, 'T': 0.7, 'K': 2, 'epsilon': 0.75, 'result': 0.01},
        {'M': 100, 'T': 0.75, 'K': 2, 'epsilon': 0.75, 'result': 0.03},
        {'M': 200, 'T': 0.5, 'K': 2, 'epsilon': 0.65, 'result': 0.0}, 
        {'M': 200, 'T': 0.5, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 200, 'T': 0.55, 'K': 2, 'epsilon': 0.65, 'result': 0.0}, 
        {'M': 200, 'T': 0.55, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 200, 'T': 0.6, 'K': 2, 'epsilon': 0.65, 'result': 0.01}, 
        {'M': 200, 'T': 0.6, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 200, 'T': 0.65, 'K': 2, 'epsilon': 0.65, 'result': 0.03}, 
        {'M': 200, 'T': 0.65, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 200, 'T': 0.7, 'K': 2, 'epsilon': 0.75, 'result': 0.01}, 
        {'M': 200, 'T': 0.75, 'K': 2, 'epsilon': 0.75, 'result': 0.07}, 
        {'M': 500, 'T': 0.5, 'K': 2, 'epsilon': 0.65, 'result': 0.0}, 
        {'M': 500, 'T': 0.5, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 500, 'T': 0.55, 'K': 2, 'epsilon': 0.65, 'result': 0.0}, 
        {'M': 500, 'T': 0.55, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 500, 'T': 0.6, 'K': 2, 'epsilon': 0.65, 'result': 0.01}, 
        {'M': 500, 'T': 0.6, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 500, 'T': 0.65, 'K': 2, 'epsilon': 0.65, 'result': 0.07}, 
        {'M': 500, 'T': 0.65, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 500, 'T': 0.7, 'K': 2, 'epsilon': 0.75, 'result': 0.01}, 
        {'M': 500, 'T': 0.75, 'K': 2, 'epsilon': 0.75, 'result': 0.06},
        {'M': 1000, 'T': 0.5, 'K': 2, 'epsilon': 0.65, 'result': 0.0}, 
        {'M': 1000, 'T': 0.5, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 1000, 'T': 0.55, 'K': 2, 'epsilon': 0.65, 'result': 0.0},
        {'M': 1000, 'T': 0.55, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 1000, 'T': 0.6, 'K': 2, 'epsilon': 0.65, 'result': 0.0}, 
        {'M': 1000, 'T': 0.6, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 1000, 'T': 0.65, 'K': 2, 'epsilon': 0.65, 'result': 0.051}, 
        {'M': 1000, 'T': 0.65, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 1000, 'T': 0.7, 'K': 2, 'epsilon': 0.75, 'result': 0.0}, 
        {'M': 1000, 'T': 0.75, 'K': 2, 'epsilon': 0.75, 'result': 0.04}]

df_0_swap = pd.DataFrame(path)

path = "results_1_swap_K2_beta.csv"
df = pd.read_csv(path)
df['result'] = df['result']/500


df_combined_uniform_beta = pd.concat([df, df_0_swap], ignore_index=True)
df_combined_uniform_beta = df_combined_uniform_beta[df_combined_uniform_beta["M"].isin([50, 100, 200, 500, 1000])]

#%%
df = df_combined_uniform_beta
df["epsilon"] = df["epsilon"].replace(0.65, 0.67)
df["T"] = df["T"].replace(0.65, 0.67)

df = df[df["epsilon"] == 0.67]
df = df


output_folder = "plots_combined"
os.makedirs(output_folder, exist_ok=True)

# Get unique values
Ms = sorted(df['M'].unique())
epsilons = sorted(df['epsilon'].unique())
Ts = sorted(df['T'].unique())

# Define colors for M
# Define a color for each value
color_map = {
    50:   "blue",  # kräftiges Blau
    100:  "#33a02c",  # kräftiges Grün
    200:  "#a6cee3",  # helles Blau
    500:  "#b2df8a",  # helles Grün
    1000: "#08519c"   # dunkles Blau
}

# Define markers for epsilon (expand if more than 3 epsilons)
marker_map = {eps: m for eps, m in zip(epsilons, ['s', '*'])}
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
