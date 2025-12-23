import pandas as pd
import ast
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
path = "results_0_swap_K2__uniform.csv"
df_new = pd.read_csv(path)
path = "results_1_swap_K2_uniform.csv"
df = pd.read_csv(path)
df['result'] = df['result']/500
df_combined = pd.concat([df_new, df], ignore_index=True)
df_combined = df_combined[df_combined["M"].isin([50, 100, 200, 500, 1000])]

#%%
df  = df_combined
df_combined = df_combined[df_combined["epsilon"] == 0.67]
df = df_combined


# Assume df['result'] is already numeric (sum of arrays)
output_folder = "plots_combined"
os.makedirs(output_folder, exist_ok=True)

# Get unique values
Ms = sorted(df['M'].unique())
epsilons = sorted(df['epsilon'].unique())
Ts = sorted(df['T'].unique())

# Define colors for M
# Define a color for each value
color_map = {
    50:   "blue",  #
    100:  "#33a02c",  
    200:  "#a6cee3",  
    500:  "#b2df8a",  
    1000: "#08519c"   
}

# Define markers for epsilon (expand if more than 3 epsilons)
marker_map = {eps: m for eps, m in zip(epsilons, ['s', '*'])}

# Aggregate sum(result) per (M, epsilon, T)
agg = df.groupby(['M', 'epsilon', 'T'], as_index=False)['result'].sum()
agg = agg.sort_values('T')
#agg.loc[df['result'] == 200] = 100
#agg.loc[df['result'] == 199] = 100

plt.figure(figsize=(10, 6))

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
plt.ylabel('Fraction of reject null hypothesis', fontsize = 20)
plt.grid(True)
plt.legend(fontsize = 16)
plt.tight_layout()
# Tick-Nummern vergrößern
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Save the combined plot
plt.savefig(os.path.join(output_folder, "combined_plot.png"))
plt.show()
