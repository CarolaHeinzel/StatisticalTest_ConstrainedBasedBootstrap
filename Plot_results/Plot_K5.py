import pandas as pd
import ast
import numpy as np
path = "results_0_swap_K5_1212_uniform.csv"

df_new = pd.read_csv(path)
#df['result'] = df['result'].apply(lambda x: np.array(ast.literal_eval(x.replace('.', '.'))))
#df1['result'] = df1['result'].apply(lambda x: np.array(ast.literal_eval(x.replace('.', '.'))))
#df['result'] = df['result'].apply(lambda x: np.sum(x)/500)

#df.to_csv("results_1_swap_K5_2111_1000_500_fracion_2.csv", index=False)
path = "K5_other_H0.csv"
df_new1 = pd.read_csv(path,      sep=",",
    skipinitialspace=True)

print(df_new1)
path = "results_1_swap_K5_2111_1000_500.csv"


df = pd.read_csv(path)
df['result'] = df['result']#/500
df['result'] = df['result'].apply(lambda x: np.array(ast.literal_eval(x.replace('.', '.'))))
#df1['result'] = df1['result'].apply(lambda x: np.array(ast.literal_eval(x.replace('.', '.'))))
df['result'] = df['result'].apply(lambda x: np.sum(x)/500)

df2 = df[df["K"] == 5]
df_combined = pd.concat([df_new, df_new1, df2], ignore_index=True)

#%%
df  = df_combined
#df["epsilon"] = df["epsilon"].replace(0.65, 0.67)

df_combined = df_combined[df_combined["epsilon"] == 0.75]


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    50: "orange",     # existing
    100: "blue",      # existing
    150: "cyan",      # added
    200: "green",     # existing
    250: "purple",    # added
    300: "yellow",    # added
    350: "pink",      # added
    400: "brown",     # added
    450: "olive",     # added
    500: "magenta",   # added
    1000: "red"       # existing
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
#%%

df = df2
# Choose one T value that should be plotted (example: T = 5)
T_value = df["T"].unique()[2]     # or set manually, e.g. T_value = 5

# Filter the DataFrame to one T
df_T = df[df["T"] == T_value]

# Define marker styles for different epsilon values
marker_map = {
    eps: marker for eps, marker in zip(
        df_T["epsilon"].unique(),
        ["o", "s", "^", "D", "v", "<", ">"]  # extend if needed
    )
}

# Start plotting
plt.figure(figsize=(7,5))

for eps in df_T["epsilon"].unique():
    print(eps)
    # Select all rows for one epsilon value
    df_eps = df_T[df_T["epsilon"] == eps]

    # Plot M vs result using a dedicated marker
    plt.plot(
        df_eps["M"],
        df_eps["result"],
        marker_map[eps],
        linestyle="-",
        label=f"epsilon = {eps}"
    )

# Axis labels and legend
plt.xlabel("M", fontsize = 16)
plt.ylabel("Fraction of reject null hypothesis", fontsize = 16)
plt.title(f"T = {T_value}", fontsize = 16)
plt.legend(fontsize = 14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
