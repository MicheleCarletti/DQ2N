"""
@author Michele Carletti
Get statistics from comparison csv file
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

num_layers = 4
model_code = "3L-DQ2N" if num_layers==3 else "4L-DQ2N"
action_dict = {0.0:"Push to the left", 1.0:"Push to the right"}
csv_file = "./logs/comp_log_20250607_0933_3l.csv" if num_layers == 3 else "logs/comp_log_20250607_0913_4l.csv"
df = pd.read_csv(csv_file, sep=';')

# Mean and std for each expZ deviation
print(f"== ExpZ error per qubit {model_code} ==")
for i in range(4):
    mean_val = df[f'diff_expZ_{i}'].mean()
    std_val = df[f'diff_expZ_{i}'].std()
    print(f"Qubit {i}: mean = {mean_val:.3f}, std = {std_val:.3f}")

# Mean and std for each Q-value deviation
print(f"\n== Q-value error per qubit {model_code} ==")
for i in range(2):
    mean_val = df[f'diff_Q_{i}'].mean()
    std_val = df[f'diff_Q_{i}'].std()
    print(f"{action_dict[i]}: mean = {mean_val:.3f}, std = {std_val:.3f}")

# Percentage of divergent actions
perc_div = df['action_diff'].mean() * 100
print(f"\nPercentage of divergent action: {perc_div:.2f} %")

# Plot error histograms for ExpZ deviation per qubit
fig, axes = plt.subplots(2, 2, figsize=(12,8))  # 2 rows x 2 columns
fig.suptitle(f'ExpZ error distribution per qubit\n{model_code}', fontsize=16)
for i, ax in enumerate(axes.flat):
    sns.histplot(df[f'diff_expZ_{i}'], kde=True, ax=ax, bins=20, color='deepskyblue' if num_layers==3 else 'mediumseagreen' )
    ax.set_title(f'Qubit {i}')
    ax.set_xlabel('Absolute error')
    ax.set_ylabel('Count')
    ax.grid()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Space for title
plt.show()

# Plot error histograms for Q-value deviation per action
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row x 2 colums
fig.suptitle(f'Q-value error distribution per action\n{model_code}', fontsize=16)
for i, ax in enumerate(axes.flat):
    sns.histplot(df[f'diff_Q_{i}'], kde=True, ax=ax, bins=20, color='tomato' if num_layers==3 else 'gold' )
    ax.set_title(f'{action_dict[i]}')
    ax.set_xlabel('Absolute error')
    ax.set_ylabel('Count')
    ax.grid()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Space for title
plt.show()
