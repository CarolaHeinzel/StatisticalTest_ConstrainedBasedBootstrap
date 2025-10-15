#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 08:56:43 2025

@author: ch1158
"""
import matplotlib.pyplot as plt
import numpy as np

# Beispiel-Daten

# original hypotheses
# K = 2, M = 100, 1000, K = 3, M = 100, 1000, K  =7, M =100, 10000
a = [sum(res_h0_K2)] *5  # Type I Error
b = [1-sum(res_h1_K2_pa1)/100, 1-sum(res_h1_K2_pa095)/100, 1-sum(res_h1_K2_pa09)/100, 1-sum(res_h1_K2_pa085)/100, 1-sum(res_h1_K2_pa08)/100]  # Type II Errir

a = [sum(res_h0_K3)] *5  # Type I Error
b = [1-sum(res_h1_K3_pa1)/100, 1-sum(res_h1_K3_pa095)/100, 1-sum(res_h1_K3_pa09)/100, 1-sum(res_h1_K3_pa085)/100, 1-sum(res_h1_K3_pa08)/100]  # Type II Errir


b = [1-sum(res_h1_K7_pa1)/100, 1-sum(res_h1_K7_pa095)/100, 1-sum(res_h1_K7_pa09)/100, 1-sum(res_h1_K7_pa085)/100, 1-sum(res_h1_K7_pa08)/100]  # Type II Errir
b = [1-sum(res_h1_K2_pa1_1000)/100, 1-sum(res_h1_K2_pa095_1000)/100, 1-sum(res_h1_K2_pa09_1000)/100, 1-sum(res_h1_K2_pa085_1000)/100, 1-sum(res_h1_K2_pa08_1000)/100]  # Type II Errir

# swapped hypotheses
# K = 2, M = 100, 10000, K = 3, M = 100, 1000, K  =7, M =100, 10000
#a = [0, 0, 0.04, 0.05, 0, 0 ]  # Type I Error
#b = [0.49, 0.07,  0.82, 0.23, 0.81, 0.39]  # Type II Errir


# Anzahl der Balkenpaare
n = len(a)

# Abstand zwischen Gruppen
group_spacing = 1.5
bar_width = 0.35

# X-Positionen für Balkenpaare
indices = np.arange(n) * group_spacing

# Plot
fig, ax = plt.subplots(figsize=(8, 4))

# Balken zeichnen
bars_a = ax.bar(indices - bar_width/2, a, width=bar_width, label='Type I Error', color='skyblue')
bars_b = ax.bar(indices + bar_width/2, b, width=bar_width, label='Type II Error', color='blue')

# X-Achse beschriften
labels = ['T = 1', 'T = 0.95','T = 0.9','T = 0.85','T = 0.8']


#labels = ['K = 2, M = 100', 'K = 2, M = 10000', 'K = 3, M = 100', 'K = 3, M = 10000', 'K = 7, M =100', 'K = 7, M = 10000']

ax.set_xticks(indices)  # Use actual bar group positions
ax.set_xticklabels(labels, rotation=90)  # Rotate labels

# Legende und Achsentitel
ax.legend()
ax.set_ylabel('Error')

# Layout und Anzeige
plt.tight_layout()
plt.show()
