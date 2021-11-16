import csv
import pandas as pd
import numpy as np

input_file = 'Quest_20210416_scene(3)_user(G)_all.csv'

df = pd.read_csv(input_file, usecols=['input_head_angular_vec_x', 'input_head_angular_vec_y', 'input_head_angular_vec_z'])
df['head_ang_vec_mean'] = df.abs().mean(axis=1)
df.drop('input_head_angular_vec_x', axis=1, inplace=True)
df.drop('input_head_angular_vec_y', axis=1, inplace=True)
df.drop('input_head_angular_vec_z', axis=1, inplace=True)
print(df['head_ang_vec_mean'].max())
print(df['head_ang_vec_mean'].min())
df.to_csv('v_data.csv', index=False)
n = df['head_ang_vec_mean'].to_numpy()
print(n)