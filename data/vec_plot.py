import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing


vec_df = pd.read_csv('ang_vec/ang_vec_3users.csv')



# vec_df = vec_df.clip(-4,4)
scaler = preprocessing.StandardScaler()
# print((vec_df['user1'] > 2) & (vec_df['user1'] <-2))

for i in range(0,3):
    user = 'user%s'%str(i+1)
    fig, ax = plt.subplots()
    ax.grid(True)
    # plt.plot(scaler.fit_transform(vec_df[user].values.reshape(-1,1)), color='C%s'%i)
    median = vec_df.loc[(vec_df[user]<4) & vec_df[user] >-4, user].median()
    vec_df[user] = vec_df[user].mask(vec_df[user]>4, median)
    vec_df[user] = vec_df[user].mask(vec_df[user]<-4, median)
    # vec_df[user] = scaler.fit_transform(vec_df[user].values.reshape(-1,1))
    plt.plot(vec_df[user], color='C%s'%i)
    plt.title('Angular Velocity Input User %s' %str(i+1))
    plt.xlabel('Timestep')
    plt.ylabel('Angular Velocity (rad/s)')
    # plt.savefig('../result/bb_angularvec_video%s_user%s_np.png' %(3, str(i+1)))
    plt.show()