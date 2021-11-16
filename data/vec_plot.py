import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

vec_df = pd.read_csv('ang_vec/ang_vec_3users.csv')

vec_df = vec_df.clip(-4,4)

for i in range(0,3):
    user = 'user%s'%str(i+1)
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.plot(vec_df[user], color='C%s'%i)
    plt.title('Angular Velocity Input User %s' %str(i+1))
    plt.xlabel('Timestep')
    plt.ylabel('Angular Velocity (rad/s)')
    # plt.savefig('../result/bb_angularvec_video%s_user%s_np.png' %(3, str(i+1)))
    plt.show()