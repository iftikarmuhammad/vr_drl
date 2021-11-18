import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import math

vec_df = pd.read_csv('ang_vec/ang_vec_3users.csv')
bb_df1 = pd.read_csv('video3user1_notpredict_blackborder.csv', header=None)
bb_df2 = pd.read_csv('video3user2_notpredict_blackborder.csv', header=None)
bb_df3 = pd.read_csv('video3user3_notpredict_blackborder.csv', header=None)
bb_df = pd.concat([bb_df1, bb_df2, bb_df3])

vec_df = vec_df.clip(-4,4)

# without absolute value and polyfit
# for i in range(0,3):
#     user = 'user%s'%str(i+1)
#     fig, ax = plt.subplots()
#     ax.grid(True)
#     plt.scatter(bb_df.iloc[i][:299], vec_df[user], color='C%s'%i)
#     plt.title('Blackborder vs Angular Velocity User %s' %str(i+1))
#     plt.xlabel('Blackborder Pct (%)')
#     plt.ylabel('Angular Velocity (rad/s)')
#     plt.savefig('../result/bb_angularvec_video%s_user%s_np.png' %(3, str(i+1)))
#     plt.show()

# with absolute value and polyfit
for i in range(0,3):
    user = 'user%s'%str(i+1)

    y = bb_df.iloc[i][:299]
    x = vec_df[user].abs() * 180/math.pi

    lin = poly.polyfit(x, y, 1)
    quad = poly.polyfit(x, y, 2)

    lin_f = poly.Polynomial(lin)
    quad_f = poly.Polynomial(quad)

    fig, ax = plt.subplots()
    ax.grid(True)
    plt.scatter(x, y, color='C%s'%i, label='Data sample')
    x1,x2,y1,y2 = plt.axis()
    x = np.arange(x2,step=0.1)
    plt.plot(x, lin_f(x), color='r', label='Linear regression')
    plt.plot(x, quad_f(x), color='b', label='Quadratic regression')
    # if i!=2:
    #     plt.axis((0,2,y1,y2))
    plt.title('Blackborder vs Angular Velocity User %s' %str(i+1))
    plt.ylabel('Blackborder Pct (%)')
    plt.xlabel('Angular Velocity (rad/s)')
    plt.legend(loc='upper right')
    plt.savefig('../result/bb_angularvec_video%s_user%s_np.png' %(3, str(i+1)))
    plt.show()