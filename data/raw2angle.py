from math import radians as rad
import scipy.io as sio
from scipy.spatial.transform import Rotation
import pandas as pd
import numpy as np

user = 3

raw = sio.loadmat('raw/motions10fps.mat')['allData'][2]		# Basketball video (idx=2)

all_ang_vec = []

for u in range(0, user):
	raw_f = raw[u]
	mean_ang = []
	for i in range(0, len(raw_f)):
		yaw = rad(raw_f[i][1])
		pitch = rad(raw_f[i][2])
		roll = rad(raw_f[i][3])
		mean_ang.append([pitch, roll, yaw])
	all_ang_vec.append(mean_ang)

all_ang_vec = np.array(all_ang_vec)
vec_dict = {
	'user1r' : all_ang_vec[0][:,0],
	'user1p' : all_ang_vec[0][:,1],
	'user1y' : all_ang_vec[0][:,2],
	'user2r' : all_ang_vec[1][:,0],
	'user2p' : all_ang_vec[1][:,1],
	'user2y' : all_ang_vec[1][:,2],
	'user3r' : all_ang_vec[2][:,0],
	'user3p' : all_ang_vec[2][:,1],
	'user3y' : all_ang_vec[2][:,2],
}

print(vec_dict)

df = pd.DataFrame(vec_dict, columns = ['user1r','user1p', 'user1y', 'user2r','user2p', 'user2y', 'user3r','user3p', 'user3y'])
export_csv = df.to_csv ('ang_3users.csv', index = None, header=True)