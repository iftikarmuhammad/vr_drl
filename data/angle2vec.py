from math import radians as rad
import scipy.io as sio
from scipy.spatial.transform import Rotation
import pandas as pd

user = 3

raw = sio.loadmat('raw/motions10fps.mat')['allData'][2]		# Basketball video (idx=2)

all_ang_vec = []

for u in range(0, user):
	raw_f = raw[u]
	mean_ang_vec = []
	for i in range(0, len(raw_f)):
		if i == 0:
			j = 1
		else:
			j = i
		yaw = rad(raw_f[j][1])
		pitch = rad(raw_f[j][2])
		roll = rad(raw_f[j][3])
		prev_yaw = rad(raw_f[j-1][1])
		prev_pitch = rad(raw_f[j-1][2])
		prev_roll = rad(raw_f[j-1][3])

		yaw_vec = (yaw - prev_yaw) / 0.1
		pitch_vec = (pitch - prev_pitch) / 0.1
		roll_vec = (roll - prev_roll) / 0.1

		mean_vec = (yaw_vec + pitch_vec + roll_vec) / 3
		mean_ang_vec.append(mean_vec)
	all_ang_vec.append(mean_ang_vec)

vec_dict = {
	'user1' : all_ang_vec[0],
	'user2' : all_ang_vec[1],
	'user3' : all_ang_vec[2]
}

df = pd.DataFrame(vec_dict, columns = ['user1','user2','user3'])
export_csv = df.to_csv ('ang_vec_3users.csv', index = None, header=True)