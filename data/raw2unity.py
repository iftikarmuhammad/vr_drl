import math
import scipy.io as sio
from scipy.spatial.transform import Rotation
import pandas as pd

raw_f = sio.loadmat('raw/motions10fps.mat')['allData'][2][2]

timestamp = []
x = []
y = []
z = []
w = []
yaw = []
pitch = []
roll = []

for i in range(0,3):
	for row in raw_f:
		print(row)
		if i== 0:
			t = row[0] * 705600000
		else:
			t += 70560000
		timestamp.append(t)
		yaw.append(math.radians(row[1]))
		pitch.append(math.radians(row[2]))
		roll.append(math.radians(row[3]))
		rot = Rotation.from_euler('xyz', [row[2], row[1], row[3]], degrees=True)
		rot_quat = rot.as_quat()
		x.append(rot_quat[0])
		y.append(rot_quat[1])
		z.append(rot_quat[2])
		w.append(rot_quat[3])


raw_data = {
			'timestamp' : timestamp,
			'biosignal_0' : 0,
			'biosignal_1' : 0,
			'biosignal_2' : 0,
			'biosignal_3' : 0,
			'biosignal_4' : 0,
			'biosignal_5' : 0,
			'biosignal_6' : 0,
			'biosignal_7' : 0,
			'input_projection_left'	:-1,
			'input_projection_top'	:1,
			'input_projection_right':1,
			'input_projection_bottom':-1,
			'predicted_projection_left'	:-1,
			'predicted_projection_top'	:1,
			'predicted_projection_right':1,
			'predicted_projection_bottom':-1,
			'prediction_time': 100,
			'input_orientation_x' : x,
			'input_orientation_y' : y,
			'input_orientation_z' : z,
			'input_orientation_w' : w,
			'input_orientation_yaw'		: 0,
			'input_orientation_pitch'	: 0,
			'input_orientation_roll'	: 0,
			'angular_vec_x' : 0,
			'angular_vec_y' : 0,
			'angular_vec_z' : 0,
			'acceleration_x' : 0,
			'acceleration_y' : 0,
			'acceleration_z' : 0,
			'magnetic_x' : 0,
			'magnetic_y' : 0,
			'magnetic_z' : 0,
			'predicted_orientation_x' : 0,
			'predicted_orientation_y' : 0,
			'predicted_orientation_z' : 0,
			'predicted_orientation_w' : 0,
			'predicted_orientation_yaw' 	: 0,
			'predicted_orientation_pitch' 	: 0,
			'predicted_orientation_roll' 	: 0,
			}

df = pd.DataFrame(raw_data, columns = ['timestamp',
                            'biosignal_0', 'biosignal_1', 'biosignal_2', 'biosignal_3', 'biosignal_4', 'biosignal_5', 'biosignal_6', 'biosignal_7',
							'acceleration_x', 'acceleration_y', 'acceleration_z',
							'angular_vec_x', 'angular_vec_y', 'angular_vec_z',
							'magnetic_x', 'magnetic_y', 'magnetic_z',
							'input_orientation_x', 'input_orientation_y', 'input_orientation_z', 'input_orientation_w',
							'input_orientation_yaw', 'input_orientation_pitch', 'input_orientation_roll',
							'input_projection_left', 'input_projection_top', 'input_projection_right', 'input_projection_bottom',
							'prediction_time',
							'predicted_orientation_x', 'predicted_orientation_y', 'predicted_orientation_z', 'predicted_orientation_w',
							'predicted_orientation_yaw', 'predicted_orientation_pitch', 'predicted_orientation_roll',
							'predicted_projection_left', 'predicted_projection_top', 'predicted_projection_right', 'predicted_projection_bottom',
				])
print(df['input_orientation_yaw'])

export_csv = df.to_csv ('unity_input_video3_user3.csv', index = None, header=True)