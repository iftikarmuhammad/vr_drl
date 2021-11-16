import argparse

import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
#import skinematics as skin
import matplotlib.pyplot as plt
import quaternion
from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers, optimizers
from disolve import remdiv
import math
# modularized library import
from preparets import preparets
from train_test_split import train_test_split_tdnn

#from eul2quat_bio import eul2quat_bio
#from quat2eul_bio import quat2eul_bio
from cap_prediction import cap_prediction
from crp_prediction import crp_prediction
from nop_prediction import nop_prediction
from solve_discontinuity import solve_discontinuity
from rms import rms
    
def eul2quat_bio (eul):
    # Convert Euler [oculus] to Quaternion [oculus]
    eul = np.deg2rad(eul)
    X_eul = eul[:,0]
    Y_eul = eul[:,1]
    Z_eul = eul[:,2]

    cos_pitch, sin_pitch = np.cos(X_eul/2), np.sin(X_eul/2)
    cos_yaw, sin_yaw = np.cos(Y_eul/2), np.sin(Y_eul/2)
    cos_roll, sin_roll = np.cos(Z_eul/2), np.sin(Z_eul/2)

    # order: w,x,y,z
    # quat = unit_q(quat)
    quat = np.nan * np.ones( (eul.shape[0],4) )
    quat[:,0] = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
    quat[:,1] = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
    quat[:,2] = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll 
    quat[:,3] = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll     
    return (quat)
	

def calc_optimal_overhead(hmd_orientation, frame_orientation, hmd_projection):
	q_d = np.matmul(
			np.linalg.inv(quaternion.as_rotation_matrix(hmd_orientation)),
			quaternion.as_rotation_matrix(frame_orientation)
			)
	#Projection Orientation:
        #hmd_projection[0] : Left (Negative X axis)
        #hmd_projection[1] : Top (Positive Y axis)
        #hmd_projection[2] : Right (Positive X axis)
        #hmd_projection[3] : Bottom (Negative Y axis)
        
	lt = np.matmul(q_d, [hmd_projection[0], hmd_projection[1], 1])
	p_lt = np.dot(lt, 1 / lt[2])
	
	rt = np.matmul(q_d, [hmd_projection[2], hmd_projection[1], 1])
	p_rt = np.dot(rt, 1 / rt[2])
	
	rb = np.matmul(q_d, [hmd_projection[2], hmd_projection[3], 1])
	p_rb = np.dot(rb, 1 / rb[2])
	
	lb = np.matmul(q_d, [hmd_projection[0], hmd_projection[3], 1])
	p_lb = np.dot(lb, 1 / lb[2])
	
	p_l = min(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
	p_t = max(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
	p_r = max(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
	p_b = min(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
	
	size = max(p_r - p_l, p_t - p_b)
	a_overfilling = size * size
	
	a_hmd = (hmd_projection[2] - hmd_projection[0]) * (hmd_projection[1] - hmd_projection[3])
	
	return (a_overfilling / a_hmd - 1)*100

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

'''
#####   SYSTEM INITIALIZATION    #####
'''

tf.reset_default_graph()
tf.set_random_seed(2)
np.random.seed(2)

parser = argparse.ArgumentParser(description='Offline Motion Prediction')
parser.add_argument('-a', '--anticipation', default=250, type=int)
args = parser.parse_args()


# Directory Name
inFile = '20210628_scene(RidingDragon)_user(1)';
inFileFull = inFile + '.csv';
outFile = inFile + '_('+str(args.anticipation)+')_ANN2021_880-250.csv';
# inFile = '20200924_scene(3)_user(5).csv';
# outFile = '20200924_scene(3)_user(5)_('+str(args.anticipation)+').csv';

try:
	stored_df_quat = pd.read_csv(inFileFull)
	train_gyro_data = np.array(stored_df_quat[['angular_vec_x', 'angular_vec_y', 'angular_vec_z']], dtype=np.float32)
	train_acce_data = np.array(stored_df_quat[['acceleration_x', 'acceleration_y', 'acceleration_z']], dtype=np.float32)
	train_magn_data = np.array(stored_df_quat[['magnetic_x', 'magnetic_y', 'magnetic_z']], dtype=np.float32)
	projection_data = np.array(stored_df_quat[['input_projection_left', 'input_projection_top', 'input_projection_right', 'input_projection_bottom']], dtype=np.int64)
	train_euler_data = np.array(stored_df_quat[['input_orientation_pitch', 'input_orientation_roll', 'input_orientation_yaw']], dtype=np.float32)
	#train_obci_data = np.array(stored_df_quat[['biosignal_0', 'biosignal_1', 'biosignal_2', 'biosignal_3', 'biosignal_4', 'biosignal_5', 'biosignal_6', 'biosignal_7']], dtype=np.float32)
	train_time_data = np.array(stored_df_quat['timestamp'], dtype=np.int64)
	train_time_data = train_time_data/705600000
	train_data_id = stored_df_quat.shape[0]
	print('\nSaved data loaded...\n')
except:
    raise
	
'''
#####   TRAINING DATA PREPROCESSING    #####
'''
print('Training data preprocessing is started...')
#convertion
# cut_variable = 0
cut_variable = 1200
# Remove zero data from collected training data
# system_rate = round((train_data_id+1)/float(np.max(train_time_data) - train_time_data[0]))
system_rate = 60
idle_period = int(2 * system_rate)
train_gyro_data = train_gyro_data[idle_period+cut_variable:train_data_id, :]*180/np.pi
train_acce_data = train_acce_data[idle_period+cut_variable:train_data_id, :]
train_magn_data = train_magn_data[idle_period+cut_variable:train_data_id, :]
train_euler_data = train_euler_data[idle_period+cut_variable:train_data_id, :]
projection_data = projection_data[idle_period+cut_variable:train_data_id]
train_time_data = train_time_data[idle_period+cut_variable:train_data_id]

train_alfa_data = np.diff(train_gyro_data, axis=0)/np.diff(train_time_data, axis=0).reshape(-1, 1)
train_alfa_data = np.row_stack([np.zeros(shape=(1, train_alfa_data.shape[1]), dtype=np.float32), train_alfa_data])
#train_alfa_data = train_alfa_data * np.pi / 180

# Calculate the head orientation
#train_gyro_data = train_gyro_data * np.pi / 180
train_acce_data = train_acce_data / 9.8
train_magn_data = train_magn_data
train_euler_data = train_euler_data*180/np.pi

"""Velocity data"""
train_velocity_data = np.diff(train_euler_data, axis=0)/np.diff(train_time_data, axis=0).reshape(-1, 1)
train_velocity_data = np.row_stack([np.zeros(shape=(1, train_velocity_data.shape[1]), dtype=np.float), train_velocity_data])

train_euler_data = solve_discontinuity(train_euler_data)

# Create data frame of all features and smoothing
sliding_window_time = 100
sliding_window_size = int(np.round(sliding_window_time * system_rate / 1000))
#sliding_window_size = 29
SG_sliding_window_size = 81

# anticipation time
anticipation_time = args.anticipation
anticipation_size = int(np.round(anticipation_time * system_rate/1000))
print ('Anticipation Size: ',anticipation_size)


ann_feature = np.column_stack([train_euler_data, 
                               train_gyro_data, 
                               train_acce_data, 
#                               train_magn_data,
                               ])
    
feature_name = ['pitch', 'roll', 'yaw',  
                'gX', 'gY', 'gZ', 
                'aX', 'aY', 'aZ', 
#                'mX', 'mY', 'mZ',
                ]

ann_feature_df = pd.DataFrame(ann_feature, columns=feature_name)
ann_feature_df = ann_feature_df.rolling(sliding_window_size, center=True, min_periods=1).mean()


# Create the time-shifted IMU data as the supervisor and assign the ann_feature as input
spv_name = ['pitch', 'roll', 'yaw']#, 'gX', 'gY', 'gZ'
target_series_df = ann_feature_df[spv_name].iloc[anticipation_size::].reset_index(drop=True)
input_series_df = ann_feature_df.iloc[:-anticipation_size].reset_index(drop=True)

input_nm = len(input_series_df.columns)
target_nm = len(target_series_df.columns)


'''
#####   NEURAL NETWORK MODEL TRAINING    #####
'''
print('Neural network training is started...')


# Neural network parameters
DELAY_SIZE = int(25 * (system_rate / 250))
# DELAY_SIZE = int(25 * (system_rate / 250))
TEST_SIZE = 0.5 
TRAIN_SIZE = 1 - TEST_SIZE
print ('Delay Size: ',DELAY_SIZE)
# Variables
TRAINED_MODEL_NAME = './best_net'


# Import datasets
input_series = np.array(input_series_df)
target_series = np.array(target_series_df)

"""""New Version """""""""
## Split training and testing data
#x_seq, t_seq = preparets(input_series, target_series, DELAY_SIZE)
#data_length = x_seq.shape[0]
#scaler = preprocessing.StandardScaler().fit(input_series)	# fit saves normalization coefficient into scaler
#
#x_seq, t_seq = remdiv(x_seq, t_seq, DELAY_SIZE)
#
##Normalize training data, then save the normalization coefficient
#for i in range(0,len(x_seq)):
#    x_seq[i,:,:] = scaler.transform(x_seq[i,:,:])
#    
#x_train, t_train, x_test, t_test = train_test_split_tdnn(x_seq, target_series, TEST_SIZE)


"""""old Version """""""""
# Normalized input and target series
# input_norm = preprocessing.scale(input_series)normalizer = preprocessing.StandardScaler()
normalizer = preprocessing.StandardScaler()
normalizer.fit(input_series)
input_norm = normalizer.transform(input_series)


# Reformat the input into TDNN format
x_seq, t_seq = preparets(input_norm, target_series, DELAY_SIZE)
data_length = x_seq.shape[0]


# Split training and testing data
x_train, t_train, x_test, t_test = train_test_split_tdnn(x_seq, t_seq, TEST_SIZE)


print('Anticipation time: {}ms\n'.format(anticipation_time))


# Reset the whole tensorflow graph
tf.reset_default_graph()

# Set up the placeholder to hold inputs and targets
x = tf.placeholder(dtype=tf.float32, shape=(None, DELAY_SIZE, input_nm))
t = tf.placeholder(dtype=tf.float32)


## Define TDNN model
#model = models.Sequential([
#        layers.InputLayer(input_tensor=x, input_shape=(DELAY_SIZE, input_nm)),
#        layers.Conv1D(27, DELAY_SIZE, activation=tf.nn.relu, input_shape=(DELAY_SIZE, input_nm), use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
#        layers.Flatten(),
#        layers.Dense(18, activation=tf.nn.relu, use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
#        layers.Dropout(0.2),
#        layers.Dense(9, activation=tf.nn.relu, use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
#        layers.Dropout(0.2),
#        layers.Dense(target_nm, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
#        ])

#Keras Model NN
model = models.Sequential()
model.add(layers.Conv1D(27, DELAY_SIZE, activation=tf.nn.relu, input_shape=(DELAY_SIZE, input_nm), use_bias=True, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Flatten())
model.add(layers.Dense(18, activation=tf.nn.relu, use_bias=True, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(9, activation=tf.nn.relu, use_bias=True, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(target_nm, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(0.01)))

    
# Get the output of the neural network model
y = model(x)


# Define the loss
loss = tf.reduce_mean(tf.square(t-y))

total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, t)))
R_squared = tf.subtract(1.0, tf.div(unexplained_error, total_error))
#
n_epochs=1000
optimizer =  tf.train.AdamOptimizer(learning_rate=0.01)
training_op = optimizer.minimize(loss)

# Start tensorflow session
with tf.Session() as sess:
    # Initiate variable values
    sess.run(tf.global_variables_initializer())
#    model.compile(loss='mean_squared_error', optimizer=optimizers.Adamax(lr=0.01),metrics=['mae'])
#    model.fit(x_train, t_train, epochs=100,validation_split = 0.5)
    
#    file_writer = tf.summary.FileWriter('logdir', sess.graph)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", R_squared)
    merged_summary = tf.summary.merge_all()
#
    for epoch in range (n_epochs):
        summary,_= sess.run([merged_summary,training_op], feed_dict={x:x_train, t:t_train})
#        print("epoch", epoch)
###        file_writer.add_summary(summary, epoch)
    # Set up the optimizer
#    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='CG', options={'maxiter': 1000, 'gtol': 1e-5})
#    
#    
#    # Minimize the error
#    optimizer.minimize(
#            session=sess, 
#            feed_dict={x:x_train, t:t_train},
#            )
    
    # Calculate the testing part result
    y_test = sess.run(y, feed_dict={x:x_test})
    test_mse  = metrics.mean_squared_error(t_test, y_test, multioutput='raw_values')
    
    print('Test result: {}\n'.format(test_mse))
    model.save_weights(TRAINED_MODEL_NAME)
    

# Evaluate performance of a fully trained network
with tf.Session() as new_sess:    
    # Load best trained network
    model.load_weights(TRAINED_MODEL_NAME)
    

    # Evaluate the network with whole sequence
    y_out = new_sess.run(y, feed_dict={x:x_seq})
    
    # Gyro data format: 
    # [0]: Pitch
    # [1]: Roll
    # [2]: Yaw

# Savgol Filter
# y_out = signal.savgol_filter(y_out, SG_sliding_window_size, 3, axis=0)

''' SoftSwitching '''
vel_seq = train_velocity_data[DELAY_SIZE:-anticipation_size]
tempNOP = train_euler_data[DELAY_SIZE:-anticipation_size]
midVel = np.nanpercentile(np.abs(vel_seq),30, axis = 0)
avgVel = np.nanmean(np.abs(vel_seq), axis=0)

for k in range(len(vel_seq)):
    velocity_onedata = vel_seq[k]
    for l in range(0,3):
        xin = (np.abs(velocity_onedata[l])-midVel[l])/avgVel[l]
        alfa = sigmoid(xin)
        y_out[k,l] = alfa*y_out[k,l] + (1-alfa)*tempNOP[k,l]

'''
#####   OCULUS PREDICTION COMPARISON   #####
'''


# Recalibrate and align current time head orientation
euler_o = train_euler_data[DELAY_SIZE:-anticipation_size]
np.savetxt('euler_o.csv', euler_o[anticipation_size:], delimiter=',')
projection_data = projection_data[DELAY_SIZE:-anticipation_size]

# Align current head velocity and acceleration
gyro_o = train_gyro_data[DELAY_SIZE:-anticipation_size]
alfa_o = train_alfa_data[DELAY_SIZE:-anticipation_size]
accel_o = train_acce_data[DELAY_SIZE:-anticipation_size]
magnet_o = train_magn_data[DELAY_SIZE:-anticipation_size]
#obci_o = train_obci_data[DELAY_SIZE:-anticipation_size]
timestamp_plot = train_time_data[int(TRAIN_SIZE*data_length)+DELAY_SIZE:-2*anticipation_size]
timestamp_plot = timestamp_plot*705600000

# Predict orientation
euler_pred_ann = y_out
euler_pred_cap = cap_prediction(euler_o, gyro_o, alfa_o, anticipation_time)
euler_pred_crp = crp_prediction(euler_o, gyro_o, anticipation_time)
euler_pred_nop = nop_prediction(euler_o, anticipation_time)

#TESTING PLOT
euler_pred_ann_test = euler_pred_ann[int(TRAIN_SIZE*data_length):-anticipation_size]
euler_pred_cap_test = euler_pred_cap[int(TRAIN_SIZE*data_length):-anticipation_size]
euler_pred_crp_test = euler_pred_crp[int(TRAIN_SIZE*data_length):-anticipation_size]
euler_pred_nop_test = euler_pred_nop[int(TRAIN_SIZE*data_length):-anticipation_size]

euler_o_test	= euler_o[int(TRAIN_SIZE*data_length)+anticipation_size::]
accel_o_test	= accel_o[int(TRAIN_SIZE*data_length)+anticipation_size::]
magnet_o_test 	= magnet_o[int(TRAIN_SIZE*data_length)+anticipation_size::]
gyro_o_test		= gyro_o[int(TRAIN_SIZE*data_length)+anticipation_size::]
projection_data = projection_data[int(TRAIN_SIZE*data_length)+anticipation_size::]

quat_predict = eul2quat_bio(euler_pred_ann_test)
quat_predict_cap = eul2quat_bio(euler_pred_cap_test)
quat_predict_crp = eul2quat_bio(euler_pred_crp_test)
quat_predict_nop = eul2quat_bio(euler_pred_nop_test)
quat_quat_data = eul2quat_bio(euler_o_test)

#TRAINING PLOT
euler_pred_ann_train = euler_pred_ann[:-(int(TRAIN_SIZE*data_length)+anticipation_size)]
euler_pred_cap_train = euler_pred_cap[:-(int(TRAIN_SIZE*data_length)+anticipation_size)]
euler_pred_crp_train = euler_pred_crp[:-(int(TRAIN_SIZE*data_length)+anticipation_size)]
euler_pred_nop_train = euler_pred_nop[:-(int(TRAIN_SIZE*data_length)+anticipation_size)]

euler_o_train	= euler_o[anticipation_size:-int(TRAIN_SIZE*data_length)]
accel_o_train	= accel_o[anticipation_size:-int(TRAIN_SIZE*data_length)]
magnet_o_train 	= magnet_o[anticipation_size:-int(TRAIN_SIZE*data_length)]
gyro_o_train	= gyro_o[anticipation_size:-int(TRAIN_SIZE*data_length)]

# Calculate prediction error
# Error is defined as difference between:
# Predicted head orientation
# Actual head orientation = Current head orientation shifted by s time
euler_ann_err = np.abs(euler_pred_ann[:-anticipation_size] - euler_o[anticipation_size:])
euler_cap_err = np.abs(euler_pred_cap[:-anticipation_size] - euler_o[anticipation_size:])
euler_crp_err = np.abs(euler_pred_crp[:-anticipation_size] - euler_o[anticipation_size:])
euler_nop_err = np.abs(euler_pred_nop[:-anticipation_size] - euler_o[anticipation_size:])

# Split error value
euler_ann_err_test = euler_ann_err[int(TRAIN_SIZE*data_length)::] 
euler_cap_err_test = euler_cap_err[int(TRAIN_SIZE*data_length)::] 
euler_crp_err_test = euler_crp_err[int(TRAIN_SIZE*data_length)::]
euler_nop_err_test = euler_nop_err[int(TRAIN_SIZE*data_length)::]

euler_ann_err_train = euler_ann_err[: int(TRAIN_SIZE*data_length)] 
euler_cap_err_train = euler_cap_err[: int(TRAIN_SIZE*data_length)]
euler_crp_err_train = euler_crp_err[: int(TRAIN_SIZE*data_length)]
euler_nop_err_train = euler_nop_err[: int(TRAIN_SIZE*data_length)]

#correction
crp_correct = euler_pred_crp[:-anticipation_size]
cap_correct = euler_pred_cap[:-anticipation_size]

numberrow = len(euler_cap_err)  

#error correction
for x in range(1,numberrow):
  if euler_crp_err[x,0]>=180:
      if crp_correct[x,0]>0:
          crp_correct[x,0]= crp_correct[x,0]-360
      else:
          crp_correct[x,0]= crp_correct[x,0]+360
  if euler_crp_err[x,1]>=180:
      if crp_correct[x,1]>0:
          crp_correct[x,1]= crp_correct[x,1]-360
      else:
          crp_correct[x,1]= crp_correct[x,1]+360
  if euler_crp_err[x,2]>=180:
      if crp_correct[x,2]>0:
          crp_correct[x,2]= crp_correct[x,2]-360
      else:
          crp_correct[x,2]= crp_correct[x,2]+360         

for y in range(1,numberrow):
  if euler_cap_err[y,0]>=180:
      if cap_correct[y,0]>0:
          cap_correct[y,0]= cap_correct[y,0]-360
      else:
          cap_correct[y,0]= cap_correct[y,0]+360
  if euler_cap_err[y,1]>=180:
      if cap_correct[y,1]>0:
          cap_correct[y,1]= cap_correct[y,1]-360
      else:
          cap_correct[y,1]= cap_correct[y,1]+360
  if euler_cap_err[y,2]>=180:
      if cap_correct[y,2]>0:
          cap_correct[y,2]= cap_correct[y,2]-360
      else:
          cap_correct[y,2]= cap_correct[y,2]+360            

euler_cap_err1 = np.abs(cap_correct - euler_o[anticipation_size:])
euler_crp_err1 = np.abs(crp_correct - euler_o[anticipation_size:])


# Calculate average error
"""Matrix 1 - MAE"""
ann_mae = np.nanmean(np.abs(euler_ann_err), axis=0)
cap_mae = np.nanmean(np.abs(euler_cap_err), axis=0)
crp_mae = np.nanmean(np.abs(euler_crp_err), axis=0)
nop_mae = np.nanmean(np.abs(euler_nop_err), axis=0)

ann_mae_train = np.nanmean(np.abs(euler_ann_err_train), axis=0)
cap_mae_train = np.nanmean(np.abs(euler_cap_err_train), axis=0)
crp_mae_train = np.nanmean(np.abs(euler_crp_err_train), axis=0)
nop_mae_train = np.nanmean(np.abs(euler_nop_err_train), axis=0)


ann_mae_test = np.nanmean(np.abs(euler_ann_err_test), axis=0)
cap_mae_test = np.nanmean(np.abs(euler_cap_err_test), axis=0)
crp_mae_test = np.nanmean(np.abs(euler_crp_err_test), axis=0)
nop_mae_test = np.nanmean(np.abs(euler_nop_err_test), axis=0)


"""Matrix 2 - Prediction Error with 99% Confidence"""
ann_99_test = np.nanpercentile(euler_ann_err_test,99, axis = 0)


cap_mae1 = np.nanmean(np.abs(euler_cap_err1), axis=0)
crp_mae1 = np.nanmean(np.abs(euler_crp_err1), axis=0)


# Calculate max error
ann_max = np.nanmax(np.abs(euler_ann_err), axis=0)
cap_max = np.nanmax(np.abs(euler_cap_err), axis=0)
crp_max = np.nanmax(np.abs(euler_crp_err), axis=0)
nop_max = np.nanmax(np.abs(euler_nop_err), axis=0)

print('\n--Final Result--')
# print('\nMetrice 0 ('+str(args.anticipation)+') : MAE[Pitch, Roll, Yaw] (Trained)')
# print('MAE ANN, {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_train[0], ann_mae_train[1], ann_mae_train[2]))

if (args.anticipation == 250):
    print('\nMetrice 1 ('+str(args.anticipation)+') : MAE[Pitch, Roll, Yaw] (Tested)')
    print('MAE ANN, {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_test[0], ann_mae_test[1], ann_mae_test[2]))

#print('\nMetrice 2 ('+str(args.anticipation)+') : MAE[Pitch, Roll, Yaw] (Tested)')
#print('99% Percentile Error, {:.2f}, {:.2f}, {:.2f}'.format(ann_99_test[0], ann_99_test[1], ann_99_test[2]))

# get rms stream Testing
ann_rms_stream_test = np.apply_along_axis(rms,1,np.abs(euler_ann_err_test))
cap_rms_stream_test = np.apply_along_axis(rms,1,np.abs(euler_cap_err_test))
crp_rms_stream_test = np.apply_along_axis(rms,1,np.abs(euler_crp_err_test))
nop_rms_stream_test = np.apply_along_axis(rms,1,np.abs(euler_nop_err_test))

# calculate error rms mean Testing
ann_rms = np.nanmean(ann_rms_stream_test)
cap_rms = np.nanmean(cap_rms_stream_test)
crp_rms = np.nanmean(crp_rms_stream_test)
nop_rms = np.nanmean(nop_rms_stream_test)


# Pitch
pitchstream = euler_ann_err_test[:,0]
pitchstream_cap = euler_cap_err_test[:,0]
pitchstream_crp = euler_crp_err_test[:,0]
pitchstream_nop = euler_nop_err_test[:,0]

pitch_mean = np.nanmean(pitchstream)
pitch_95 = np.nanpercentile(pitchstream, 95)
pitch_99 = np.nanpercentile(pitchstream, 99)
pitch_99_cap = np.nanpercentile(pitchstream_cap, 99)
pitch_99_crp = np.nanpercentile(pitchstream_crp, 99)
pitch_99_nop = np.nanpercentile(pitchstream_nop, 99)
pitch_max = np.nanmax(pitchstream)

# Roll
rollstream = euler_ann_err_test[:,1]
rollstream_cap = euler_cap_err_test[:,1]
rollstream_crp = euler_crp_err_test[:,1]
rollstream_nop = euler_nop_err_test[:,1]

roll_mean = np.nanmean(rollstream)
roll_95 = np.nanpercentile(rollstream, 95)
roll_99 = np.nanpercentile(rollstream, 99)
roll_99_cap = np.nanpercentile(rollstream_cap, 99)
roll_99_crp = np.nanpercentile(rollstream_crp, 99)
roll_99_nop = np.nanpercentile(rollstream_nop, 99)
roll_max = np.nanmax(rollstream)

# Yaw
yawstream = euler_ann_err_test[:,2]
yawstream_cap = euler_cap_err_test[:,2]
yawstream_crp = euler_crp_err_test[:,2]
yawstream_nop = euler_nop_err_test[:,2]

yaw_mean = np.nanmean(yawstream)
yaw_95 = np.nanpercentile(yawstream, 95)
yaw_99 = np.nanpercentile(yawstream, 99)
yaw_99_cap = np.nanpercentile(yawstream_cap, 99)
yaw_99_crp = np.nanpercentile(yawstream_crp, 99)
yaw_99_nop = np.nanpercentile(yawstream_nop, 99)
yaw_max = np.nanmax(yawstream)

#RMS Percentile
ann_rms_99 = np.nanpercentile(ann_rms_stream_test,99)
cap_rms_99 = np.nanpercentile(cap_rms_stream_test,99)
crp_rms_99 = np.nanpercentile(crp_rms_stream_test,99)
nop_rms_99 = np.nanpercentile(nop_rms_stream_test,99)


# # RMS
# rms_mean = np.nanmean(ann_rms_stream)
# rms_95 = np.nanpercentile(ann_rms_stream, 95)
# rms_99 = np.nanpercentile(ann_rms_stream, 99)
# rms_max = np.nanmax(ann_rms_stream)
        

#print('\nMAE Error (Tested), Pitch, Roll, Yaw,')
#print('99% ANN, {:.2f}, {:.2f}, {:.2f}'.format(pitch_99, roll_99, yaw_99))

if (args.anticipation == 300):
    print('\nMetrice 2 ('+str(args.anticipation)+') : 99% RMS (Tested)')
    print('99% RMS ANN, {:.2f}'.format(ann_rms_99))

#Print the maximum data
idmaxRMS = 0;
idminRMS = 0;
for i in range(0, len(ann_rms_stream_test)-1):
      if (ann_rms_stream_test[i]==max(ann_rms_stream_test)):
          idmaxRMS = i;
      elif (ann_rms_stream_test[i]==min(ann_rms_stream_test)):
          idminRMS = i;
        
# Overhead Orientation
idx = [0,3,1,2]
hmd_orientation = quat_quat_data[:,idx]
frame_orientation = quat_predict[:,idx]

optimal_overhead = []

for i in range(0, len(hmd_orientation)):
    if (ann_rms_stream_test[i] < ann_rms_99):
         input_orientation = np.quaternion(
            hmd_orientation[i,0],
            hmd_orientation[i,1],
            hmd_orientation[i,2],
            hmd_orientation[i,3]
            )
         predict_orientation = np.quaternion(
            frame_orientation[i,0],
            frame_orientation[i,1],
            frame_orientation[i,2],
            frame_orientation[i,3]
            )
         input_projection = [
            projection_data[i,0],
            projection_data[i,1],
            projection_data[i,2],
            projection_data[i,3]
            ]
         optimal_overhead.append(calc_optimal_overhead(input_orientation, predict_orientation, input_projection))

OptimalOH_MAE_test = np.nanmean(np.abs(optimal_overhead))

"""Content Overhead with 99% Confidence"""
optimal_overhead_99 = np.nanpercentile(optimal_overhead,99)

    #Print the maximum data
idmax = 0;
idmin = 0;
for i in range(0, len(optimal_overhead)-1):
    if (optimal_overhead[i]==max(optimal_overhead)):
        idmax = i;
    elif (optimal_overhead[i]==min(optimal_overhead)):
        idmin = i;


#print('\nOuter Value of Optimal Overhead:')
#print('Overcross Error of Max OH (Pitch, Roll, Yaw): {:.2f}, {:.2f}, {:.2f}'.format(euler_ann_err_test[idmax, 0],euler_ann_err_test[idmax, 1],euler_ann_err_test[idmax, 2]))
#print('Max of Optimal OverHead : {:.2f}%'.format(max(optimal_overhead)))
#print('Overcross Error of Min OH (Pitch, Roll, Yaw): {:.2f}, {:.2f}, {:.2f}'.format(euler_ann_err_test[idmin, 0],euler_ann_err_test[idmin, 1],euler_ann_err_test[idmin, 2]))
#print('Min of Optimal OverHead : {:.2f}%'.format(min(optimal_overhead)))

if (args.anticipation == 300):
    print('\nMetrice 3  ('+str(args.anticipation)+') : Optimal Over Head of Prediction (Tested)')
    #print('Average Value of Optimal Over_head: {:.2f}%'.format(OptimalOH_MAE_test))
    print('99% Optimal Over_head: {:.2f}%'.format(optimal_overhead_99))


'''
#####   WRITING THE RESULT TO CSV  #####
'''

raw_data = {'timestamp' : timestamp_plot,
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
							'input_orientation_x' : quat_quat_data[:,3],
							'input_orientation_y' : quat_quat_data[:,1],
							'input_orientation_z' : quat_quat_data[:,2],
							'input_orientation_w' : quat_quat_data[:,0],
							'input_orientation_yaw'		: euler_o_test[:,2]*np.pi/180,
							'input_orientation_pitch'	: euler_o_test[:,0]*np.pi/180,
							'input_orientation_roll'	: euler_o_test[:,1]*np.pi/180,
							'angular_vec_x' : gyro_o_test[:,0],
							'angular_vec_y' : gyro_o_test[:,1],
							'angular_vec_z' : gyro_o_test[:,2],
							'acceleration_x' : accel_o_test[:,0],
							'acceleration_y' : accel_o_test[:,1],
							'acceleration_z' : accel_o_test[:,2],
							'magnetic_x' : magnet_o_test[:,0],
							'magnetic_y' : magnet_o_test[:,1],
							'magnetic_z' : magnet_o_test[:,2],
#                           'predicted_orientation_x' : quat_predict_cap[:,3],
#							'predicted_orientation_y' : quat_predict_cap[:,1],
#							'predicted_orientation_z' : quat_predict_cap[:,2],
#							'predicted_orientation_w' : quat_predict_cap[:,0],
#							'predicted_orientation_yaw' 	: euler_pred_cap_test[:,2]*np.pi/180,
#							'predicted_orientation_pitch' 	: euler_pred_cap_test[:,0]*np.pi/180,
#							'predicted_orientation_roll' 	: euler_pred_cap_test[:,1]*np.pi/180,
# 							'predicted_orientation_x' : quat_predict_crp[:,3],
# 							'predicted_orientation_y' : quat_predict_crp[:,1],
# 							'predicted_orientation_z' : quat_predict_crp[:,2],
# 							'predicted_orientation_w' : quat_predict_crp[:,0],
# 							'predicted_orientation_yaw' 	: euler_pred_crp_test[:,2]*np.pi/180,
# 							'predicted_orientation_pitch' 	: euler_pred_crp_test[:,0]*np.pi/180,
# 							'predicted_orientation_roll' 	: euler_pred_crp_test[:,1]*np.pi/180,
 							'predicted_orientation_x' : quat_predict[:,3],
 							'predicted_orientation_y' : quat_predict[:,1],
 							'predicted_orientation_z' : quat_predict[:,2],
 							'predicted_orientation_w' : quat_predict[:,0],
 							'predicted_orientation_yaw' 	: euler_pred_ann_test[:,2]*np.pi/180,
 							'predicted_orientation_pitch' 	: euler_pred_ann_test[:,0]*np.pi/180,
 							'predicted_orientation_roll' 	: euler_pred_ann_test[:,1]*np.pi/180,
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
export_csv = df.to_csv (r'E:\Verification Test 2021\Prediction\Prediction_' + str(outFile), index = None, header=True)
print('\nPredicted Data Saved...\n')

##SAVING EVALUATION DATA
#stored_df = pd.read_csv('EvaluationResult.csv')
#Evaluation_df = np.array(stored_df[['anti_time', 'RMS', '99% RMS',  'Average OH', '99% OH', 'MAE_Pitch', 'MAE_Roll',  'MAE_Yaw', '99% MAE_Pitch', '99% MAE_Roll','99% MAE_Yaw']], dtype=np.float32)
#s = np.array([[anticipation_time, ann_rms, ann_rms_99, OptimalOH_MAE_test, optimal_overhead_99, ann_mae_test[0], ann_mae_test[1], ann_mae_test[2], pitch_99, roll_99, yaw_99]])
#if (anticipation_time == 100) : #Starting Evaluation
#    zerArray = np.array([[0,0,0,0,0,0,0,0,0,0,0]])
#    Evaluation_df = np.concatenate((Evaluation_df,zerArray), axis=0)
#Ev=np.concatenate((Evaluation_df,s), axis=0)
#raw_data = {'anti_time' :Ev[:,0],
#            'RMS'	            :Ev[:,1],
#            '99% RMS'	        :Ev[:,2],
#            'Average OH'        :Ev[:,3],
#			'99% OH'	        :Ev[:,4],
#            'MAE_Pitch'	        :Ev[:,5],
#            'MAE_Roll'	        :Ev[:,6],
#            'MAE_Yaw'	        :Ev[:,7],
#            '99% MAE_Pitch'	    :Ev[:,8],            
#            '99% MAE_Roll'	    :Ev[:,9],
#            '99% MAE_Yaw'	    :Ev[:,10]
#			}
#df = pd.DataFrame(raw_data, columns = ['anti_time', 'RMS', '99% RMS',  'Average OH', '99% OH', 'MAE_Pitch', 'MAE_Roll',  'MAE_Yaw', '99% MAE_Pitch', '99% MAE_Roll','99% MAE_Yaw'])
#export_csv = df.to_csv (r'C:\Users\wnlab\Documents\code\MotionPred\onAirVR_Pred\Verification_Test_Result\EvaluationResult.csv', index = None, header=True)
#print('\nEvaluation Data Saved...\n')

#
#PLOT DATA
timestamp_plot = train_time_data[int(TRAIN_SIZE*data_length)+DELAY_SIZE:-2*anticipation_size]
time_offset = timestamp_plot[0]
timestamp_plot = np.array(timestamp_plot)-time_offset

timestamp_plot_train = train_time_data[DELAY_SIZE:-(int(TRAIN_SIZE*data_length)+2*anticipation_size)]
time_offset2 = timestamp_plot_train[0]
timestamp_plot_train = np.array(timestamp_plot_train)-time_offset2

#PITCH
#plt.figure()
#plt.plot(timestamp_plot_train, euler_pred_ann_train[:, 0], linewidth=1,color='red')
#plt.plot(timestamp_plot_train, euler_o_train[:, 0], linewidth=1, color='black')
#plt.legend(['ANN','Actual'])
#plt.title('Orientation Prediction TRAIN (Pitch) Anticipation Time: ' + str(anticipation_time))
#plt.grid()
#plt.xlim() 
#plt.xlabel('Time (s)')
#plt.ylabel('Orientation (deg)')

plt.figure()
plt.plot(timestamp_plot, euler_pred_ann_test[:, 0], linewidth=1,color='red')
#plt.plot(timestamp_plot, euler_pred_crp_test[:, 0], linewidth=1,color='green')
#plt.plot(timestamp_plot, euler_pred_cap_test[:, 0], linewidth=1,color='blue')
#plt.plot(timestamp_plot, euler_pred_ann_test[:, 0], linewidth=1,color='red')
plt.plot(timestamp_plot, euler_o_test[:, 0], linewidth=1, color='black')
plt.legend(['ANN', 'Actual'])
plt.title('Orientation Prediction TEST (Pitch) Anticipation Time: ' + str(anticipation_time))
plt.grid()
plt.xlim(timestamp_plot[idmaxRMS]-50, timestamp_plot[idmaxRMS]+50) 
plt.xlabel('Time (s)')
plt.ylabel('Orientation (deg)')
plt.savefig('E:\Verification Test 2021\Result Image\s' + str(inFile) + '_Pitch.png')


#ROLL
#plt.figure()
#plt.plot(timestamp_plot_train, euler_pred_ann_train[:, 1], linewidth=1,color='red')
#plt.plot(timestamp_plot_train, euler_o_train[:, 1], linewidth=1, color='black')
#plt.legend(['ANN','Actual'])
#plt.title('Orientation Prediction TRAIN (Roll) Anticipation Time: ' + str(anticipation_time))
#plt.grid()
#plt.xlim() 
#plt.xlabel('Time (s)')
#plt.ylabel('Orientation (deg)')

plt.figure()
plt.plot(timestamp_plot, euler_pred_ann_test[:, 1], linewidth=1,color='red')
#plt.plot(timestamp_plot, euler_pred_crp_test[:, 1], linewidth=1,color='green')
#plt.plot(timestamp_plot, euler_pred_cap_test[:, 1], linewidth=1,color='blue')
#plt.plot(timestamp_plot, euler_pred_ann_test[:, 1], linewidth=1,color='red')
plt.plot(timestamp_plot, euler_o_test[:, 1], linewidth=1, color='black')
plt.legend(['ANN', 'Actual'])
plt.title('Orientation Prediction TEST (Roll) Anticipation Time: ' + str(anticipation_time))
plt.grid()
plt.xlim(timestamp_plot[idmaxRMS]-50, timestamp_plot[idmaxRMS]+50) 
plt.xlabel('Time (s)')
plt.ylabel('Orientation (deg)')
plt.savefig('E:\Verification Test 2021\Result Image\s' + str(inFile) + '_Roll.png')



#YAW
#plt.figure()
#plt.plot(timestamp_plot_train, euler_pred_ann_train[:, 2], linewidth=1,color='red')
#plt.plot(timestamp_plot_train, euler_o_train[:, 2], linewidth=1, color='black')
#plt.legend(['ANN','Actual'])
#plt.title('Orientation Prediction TRAIN (Yaw) Anticipation Time: ' + str(anticipation_time))
#plt.grid()
#plt.xlim() 
#plt.xlabel('Time (s)')
#plt.ylabel('Orientation (deg)')

plt.figure()
plt.plot(timestamp_plot, euler_pred_ann_test[:, 2], linewidth=1,color='red')
#plt.plot(timestamp_plot, euler_pred_crp_test[:, 2], linewidth=1,color='green')
#plt.plot(timestamp_plot, euler_pred_cap_test[:, 2], linewidth=1,color='blue')
#plt.plot(timestamp_plot, euler_pred_nop_test[:, 2], linewidth=1,color='red')
plt.plot(timestamp_plot, euler_o_test[:, 2], linewidth=1, color='black')
plt.legend(['ANN', 'Actual'])
plt.title('Orientation Prediction TEST (Yaw) Anticipation Time: ' + str(anticipation_time))
plt.grid()
#plt.xlim(timestamp_plot[idmaxRMS]-50, timestamp_plot[idmaxRMS]+50) 
plt.xlabel('Time (s)')
plt.ylabel('Orientation (deg)')
plt.savefig('E:\Verification Test 2021\Result Image\s' + str(inFile) + '_Yaw.png')

#plt.figure()
#plt.plot(timestamp_plot, optimal_overhead, linewidth=1,color='blue')
#plt.legend(['Optimal Overhead','Actual'])
#plt.title('Value of Optimal Overhead Data Anticipation Time: ' + str(anticipation_time))
#plt.grid()
#plt.xlim(timestamp_plot[idmaxRMS]-50, timestamp_plot[idmaxRMS]+50) #Find boundary Value
#plt.xlabel('Time (s)')
#plt.ylabel('Percentage of Optimal OverHead Projection (%)')
#plt.show(block=False)
#

# plt.figure()
# x = np.sort(optimal_overhead)
# y = np.arange(1, len(x)+1)/len(x)
# _ = plt.plot(x, y, marker='.', linestyle='none')
# _ = plt.xlabel('Percentage of Optimal OverHead Projection (%)')
# _ = plt.ylabel('likehood of Occurance')
# plt.title('CDF of Optimal OverHead Anticipation Time: ' + str(anticipation_time))
# plt.margins(0.02)

# plt.show()