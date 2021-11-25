import numpy as np
import math
import random as rn
from collections import defaultdict

def robust_overfilling(input_orientation, prediction, input_projection, offset = 1.1, fixed_param = 1.3):
    #Get Input Projection Distance for each side in x,y coordinate
    IPDx = input_projection[2]-input_projection[0]
    IPDy = input_projection[1]-input_projection[3]

    #Define projection distance to the user
    h = 1

    #Get the corner point distance to the rotation center for each side in x,y coordinate
    rx = np.sqrt(h**2+1/4*IPDx**2)
    ry = np.sqrt(h**2+1/4*IPDy**2)

    #Get initial input angle to the rotational center
    input_anglex = np.arctan(IPDx/(2*h))
    input_angley = np.arctan(IPDy/(2*h))

    #Get user's direction based on prediction motion
    pitch_diff = (prediction[0]-input_orientation[0])
    roll_diff = (prediction[1]-input_orientation[1])
    yaw_diff = (prediction[2]-input_orientation[2])
    # print("pitch: %s"%pitch_diff)
    # print("roll: %s"%roll_diff)
    # print("yaw: %s"%yaw_diff)

    #Calculate predicted margin based on translation movement
    x_r = max(input_projection[2],rx*abs(np.sin(input_anglex-yaw_diff)))
    x_l = min(input_projection[0],-rx*abs(np.sin(input_anglex+yaw_diff)))
    y_t = max(input_projection[1],ry*abs(np.sin(input_angley+pitch_diff)))
    y_b = min(input_projection[3],-ry*abs(np.sin(input_angley-pitch_diff)))

    #Calculate predicted margin based on rotational movement
    x_rr = rx*abs(np.sin(input_anglex+abs(roll_diff)))
    x_ll = -rx*abs(np.sin(input_anglex+abs(roll_diff)))
    y_tt = ry*abs(np.sin(input_angley+abs(roll_diff)))
    y_bb = -ry*abs(np.sin(input_angley+abs(roll_diff)))

    #Calculate final movement
    p_r = x_r+x_rr-IPDx/2
    p_l = x_l+x_ll+IPDx/2
    p_t = y_t+y_tt-IPDy/2
    p_b = y_b+y_bb+IPDy/2
    # print(p_r, p_l, p_t, p_b)

    # x = (p_l, 0, p_r, 0)
    # y = (0, p_t, 0, p_b)

    # x = (-0.8, -0.8, 0.8, 0.8)
    # y = (-0.72, 0.72, 0.72, -0.72)

    x = (p_l, p_l, p_r, p_r)
    y = (p_b, p_t, p_t, p_b)

    # input_projection = [-0.800, 0.720, 0.800, -0.720]

    return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]

    # area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    # return area

def step(ind, action, subN, ue, band, all_sub, v_array, prev_a_array, current_a_array):
    loc = ind                   # locator of the indices of the channel gains h
    sub = np.array([action])    # allocated sub-carriers
    sorter = []                 # this sorts out the MD
    n = ue                      # number of users
    b = band

    for i in range(0, n, 1):
        sorter.append(i)

    p = 100             # 100 mW / 20 dBm
    fn = 1 * 10**9
    fc = 10 * 10**9
    cn = 1 * 10**3
    no = 1 * 10**-10    # 10^-10 mW / 0.1 pW / -100 dBm
    noise = no          # the total noise of the denominator of the remote rate is initialized to power noise
    e1 = math.log(2)
    rl = fn / cn        # initialize the local data rate
    wi = np.array([1.5 if md % 2 == 1 else 1 for md in range(n)])

    d = 0.1
    v = v_array                          # velocity of user fixation (rad/s)
    L = 1600 * 1440 	                 # viewport pixels 1600x1440
    # k = 10**2                            # pixels coefficient
    lmd = 1                              # L/r < d coefficient
    Pl = 10 * 10**-5                     # local computing power
    Po = 0.1 * 10**-5	                 # offload computing power
    k_coef = [100, 200, 300, 400, 500]      # range [10,50,100,200,300]ms

    k = [k_coef[action[i]] for i in range(0,len(action)) if i%2]               # playout delay
    sub = np.array([action[i] for i in range(0,len(action)) if not i%2])    # allocated sub-carriers

    input_projection = [-0.800, 0.720, 0.800, -0.720]

    out_dict = defaultdict(list)

    i = 0

    for s, n in zip(sub, sorter):  # outer loop to compute the system wide weighted sum data rate
        prediction = [current_a_array[i], current_a_array[i+1], current_a_array[i+2]]
        input_orientation = [prev_a_array[i], prev_a_array[i+1], prev_a_array[i+2]]
        opt_coor = robust_overfilling(input_orientation , prediction, input_projection)
        if s == 0:
            r = wi[n]*rl
            e = Pl * L
            # print('e loc : %s' % e)
            ovf = 0
            d_bound = 0
            Li = L
            bb_coor = np.abs(np.abs(input_projection) - np.abs(opt_coor))
            bb = np.sum(bb_coor)
            # print('Local bb : %s '%bb)
        else:
            # Energy Consumption Calculation
            cur_sub = np.array(all_sub[s-1])  # we use s-1 because our sub-carrier array counts from zero
            h = cur_sub[loc, :]
            s_power = h[n] * p  # signal power computed. this is the numerator of the signal_to_noise
            for c, m in zip(sub, sorter):  # this loop computes noise by first sorting usres sharing same sub-carriers
                if c == 0:
                    noise = noise + 0.0
                else:
                    if s == c and n != m and h[m] < h[n]:  # no is augmented only if there are equal sub-carriers,
                        # but not the current sub-carrier and bijection order checked
                        # noise = noise + ((h[m]*p)/10000)
                        noise = noise + (h[m] * p)
                        # here I check the cases when users have the same sub-carriers
                    else:
                        noise = noise+0.0
            signal_to_noise = s_power/noise
            e2 = math.log(1+signal_to_noise)
            ro = b * e2/e1
            r = wi[n]*ro

            ovf = k[n] * v[n]**2 * d**2
            Li = L * (1 + ovf)
            d_bound = (Li / r) - d

            e = Po * Li + lmd * d_bound

            # Black border Calculation
            alg_coor = np.array(input_projection) * (1 + ovf)
            bb_coor = np.abs(opt_coor) - np.abs(alg_coor)
            bb_coor = np.clip(bb_coor, 0, None)
            bb = np.sum(bb_coor)
            # print('Offload bb : %s '%bb)

            opt_perim = (opt_coor[2] - opt_coor[0] + opt_coor[1] - opt_coor[3]) * 2
            alg_perim = (alg_coor[2] - alg_coor[0] + alg_coor[1] - alg_coor[3]) * 2
            noovf_perim = (input_projection[2] - input_projection[0] + input_projection[1] - input_projection[3]) * 2
            # print('k user %s: %s'%(n+1,k[n]))
            # print('v : %s'%v[n])
            # print('Optimal : %s'%opt_coor)
            # print('Algorithm : %s'%alg_coor)
        i+=1
        out_dict['x'].append(s)         
        out_dict['rate'].append(r)      # transmission rate
        out_dict['energy'].append(e)      # energy consumption
        out_dict['overfill'].append(ovf)
        out_dict['k_coef'].append(k[n])
        # out_dict['delay_bound'].append(d_bound)
        # out_dict['optimal_perim'].append(opt_perim)
        # out_dict['algorithm_perim'].append(alg_perim)
        # out_dict['nooverfill_perim'].append(noovf_perim)
        out_dict['blackborder'].append(bb)
    return out_dict
