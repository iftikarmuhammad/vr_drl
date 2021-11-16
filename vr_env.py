import numpy as np
import math
import random as rn
from collections import defaultdict

def step(ind, action, subN, ue, band, all_sub, v_array):
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

    v = v_array                         # velocity of user fixation (rad/s)
    L = 1600 * 1440 	                # viewport pixels 1600x1440
    k = 10**2                           # pixels coefficient
    lmd = 1                             # L/r < d coefficient
    Pl = 10 * 10**-5                    # local computing power
    Po = 0.1 * 10**-5	                # offload computing power
    dly = [0.01, 0.05, 0.1, 0.2, 0.3]   # range [10,50,100,200,300]ms  

    d = [dly[action[i]] for i in range(0,len(action)) if i%2]               # playout delay
    sub = np.array([action[i] for i in range(0,len(action)) if not i%2])    # allocated sub-carriers

    out_dict = defaultdict(list)

    for s, n in zip(sub, sorter):  # outer loop to compute the system wide weighted sum data rate
        if s == 0:
            r = wi[n]*rl
            e = Pl * L
            ovf = 0
            d_bound = 0
        else:
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
            ovf = k * v[n]**2 * d[n]**2
            d_bound = (L * (1 + ovf) / r) - d[n]
            e = Po * L * (1 + ovf) + lmd * d_bound
        out_dict['x'].append(s)         
        out_dict['rate'].append(r)      # transmission rate
        out_dict['energy'].append(e)      # energy consumption
        out_dict['overfill'].append(ovf)
        out_dict['delay'].append(d[n])
        out_dict['delay_bound'].append(d_bound)
    return out_dict
