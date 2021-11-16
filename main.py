import os
import pandas as pd
from datetime import datetime
# from EXHAUSTIVE_new import compute_q as cq_exh
# from MC_NOMA_TF2 import compute_q as cq_mcnoma
# from RANDOM import compute_q as cq_rand
# from LOCAL import compute_q as cq_lc
from vr_dqn import run as run_dqn


if __name__ == "__main__":

    channels = [1, 2, 3]
    # channels = [1]
    averagedRate_exh = []
    averagedRate_rand = []
    averagedRate_lc = []
    averagedRate_mcnoma = []
    averagedRate_dqn = []

    avgReward_dqn = []

    avgRateUE_exh = []
    avgRateUE_rand = []
    avgRateUE_lc = []
    avgRateUE_mcnoma = []
    avgRateUE_dqn = []

    avgEnergy_dqn = []
    avgEnergyUE_dqn = []

    RESULT_DIR = os.getcwd() + "/result/" + datetime.now().strftime("%y%m%d_%H%M") + "/"
    print(RESULT_DIR)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    for ch in channels:
        print(ch)
        ue = 3  # number of users
        n = 300  # time frame
        # q_i_exh, q_ue_exh = cq_exh(n, ch, ue)  # for exhaustive
        # q_i_rand, q_ue_rand = cq_rand(n, ch, ue)  # for random
        # q_i_lc, q_ue_lc = cq_lc(ue) # for local
        # q_i_mcnoma, q_ue_mcnoma = cq_mcnoma(n, k_level, delta, ch, ue)  # for MC-NOMA
        e_i_dqn, e_ue_dqn = run_dqn(n, ch, ue, True, RESULT_DIR)

        # avgReward_dqn.append(rwd_dqn)

        # averagedRate_exh.append(q_i_exh)
        # averagedRate_rand.append(q_i_rand)
        # averagedRate_lc.append(q_i_lc)
        # averagedRate_mcnoma.append(q_i_mcnoma)
        # averagedRate_dqn.append(q_i_dqn)
        avgEnergy_dqn.append(e_i_dqn)

        # avgRateUE_exh.extend(q_ue_exh)
        # avgRateUE_rand.extend(q_ue_rand)
        # avgRateUE_lc.extend(q_ue_lc)
        # avgRateUE_mcnoma.extend(q_ue_mcnoma)
        # avgRateUE_dqn.extend(q_ue_dqn)
        avgEnergyUE_dqn.extend(e_ue_dqn)


    
    # df = pd.DataFrame(averagedRate_exh, columns=["exhaustive"])
    # df.to_csv(log_path + "MovingAverageReward_exh_energy.csv", index=False)

    # df = pd.DataFrame(averagedRate_rand, columns=["rand"])
    # df.to_csv(log_path + "MovingAverageReward_rand_energy.csv", index=False)

    # df = pd.DataFrame(averagedRate_lc, columns=["local"])
    # df.to_csv(log_path + "MovingAverageRate_lc_2.csv", index=False)

    # df = pd.DataFrame(averagedRate_mcnoma, columns=["mcnoma"])
    # df.to_csv(log_path + "MovingAverageRewardMCNOMA_energy.csv", index=False)

    # df = pd.DataFrame(avgReward_dqn, columns=["dqn"])
    # df.to_csv(log_path + "MovingAverageRewardDQN_energy.csv", index=False)

    # df = pd.DataFrame(averagedRate_dqn, columns=["dqn"])
    # df.to_csv(log_path + "MovingAverageRateDQN_energy.csv", index=False)

    df = pd.DataFrame(avgEnergy_dqn, columns=["dqn"])
    df.to_csv(RESULT_DIR + "MovingAverageEnergyDQN.csv", index=False)

    # Saving per-UE computation rates for CDF plotting
    # df = pd.DataFrame(avgRateUE_exh, columns=["exhaustive"])
    # df.to_csv(log_path + "UERateEXH_test2.csv", index=False)

    # df = pd.DataFrame(avgRateUE_rand, columns=["rand"])
    # df.to_csv(log_path + "UERateRAND_test.csv", index=False)

    # df = pd.DataFrame(avgRateUE_lc, columns=["local"])
    # df.to_csv(log_path + "UERateLOCAL_test2.csv", index=False)

    # df = pd.DataFrame(avgRateUE_mcnoma, columns=["mcnoma"])
    # df.to_csv(log_path + "UERateMCNOMA_test3.csv", index=False)

    df = pd.DataFrame(avgEnergyUE_dqn, columns=["dqn"])
    df.to_csv(RESULT_DIR + "UEEnergyDQN.csv", index=False)

    # df = pd.DataFrame(avgRateUE_dqn, columns=["dqn"])
    # df.to_csv(log_path + "UERateDQN_energy.csv", index=False)
