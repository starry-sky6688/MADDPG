import numpy as np
import matplotlib.pyplot as plt


def analyze():
    result = np.load("./MAML_result/training_info.npy")
    x = []
    returns = []
    a_loss = []
    q_loss = []
    meta_q_loss = []
    for ele in result:
        x.append(ele[0])
        returns.append(ele[1])
        # a_loss.append(ele[2])
        # q_loss.append(ele[3])
        # meta_q_loss.append(ele[4])
    fig, ax1 = plt.subplots(1, 2, figsize=(12, 4))
    ax1[0].plot(x, returns)
    # ax1[1].plot(x, a_loss)
    # ax1[2].plot(x, q_loss)
    # ax1[3].plot(x, meta_q_loss)
    ax1[0].set_xlabel("Meta Update iterations")
    # ax1[1].set_xlabel("Meta Update iterations")
    # ax1[2].set_xlabel("Meta Update iterations")
    # ax1[3].set_xlabel("Meta Update iterations")
    ax1[0].set_ylabel('validation returns')
    # ax1[1].set_ylabel('inner actor loss')
    # ax1[2].set_ylabel('inner critic loss')
    # ax1[3].set_ylabel('meta critic loss')
    plt.show()


if __name__ == "__main__":
    analyze()
