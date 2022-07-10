import numpy as np
import matplotlib.pyplot as plt


def smooth(scalars, weight=0.6):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def analyze():
    result = np.load("../MAML_result/training_info.npy")
    inner_result = np.load("../MAML_result/inner_returns.npy")
    x = []
    inner_x=[]
    returns = []
    inner_returns=[]
    a_loss = []
    q_loss = []
    meta_q_loss = []
    for ele in result:
        x.append(ele[0])
        returns.append(ele[1])
        # a_loss.append(ele[2])
        # q_loss.append(ele[3])
        # meta_q_loss.append(ele[4])
    for ele in inner_result:
        inner_x.append(ele[0])
        inner_returns.append(ele[1])
    returns = smooth(returns)
    inner_returns=smooth(inner_returns)
    print(inner_result)
    fig, ax1 = plt.subplots(1, 2, figsize=(12, 4))
    ax1[0].plot(x, returns)
   
    # ax1[2].plot(x, q_loss)
    # ax1[3].plot(x, meta_q_loss)
    ax1[0].set_xlabel("Meta Update iterations")
    
    # ax1[2].set_xlabel("Meta Update iterations")
    # ax1[3].set_xlabel("Meta Update iterations")
    ax1[0].set_ylabel('validation returns')
    
    # ax1[2].set_ylabel('inner critic loss')
    # ax1[3].set_ylabel('meta critic loss')

    ax1[1].plot(range(len(inner_x)), inner_returns)
    ax1[1].set_xlabel("inner Update iterations")
    ax1[1].set_ylabel('inner returns')

    plt.show()

def smooth(scalars, weight=0.6):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

if __name__ == "__main__":
    analyze()