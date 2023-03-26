from pandas import read_csv
from matplotlib import pyplot as plt
import numpy as np



# # load dataset
# data=np.load('../data/evaluation_datasets/gyr/Watch_gyroscope_combined.npy')
# n=data.shape[0]
# # specify columns to plot
# groups=[0,1,2]
# i=1
#
# # plot each row
# plt.figure()
# for group in groups:
#     plt.subplot(len(groups),1,i)
#     plt.plot(data[group,:])
#     i+=1
# plt.show()



# data=np.load('../data/evaluation_datasets/acc/Watch_accelerometer_combined.npy')
# n=data.shape[0]
# # specify columns to plot
# groups=[0,1,2]
# i=1
#
# # plot each row
# plt.figure()
# for group in groups:
#     plt.subplot(len(groups),1,i)
#     plt.plot(data[group,:])
#     i+=1
# plt.show()



data=np.load('../data/evaluation_datasets/gyr/Watch_gyroscope_combined.npy')
print(data.shape)
n=data.shape[0]
# specify columns to plot
groups=[0,1,2]
i=1

# plot each row
plt.figure()
for group in groups:
    plt.subplot(len(groups),1,i)
    plt.plot(data[group,3000:5000])
    i+=1
plt.show()



# data=np.load('../data/evaluation_datasets/electricity/electricity.npy')
# # print(data.shape)      # (26304, 321)
# data=np.transpose(data)
# n=10
# # specify columns to plot
# groups=[0,1,2,3,4,5,6,7,8,9]
# i=1
#
# # plot each row
# plt.figure()
# for group in groups:
#     plt.subplot(len(groups),1,i)
#     plt.plot(data[group,1000:1300])
#     i+=1
# plt.show()