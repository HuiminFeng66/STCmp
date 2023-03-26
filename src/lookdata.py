import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# data=np.load('../data/evaluation_datasets/dna/nanopore.npy')
# print('shape of nanopore_.npy\n',data.shape)   # (1167877,)
# print(data.size)                               # 1167877

# data=np.load('../data/evaluation_datasets/dna/nanopore_train.npy')
# print('shape of nanopore_train.npy\n',data.shape)                # (389292,)
# print(data.size)                                                 # 389292
# print(data)

# data=np.load('../data/evaluation_datasets/dna/nanopore_test.npy')
# print('shape of nanopore_test.npy\n',data.shape)                           # (389293,)
# print(data.size)                                                           #  389293

# data=np.load('../data/evaluation_datasets/dna/nanopore_valid.npy')
# print('shape of nanopore_valid.npy\n',data.shape)                          # (389292,)
# print(data.size)                                                         # 389292
# print(data.__len__())                                                    # 389292



# data=np.load('../data/evaluation_datasets/dna/nanopore_valid.npy')
# print('shape of nanopore_valid.npy\n',data.shape)
# print(data.size)


# data=np.load('../data/evaluation_datasets/acc/Watch_accelerometer_combined.npy')
# print('shape of Watch_accelerometer_combined.npy:',data.shape)               # (3,3540962)
# print('data.size:',data.size)                                                # 10622886
# print('data.shape[0]:',data.shape[0])                                        # 3
# print('data.shape[1]:',data.shape[1])                                        # 3540962
# X=np.transpose(data)
# print('X.shape:',X.shape)                 # (3540962,3)
# print('X.__len__:',X.__len__())             # 3540962



# data=np.load('../data/evaluation_datasets/gas/HT_Sensor_dataset_combined_test.npy')
# print('shape of HT_Sensor_dataset_combined_test.npy:',data.shape)           # (278698,8)
# print('data.size:',data.size)                     # 2229584
# print('data.shape[0]:',data.shape[0])               # 278698
# print('data.shape[1]:',data.shape[1])               # 8
# print('data:\n',data)
# X=np.transpose(data)
# print('X.shape:',X.shape)                 # (8,278698)
# print('X.__len__:',X.__len__())                 # 8



# data=np.load('../data/evaluation_datasets/gas/HT_Sensor_dataset_combined.npy')
# print('shape of HT_Sensor_dataset_combined.npy:',data.shape)           # (928991, 8)
# print('data.size:',data.size)                     # 7431928
# print('data.shape[0]:',data.shape[0])               # 928991
# print('data.shape[1]:',data.shape[1])               # 8
# print('data:\n',data)
# X=np.transpose(data)
# print('X.shape:',X.shape)                 # (8, 928991)
# print('X.__len__:',X.__len__())                 # 8



# data=np.load('../data/evaluation_datasets/PPG/S2_BVP.npy')
# print('shape of S2_BVP.npy:',data.shape)    # (503944,)

# print(data[:3,:3])





####################################################
"""
# # from txt to npy
# data=np.genfromtxt('../data/evaluation_datasets/electricity/electricity.txt',delimiter=',')
# print(data.shape)    # (26304,321)
data=StandardScaler().fit_transform(data)
print(data)
data=np.array(data,dtype=np.float32)
# print(data)
# np.save('../data/evaluation_datasets/electricity/electricity_combined.npy',data)

"""

# data=np.genfromtxt('../data/evaluation_datasets/sml/sml1.txt',delimiter='')
# print(data.shape)
# print(data[0,:])
# data=np.array(data,dtype=np.float32)
# data=np.delete(data,[0,1,4,11,14,15,23,20,19,18],axis=1)
# print(data[0,:])
# print(data.shape)   # (2764, 14)
# # data=StandardScaler().fit_transform(data)
# print(data)
# np.save('../data/evaluation_datasets/sml/sml1.npy',data)


"""
"""
# data=np.genfromtxt('../data/evaluation_datasets/sml/sml2.txt',delimiter='')
# print(data.shape)      # (1373, 24)
# print(data[0,:])
# data=np.array(data,dtype=np.float32)
# data=np.delete(data,[0,1,4,11,14,15,23,20,19,18],axis=1)
# print(data[0,:])
# print(data.shape)   # (1373, 14)
# data=StandardScaler().fit_transform(data)
# print(data)
# np.save('../data/evaluation_datasets/sml/sml2.npy',data)



# data=np.genfromtxt('../data/evaluation_datasets/solar/solar.txt',delimiter=',')
# print(data.shape)    # (52560, 137)
# print('data',data)
# data=StandardScaler().fit_transform(data)
# print('data',data)
# data=np.array(data,dtype=np.float32)
# print(data)
# np.save('../data/evaluation_datasets/solar/solar.npy',data)


# data=np.genfromtxt('../data/evaluation_datasets/electricity/electricity.txt',delimiter=',')
# print(data.shape)    # (26304,321)
# print(data)
# data=StandardScaler().fit_transform(data)    ###############  Standard
# print(data)
# data=np.array(data,dtype=np.float64)
# print(data)
# np.save('../data/evaluation_datasets/electricity/electricity.npy',data)


data=np.genfromtxt('../data/evaluation_datasets/traffic/traffic.txt',delimiter=',')
print(data.shape)     # (17544,862)-------->250
print('data:',data)
# data=StandardScaler().fit_transform(data)
data=data[:,:100]            ###########
print('data:',data)
data=np.asarray(data,dtype=np.float64)
print(data.shape)
np.save('../data/evaluation_datasets/traffic/traffic.npy',data)




# data=np.genfromtxt('../data/evaluation_datasets/exchangerate/exchangerate.txt')
# print(data.shape)     #  (7588, 8)
# print('data:',data)
# data=np.asarray(data,dtype=np.float64)
# print(data.shape)   # (7588, 8)
# np.save('../data/evaluation_datasets/exchangerate/exchangerate.npy',data)



# data=np.genfromtxt('../data/evaluation_datasets/acc4/Watch_accelerometer_combined4.txt')
# print(data.shape)
# print('data:',data)
# np.save('../data/evaluation_datasets/acc4/Watch_accelerometer_combined4.npy',data)



# data=np.genfromtxt('../data/evaluation_datasets/gas/HT_Sensor_dataset_combined.txt')
# print(data.shape)     #  (928991, 8)
# print('data:',data)
# data=StandardScaler().fit_transform(data)
# print('data:',data)
# data=np.asarray(data,dtype=np.float32)
# print(data.shape)   # (928991, 8)
# np.save('../data/evaluation_datasets/gas/HT_Sensor_dataset_combined.npy',data)




#################################################
"""
# # from npy to txt
# data=np.load('../data/evaluation_datasets/gas/HT_Sensor_dataset_combined_test.npy')
# np.savetxt('../data/evaluation_datasets/gas/HT_Sensor_dataset_combined_test.txt',data)

"""


# data=np.load('../data/evaluation_datasets/pow/active_power.npy')
# np.savetxt('../data/evaluation_datasets/pow/active_power.txt',data)

# data=np.load('../data/evaluation_datasets/acc/Watch_accelerometer_combined.npy')
# data=np.transpose(data)
# print(data.shape)
# print(data)
# col4=data[:,1]+data[:,2]
# data=np.insert(data,3,col4,axis=1)
# print(data.shape)
# print(data)
# np.savetxt('../data/evaluation_datasets/acc/Watch_accelerometer_combined4.txt',data)




# data=np.load('../data/evaluation_datasets/dna/nanopore.npy')
# np.savetxt('../data/evaluation_datasets/dna/nanopore.txt',data)

# data=np.load('../data/evaluation_datasets/exchangerate/exchangerate.npy')
# np.savetxt('../data/evaluation_datasets/exchangerate/exchangerate.txt',data)

# data=np.load('../data/evaluation_datasets/gas/HT_Sensor_dataset_combined.npy')
# np.savetxt('../data/evaluation_datasets/gas/HT_Sensor_dataset_combined.txt',data)

# data=np.load('../data/evaluation_datasets/gyr/Watch_gyroscope_combined.npy')
# np.savetxt('../data/evaluation_datasets/gyr/Watch_gyroscope_combined.txt',data)

# data=np.load('../data/evaluation_datasets/PPG/S2_BVP.npy')
# np.savetxt('../data/evaluation_datasets/PPG/S2_BVP.txt',data)

# data=np.load('../data/evaluation_datasets/nasdaq/nasdaq100_combined.npy')
# np.savetxt('../data/evaluation_datasets/nasdaq/nasdaq100_combined.txt',data)




###########################################################################
"""
#  # from csv to npy
data=pd.read_csv('../data/evaluation_datasets/nasdaq/nasdaq100_padding.csv').values
print(data.shape)      # (40560, 82)
data=StandardScaler().fit_transform(data)
print(data)
data=np.array(data,dtype=np.float32)
print(data.shape)       # (40560, 82)
np.save('../data/evaluation_datasets/nasdaq/nasdaq100_combined.npy',data)
"""


# data=pd.read_csv('../data/evaluation_datasets/nasdaq/nasdaq100_padding.csv').values
# print(data.shape)      # (40560, 82)
# # data=StandardScaler().fit_transform(data)
# print(data)
# data=np.array(data,dtype=np.float64)
# print(data.shape)       # (40560, 82)
# np.save('../data/evaluation_datasets/nasdaq/nasdaq.npy',data)



# data=pd.read_csv('../data/evaluation_datasets/energy/energy.csv').values
# print('data.shape:',data.shape)      #(19735, 29)
# print('data:',data)
# print(data[0,:])
# data=np.delete(data,[0,1,2,4,12,14,16,23,25,27,28],axis=1)
# print('data.shape:',data.shape)    # (19735, 18)
# print('data:',data)
# print(data[0,:])
# # data=StandardScaler().fit_transform(data)
# print(data)
# data=np.array(data,dtype=np.float32)
# print('data.shape:',data.shape)       # (19735, 26)
# print(data)
# np.save('../data/evaluation_datasets/energy/energy.npy',data)



# data=pd.read_csv('../data/evaluation_datasets/webtraffic/webtraffic.csv').values
# print(data.shape)   # (145063, 18)
# data=data[:,2:]
# print('data',data)
# data=StandardScaler().fit_transform(data)
# print('data',data)
# data=np.array(data,dtype=np.float32)
# print(data.shape)   #  (145063, 16)
# np.save('../data/evaluation_datasets/webtraffic/webtraffic.npy',data)



# data=pd.read_csv('../data/evaluation_datasets/temperature/temperature01.csv').values
# print(data.shape)      # (295719, 20)
# data=data[:,2:]
# print(data)
# print(data.shape)     # (295719, 18)
# data=StandardScaler().fit_transform(data)
# print(data)
# data=np.array(data,dtype=np.float64)
# print(data.shape)      #  (295719, 18)
# np.save('../data/evaluation_datasets/temperature/temperature01.npy', data)




# data=pd.read_csv('../data/evaluation_datasets/webtraffic/webtraffic.csv',delimiter=',')
# print(data.shape)   # (145063, 18)
# data=np.asarray(data,dtype=np.float32)
# print(data.shape)   #  (145063, 18)
# data=data[1:145056,2:]
# print(data.shape)    #  (145055, 16)
# np.save('../data/evaluation_datasets/webtraffic/webtraffic_combined.npy',data)


# data=pd.read_csv('../data/evaluation_datasets/temperature/temperature01.csv',delimiter=',')
# print(data.shape)   # (295719, 20)
# data=np.asarray(data,dtype=np.float32)
# print(data.shape)   #  (295719, 20)
# # np.save('../data/evaluation_datasets/temperature/temperature01.npy',data)

# data=pd.read_csv('../data/evaluation_datasets/temperature/temperature02.csv',delimiter=',')
# print(data.shape)   #   (295516, 20)
# data=np.asarray(data,dtype=np.float32)
# print(data.shape)   #   (295516, 20)
# np.save('../data/evaluation_datasets/temperature/temperature02.npy',data)






#########################################
"""
# # read data from dat file
"""
# read data from dat file

# file=open('../data/evaluation_datasets/gas/HT_Sensor_dataset/HT_Sensor_dataset.dat')
# data=file.readlines()
# data=np.array(data)
# print(data.shape)
# print(data[0])
# print(data[1])
# file.close()