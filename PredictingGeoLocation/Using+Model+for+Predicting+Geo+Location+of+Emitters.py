
# coding: utf-8

# # Introduction

# In[1]:


# import libraries
import model_param
import sys
import os
import math
import numpy as np
np.set_printoptions(precision=3)


# # Input file

# In[2]:


data_file = ''

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if is_interactive():
    data_file = 'Calib_1.str'
else:
    if ( len(sys.argv) < 2 ):
        print('\nSupply WinProp Raydata file in ASCII format.\n\t\tUsage ', 
              sys.argv[0], ': Raydata_file', '\n')
        exit()
    data_file = str(sys.argv[1])

if ( not os.path.isfile(data_file) ):
    print("\nPlease check the filename", data_file, " and try again.\n")
    exit()


# # Parser
# 
# This section is a module for parsing incoming ascii file (in str format)
# 
# It returns a numerical matrix with 5 columns as shown below:
# 
# 
# | Delay(ns)  | Strength(dBuV/m)  |  X-coordinate |  Y-coordinate |  Z-coordinate |
# |------------|-------------------|---------------|---------------|---------------|
# | 2349.565   |      58.71        |     396.00    |     580.00    |    515.46     |
# |  926.787   |      74.92        |     736.00    |     980.00    |    515.06     |
# |  ......    |      .....        |     .....     |     ......    |    ......     |
# |  ......    |      .....        |     .....     |     ......    |    ......     |
# |  808.820   |      76.04        |     756.00    |     980.00    |    517.05     |

# In[3]:


# Description:
#   Parse incoming ascii file (in str format)
#
# returns:
#  numpy matrix with 5 columns as follows:
#
#   Delay    FieldStrength   X-coordinate  Y-coordinate  Z-coordinate
#   [ns]       [dBuV/m]       
#   
#
def parse_input(filename):

    point = []
    dataset = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            columns = line.split()

            # Do not look at lines that are empty, 
            # commented or have less than 4 cols.
            if ( len(columns) == 0 or columns[0] == '*' or len(columns) < 4):
                continue ;

            if ( columns[0] == 'PATH' and len(point)==3 ):
                # Parameters: delay (ns) , power (dBuV/m)
                record = [float(num) for num in columns[1:3] + point[:]]
                dataset += record

            if ( columns[0] == 'POINT' ):
                # Co-ordinates x, y, z
                point = columns[1:4]

    return np.asarray(dataset).reshape(-1,5)


# # Prepare regression dataset

# In[4]:


# Format: Delay Power X Y Z
dataset = parse_input(data_file)


# # Normalize Dataset

# In[5]:


# normalize the dataset
data_mean = np.min(dataset, axis=0)*0.99
data_std  = (np.max(dataset, axis=0)-np.min(dataset, axis=0));

def normalize(d, mean, std):
    return (d - mean) / std

def denormalize(d, mean, std):
    return (d * std) + mean

dataset = normalize(dataset, data_mean, data_std)

print('Dataset properties:')
print('Dataset size', dataset.shape)
for i in range(dataset.shape[1]):
    print('Min=', np.min(dataset[:,i]),  'Max=',np.max(dataset[:,i]), 
          'Mean=',np.mean(dataset[:,i]), 'sigma=', np.std(dataset[:,i]))


# # Predict Location

# In[6]:


def locationModel(d, p):
    x = np.sqrt(model_param.idwx*d*d + model_param.ipwx*(p*p-p) + model_param.ibx)
    y = np.sqrt(model_param.idwy*d*d + model_param.ipwy*(p*p-p) + model_param.iby)
    z = np.sqrt(model_param.idwz*d*d + model_param.ipwz*(p*p-p) + model_param.ibz)

    return x, y, z

def predictLocation(delay, power):
    d = normalize(delay, model_param.data_mean[0], model_param.data_std[0])
    p = normalize(power, model_param.data_mean[1], model_param.data_std[1])
    
    x,y,z = locationModel(d, p)
    
    x = denormalize(x, model_param.data_mean[2], model_param.data_std[2])
    y = denormalize(y, model_param.data_mean[3], model_param.data_std[3])
    z = denormalize(z, model_param.data_mean[4], model_param.data_std[4])

    return x, y, z


# # Predict Location

# In[7]:


testset = parse_input(data_file)
print ('# Path_Delay Field Strength : [X Y X] => predicted_X predicted_Y predicted_Z')
for i in range(testset.shape[0]):
    print("{:.1f}".format(testset[i,0]), "{:.1f}".format(testset[i,1]), 
          ':', testset[i, 2:], ' => ', '[ {:.3f} {:.3f} {:.3f}]'.format(*predictLocation(testset[i, 0], testset[i, 1])) )

