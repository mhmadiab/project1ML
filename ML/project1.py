import numpy as np

sample_list=[10,20,30,40,50]

sample_numpy_1d_array=np.array(sample_list)
sample_numpy_2d_array=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
new_array=sample_numpy_2d_array.reshape(2,6)



import pandas as pd
sample=pd.Series([10,20,30,40],['a','b','c','d'])

sample2=pd.DataFrame([[10,20,30],[40,50,60],[70,80,90]],['a','b','c'],['A','B','C'])

numpy_new=sample2.iloc[0:2 , 1:3].values

print(sample2[['C','A']]>50)

f=pd.read_csv("project1.csv")

f.head()