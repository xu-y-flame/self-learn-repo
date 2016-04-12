# coding=utf-8
'''
Created on 2016年4月3日

@author: xuyan

'''
import numpy as np

arr1 = np.array([[2, 2, 3], [3, 4, 4], [4, 5, 5]])
print arr1.flatten()
print arr1

print arr1.ravel()
print arr1                                                                                                                                                                                                                                                                                                                                                                                     


arr2 = np.random.randn(50000, 10)
# print arr2
print np.mean(arr2, axis=0)
print np.mean(arr2)