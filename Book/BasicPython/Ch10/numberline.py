# coding=utf-8
'''
Created on 2016年3月26日

@author: esm
'''

import fileinput                         # 10
                                         # 11
for line in fileinput.input(inplace=True): # 12
    line = line.rstrip()                 # 13
    num = fileinput.lineno()             # 14
    print '%-40s # %2i' %(line, num)     # 15