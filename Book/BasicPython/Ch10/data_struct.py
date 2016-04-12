# coding=utf-8
'''
Created on 2016年3月26日

@author: esm
'''

print type(set(range(10)))
print type(range(10))

a = set(range(10))
a.copy()
print a.copy()

print a.copy() is a

from random import shuffle
from heapq import heappush, heappop

data = range(10)
shuffle(data)
heap = []
for n in data:
    heappush(heap, n)
print heap

heappush(heap, 0.5)
print heap

print heappop(heap)
print heappop(heap)
print heappop(heap)

import time

print time.time()
print time.asctime()
print time.localtime()
print time.mktime(time.localtime())

time.sleep(1)

print "Wake up!"

print time.strftime("12")