# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 07:45:10 2015

@author: wq
"""

from pymouse import PyMouse
from time import sleep
m = PyMouse()

initX = 589
initY = 419
positionY = initY
clickX1 = 480
clickX2 = 493
clickY1 = 321
clickY2 = 372
exitX = 801
exitY = 71

for i in range(10):
    m.click(initX, positionY)
    sleep(1)
    for j in range(20):        
        m.click(clickX1, clickY1)
        sleep(30)
        m.click(clickX2, clickY2)
        sleep(30)
    m.click(exitX, exitY)
    sleep(1)
    positionY = positionY + 19