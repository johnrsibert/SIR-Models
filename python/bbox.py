#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:32:41 2020

@author: jsibert
"""
import matplotlib.pyplot as plt
r = plt.Rectangle((2,2), 10, 10, fill = False)
plt.gca().add_patch(r)
ymin, ymax = (0, 14)
plt.axis(xmin = 0, xmax = 14, ymin=ymin, ymax=ymax)

# Get dimensions of y-axis in pixels
y1, y2 = plt.gca().get_window_extent().get_points()[:, 1]

# Get unit scale
yscale = (y2-y1)/(ymax-ymin)

# We want 2 of these as fontsize
fontsize = 2*yscale
print(fontsize, 'pixels')

txt = plt.text(7, 7, u"\u25AF" + 'my rectangle', fontsize=fontsize, 
               ha='center', va='center')

plt.savefig('test.png')
