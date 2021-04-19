#!/usr/bin/env python
# coding: utf-8

# In[2]:


from collections import defaultdict
from PIL import Image
import numpy as np
import os


# In[3]:


def load_images_list(filepath):
    '''
    Load names of train/test/val images into a list from a text file
    '''
    images_txt = open(filepath,'r')
    images_list = []
    for line in images_txt:
        images_list.append(line.strip())
    return images_list


# In[ ]:




