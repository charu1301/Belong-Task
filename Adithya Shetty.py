#!/usr/bin/env python
# coding: utf-8

# # Ultra Therapeutics (UTC)

# ## Import libraries

# In[1]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL


# ## Load images with variations

# In[2]:


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_dataset = datagen.flow_from_directory(
    "Data/train",
    seed=1,
    target_size=(224,224),
    batch_size=1,
    shuffle=True,
    class_mode='categorical',
    color_mode="rgb"
)

val_dataset = datagen.flow_from_directory(
    "Data/valid",
    seed=1,
    target_size=(224,224),
    batch_size=1,
    shuffle=True,
    class_mode='categorical',
    color_mode="rgb"
)

test_dataset = datagen.flow_from_directory(
    "Data/test",
    seed=1,
    target_size=(224,224),
    batch_size=1,
    shuffle=True,
    class_mode='categorical',
    color_mode="rgb"
)


# ## Visualize few training images

# In[3]:


class_map = train_dataset.class_indices
class_names = {v: k for k, v in class_map.items()}
print(class_names)
    

plt.figure(figsize=(10, 10))
for i in range(9):
    image, label = train_dataset.next()
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[0].astype("uint8"))
    plt.title(class_names[np.argmax(label[0])])
    plt.axis("off")


# In[ ]:




