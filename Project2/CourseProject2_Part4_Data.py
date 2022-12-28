"""
CAP 5404 Deep Learning for Computer Graphics
Project II. Neural Networks & Computer Graphics

Pranath Reddy Kumbam (UFID: 8512-0977)

Part 4: Dataset Processing
"""

# Import libraries
import numpy as np
import cv2
import os
import glob
import random
import matplotlib.pyplot as plt

"""
Function to load images
The following article has been used as a reference for this Function
https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
"""
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

"""
Data Augmentation Functions
The following article has been used as a reference for the augmentation functions
https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5
Augmentations Used: Horizontal Flip, Scale (Zoom), Random Crops (Achieved by combining Random Shifts with Scale)
"""

# Reset the dimension of the image to original resolution
def fill(img, h, w):
  img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
  return img

# Function to ranodmly scale the image
def scale(img, value):
  value = random.uniform(value, 1)
  h, w = img.shape[:2]
  h_taken = int(value*h)
  w_taken = int(value*w)
  h_start = random.randint(0, h-h_taken)
  w_start = random.randint(0, w-w_taken)
  img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
  img = fill(img, h, w)
  return img

# Function to randomly shift the image
# Combined with scale to achieve random crops
def vertical_shift(img, ratio=0.0):
  if ratio > 1 or ratio < 0:
      print('Value should be less than 1 and greater than 0')
      return img
  ratio = random.uniform(-ratio, ratio)
  h, w = img.shape[:2]
  to_shift = h*ratio
  if ratio > 0:
      img = img[:int(h-to_shift), :, :]
  if ratio < 0:
      img = img[int(-1*to_shift):, :, :]
  img = fill(img, h, w)
  return img

# Function to randomly shift the image
# Combined with scale to achieve random crops
def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

# Image dimension
image_size = 128
image_directory = "./Data/datasets/NCD/ColorfulOriginal" # Directory of the images
# Load images
images = load_images_from_folder(image_directory)

# Resize images and put them into a numpy array
npimages = np.zeros((len(images), image_size, image_size, 3))
for i in range(len(images)):
  npimages[i] = cv2.cvtColor(cv2.resize(images[i], (image_size, image_size)), cv2.COLOR_BGR2RGB) / 255.0

# Split into train and test sets / augment only on train set
npimages_test = npimages[int(npimages.shape[0]*0.8):]
npimages = npimages[:int(npimages.shape[0]*0.8)]

print(npimages.shape)
print(npimages_test.shape)

# Expanding the dataset ten times by adding augmentations
# Value set to range [0.6,1]
augimages = np.zeros((len(npimages)*10, image_size, image_size, 3))
for i in range(len(npimages)):
  augimages[i*10+0] = npimages[i]
  augimages[i*10+1] = cv2.flip(npimages[i], 1) # Horizontal flip
  augimages[i*10+2] = scale(npimages[i], .6) # Scale
  augimages[i*10+3] = scale(vertical_shift(npimages[i], .6), .6) # Vertical shift + scale to achieve random crop
  augimages[i*10+4] = scale(horizontal_shift(npimages[i], .6), .6) # Horizontal shift + scale to achieve random crop
  # Combining all three
  augimages[i*10+5] = cv2.flip(scale(vertical_shift(npimages[i], .6), .6), 1) # Vertical shift + scale + horizontal flip
  augimages[i*10+6] = cv2.flip(scale(horizontal_shift(npimages[i], .6), .6), 1) # Horizontal shift + scale + horizontal flip
  augimages[i*10+7] = cv2.flip(npimages[i], 1) # Horizontal flip
  augimages[i*10+8] = cv2.flip(npimages[i], 1) # Horizontal flip
  augimages[i*10+9] = scale(npimages[i], .6) # Scale
np.random.shuffle(augimages)

#Save Augmented Training Datasets for Submission
if not os.path.exists("./Data/datasets/NCD/augmented"):
    os.mkdir("./Data/datasets/NCD/augmented")
if not os.path.exists("./Data/datasets/NCD/test"):
    os.mkdir("./Data/datasets/NCD/test")
if not os.path.exists("./Data/datasets/NCD/L"):
    os.mkdir("./Data/datasets/NCD/L")
if not os.path.exists("./Data/datasets/NCD/a"):
    os.mkdir("./Data/datasets/NCD/a")
if not os.path.exists("./Data/datasets/NCD/b"):
    os.mkdir("./Data/datasets/NCD/b")

for i in range(augimages.shape[0]):
  cv2.imwrite("./Data/datasets/NCD/augmented/img_" + str(i) + ".jpg", cv2.cvtColor(augimages[i].astype('float32')*255, cv2.COLOR_RGB2BGR))
for i in range(npimages_test.shape[0]):
  cv2.imwrite("./Data/datasets/NCD/test/img_" + str(i) + ".jpg", cv2.cvtColor(npimages_test[i].astype('float32')*255, cv2.COLOR_RGB2BGR))

# Convert to LAB
# Training Samples
l_images = np.zeros((augimages.shape[0], image_size, image_size))
a_images = np.zeros((augimages.shape[0], image_size, image_size))
b_images = np.zeros((augimages.shape[0], image_size, image_size))
i = 0
for file in os.listdir("./Data/datasets/NCD/augmented/"):
  temp_img = cv2.imread("./Data/datasets/NCD/augmented/" + file)
  lab_image = cv2.cvtColor(temp_img, cv2.COLOR_BGR2LAB)
  l,a,b = cv2.split(lab_image)
  l_images[i] = l
  a_images[i] = a
  b_images[i] = b
  i += 1

# Test Samples
l_images_ts = np.zeros((npimages_test.shape[0], image_size, image_size))
a_images_ts = np.zeros((npimages_test.shape[0], image_size, image_size))
b_images_ts = np.zeros((npimages_test.shape[0], image_size, image_size))
i = 0
for file in os.listdir("./Data/datasets/NCD/test/"):
  temp_img = cv2.imread("./Data/datasets/NCD/test/" + file)
  lab_image = cv2.cvtColor(temp_img, cv2.COLOR_BGR2LAB)
  l,a,b = cv2.split(lab_image)
  l_images_ts[i] = l
  a_images_ts[i] = a
  b_images_ts[i] = b
  i += 1

#Save Augmented Training Datasets for Submission
if not os.path.exists('./Data/arrays/NCD'):
    os.makedirs('./Data/arrays/NCD')

# Save numpy arrays for training
np.save('./Data/arrays/NCD/L_train.npy', l_images)
np.save('./Data/arrays/NCD/a_train.npy', a_images)
np.save('./Data/arrays/NCD/b_train.npy', b_images)
np.save('./Data/arrays/NCD/L_test.npy', l_images_ts)
np.save('./Data/arrays/NCD/a_test.npy', a_images_ts)
np.save('./Data/arrays/NCD/b_test.npy', b_images_ts)

# Save each channel to its own folder and zip them for submission
for i in range(augimages.shape[0]):
  cv2.imwrite("./Data/datasets/NCD/L/img_" + str(i) + ".jpg", l_images[i])
  cv2.imwrite("./Data/datasets/NCD/a/img_" + str(i) + ".jpg", a_images[i])
  cv2.imwrite("./Data/datasets/NCD/b/img_" + str(i) + ".jpg", b_images[i])
print('Done!')
