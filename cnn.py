"""
CNN 
"""

import tensorflow as tf 
import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt

# import dataset
paths = ['/sushi_small/', '/airplanes_small/']
X = []
y = []
num_pics =  0
for path in paths:
    for filename in os.listdir(os.getcwd() + path):
        im = cv2.imread(os.getcwd() + path+ filename, cv2.IMREAD_COLOR)
        if im is not None: 
             if not (im.shape[0]==150 and im.shape[1]==150 and im.shape[2]==3):
               print("size mis-match")
               break;
               """
               if (im.shape[0] > 150):
                       im = im[im.shape[0]//2-75:im.shape[0]//2+75, ]
               if (im.shape[1] > 150):
                       im = im[:,im.shape[1]//2-75:im.shape[1]//2+75,]
               if (im.shape[0] < 150 or im.shape[1] < 150 or im.shape[2] <3):
                       zeros = np.zeros((150,150,3))
                       zeros[:im.shape[0], :im.shape[1], im.shape[2]] = im
                       im = zeros    
               """        
             num_pics+=1
             for o in im.flatten():
                 X.append(o)
             if path == paths[0]:
                 y.append(0)
             else: 
                 y.append(1)
             
             print(f"{filename} processed")
                            
        else:
            print("Nonetype detected")                

X = np.array(X).reshape((num_pics,150,150,3))
y = np.array(y).reshape((-1,1))
from sklearn.preprocessing import OneHotEncoder


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

X, y = shuffle_in_unison(X,y)



#Helper functions to build CNN (wrap around tensorflow's methods & setting params)
#Initialize weights
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1) #normal dist of weights, of a given shape
    return tf.Variable(init_random_dist) #create Variable of given dis/shape

#Initialize bias
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

#CONV2D (filter); x = [batch_size (how many images), Height, Weight, Channels], 
#W= [filter Height, filter Width, Channels in (eg pic colors), Channels out (# filters = # filtered result)]
#order of x and W according to Tensorflow spec!
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME') #SAME adds 0 to paddding 
    
#Pooling (of heigh $ width); x = [batch_size, Height, Weight, Channels], thus [same, pool, pool, same]
#ksize=size of filter (if =1, not max_pooling in that dimension), strides=stride of filter
def max_pooling2(x): 
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
def max_pooling3(x): 
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')
def max_pooling5(x): 
    return tf.nn.max_pool(x, ksize=[1,5,5,1], strides=[1,5,5,1], padding='SAME')

# Create final CNN Ops (conv and fc) from previous functions
#convolutional: image => conv2d(f(image, weights)) + bias 
def convolutional(input_x, shape): #shape assigned to weights
    W = init_weights(shape) 
    b = init_bias([shape[3]]) #third dimension of weight's shape (i.e. weight's channels out)
    return tf.nn.relu(conv2d(input_x,W)+b)

#Conv => Fully conncected 
def fc(input_layer, size):#size = how many weights
    input_size = int(input_layer.get_shape()[1]) #how many images
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

#placeholders
x = tf.placeholder(tf.float32, shape=[None, 150,150,3])
y_true = tf.placeholder(tf.float32, shape=[None, 1])

#layers
hold_prob = tf.placeholder(tf.float32) #for dropout

x_image = tf.reshape(x, [-1,150,150,3])
conv1 = convolutional(x_image, shape=[2,2,3,32]) #2x2=filter size; 3=filter height (3 channels);32 filters=32 filtersed images/features =>result: same image size 150x150 (b/c padding=same) but now 32 filtered images

batch_normalized = tf.layers.batch_normalization(conv1)
conv1_pooling = max_pooling2(batch_normalized) #50% reduction of image size => 75x75

conv2 = convolutional(conv1_pooling, shape=[2,2,32,64]) #2x2patches, 32 patches from filtered images, 64 filters
conv3 = convolutional(conv2, shape=[2,2,64,64])
conv4 = convolutional(conv3, shape=[2,2,64,64])
conv5 = convolutional(conv4, shape=[2,2,64,64])
conv5_pooling = max_pooling3(conv5) #1/3 reduction of image size => 25*25

conv6 = convolutional(conv5_pooling, shape=[3,3,64,128])
conv7 = convolutional(conv6, shape=[3,3,128,128])
conv8 = convolutional(conv7, shape=[3,3,128,128])
conv9 = convolutional(conv8, shape=[3,3,128,128])
conv9_pooling  = max_pooling5(conv9) #1/5 reduction => 5*5

conv10 = convolutional(conv9_pooling, shape=[3,3,128,256])
conv11 = convolutional(conv10, shape=[3,3,256,256])
conv12 = convolutional(conv11, shape=[3,3,256,256])
conv13 = convolutional(conv12, shape=[3,3,256,512])
conv14 = convolutional(conv13, shape=[3,3,512,512])


conv14_flat = tf.reshape(conv14,[-1,5*5*512]) 
fc1= tf.nn.relu(fc(conv14_flat, 4096)) #4096 neurons
fc2= tf.nn.relu(fc(fc1, 4096))
fc2_dropout = tf.nn.dropout(fc2, keep_prob=hold_prob)
fc3= tf.nn.relu(fc(fc2_dropout, 1024))

fc4 = tf.nn.relu(fc(fc3, 1))
y_pred =tf.sigmoid(fc4)
predicted_class = tf.greater(y_pred,0.5)


#loss
sigmoid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
#optimizer/sgd
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(sigmoid)

init = tf.global_variables_initializer()

steps = 4000
batch_size = 20
samples = y.shape[0]
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        print(f"step {i} started")
        if (batch_size*(i+1)%samples < batch_size*i%samples):
            x_batch = X[(batch_size*i%samples):, :,:,:]
            y_batch = y[(batch_size*i%samples):, :]
        else:
            x_batch = X[(batch_size*i%samples):(batch_size*(i+1)%samples), :,:,:]
            y_batch = y[(batch_size*i%samples):(batch_size*(i+1)%samples), :]
        _, loss, preds = sess.run([train, sigmoid, predicted_class], feed_dict={x:x_batch, y_true: y_batch, hold_prob: 0.6})
        print(f"loss: {loss}")
        correct_preds =np.equal(preds, y_batch)
        acc = np.mean((correct_preds.astype(float)))
        print(f"accuracy: {acc}")
        if (i%40 == 0):
            saver.save(sess, f'./jp_food_{i}.ckpt')

            
