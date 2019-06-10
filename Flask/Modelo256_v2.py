
# coding: utf-8

# In[1]:


from tensorflow.python.client import device_lib 


# In[2]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

print(device_lib.list_local_devices())


# In[3]:



class Model():
    
    def __init__(self,batchSize):
        self.batch_size=batchSize
        
    def buildModel(self,input_image):
        bs=self.batch_size
        def truncated_normal_var(name, shape, dtype):
            return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
        def zero_var(name, shape, dtype):
            return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

        with tf.variable_scope('conv0') as scope:#256x256x3
            #variables  PESOS y BIAS
            conv0_kernel = truncated_normal_var(name='conv_kernel0', shape=[5, 5, 3, 64], dtype=tf.float32)
            conv0_bias = zero_var(name='conv_bias0',shape=[64],dtype=tf.float32)

            conv0=tf.nn.conv2d(input_image,conv0_kernel,[1,1,1,1],padding='SAME') #(input,filter,stride,padding)#"SAME"nos devulve las mism hxw
            conv0_add_bias=tf.nn.bias_add(conv0,conv0_bias)
            conv0_relu=tf.nn.relu(conv0_add_bias)
      
        pool0=tf.nn.max_pool(conv0_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer1")#32x32  *128
        
        #Salida 128x128x64
        

        with tf.variable_scope('conv1') as scope:#128x128x64
            #variables  PESOS y BIAS
            conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 64, 64], dtype=tf.float32)
            conv1_bias = zero_var(name='conv_bias1',shape=[64],dtype=tf.float32)

            conv1=tf.nn.conv2d(pool0,conv1_kernel,[1,1,1,1],padding='SAME') #(input,filter,stride,padding)
            conv1_add_bias=tf.nn.bias_add(conv1,conv1_bias)
            conv1_relu=tf.nn.relu(conv1_add_bias)
        
        pool1=tf.nn.max_pool(conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer2")
        
        #Salida 64x64x64

        with tf.variable_scope('conv2') as scope:#64x64x64
            #variables  PESOS y BIAS
            conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 128], dtype=tf.float32)
            conv2_bias = zero_var(name='conv_bias2',shape=[128],dtype=tf.float32)

            conv2=tf.nn.conv2d(pool1,conv2_kernel,[1,1,1,1],padding='SAME')
            conv2_add_bias=tf.nn.bias_add(conv2,conv2_bias)
            conv2_relu=tf.nn.relu(conv2_add_bias)

        pool2=tf.nn.max_pool(conv2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer3")
        
        #Salida 32x32x128

        with tf.variable_scope('conv3') as scope:#32x32x128
            #variables  PESOS y BIAS
            conv3_kernel = truncated_normal_var(name='conv_kernel3', shape=[3, 3, 128, 256], dtype=tf.float32)
            conv3_bias = zero_var(name='conv_bias3',shape=[256],dtype=tf.float32)

            conv3=tf.nn.conv2d(pool2,conv3_kernel,[1,1,1,1],padding='SAME')
            conv3_add_bias=tf.nn.bias_add(conv3,conv3_bias)
            conv3_relu=tf.nn.relu(conv3_add_bias)

        pool3=tf.nn.max_pool(conv3_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer4")
        
        #Salida 16x16x256
        
        #residual block
        with tf.variable_scope('residual1') as scope:#32x32x128
            #variables  PESOS y BIAS
            res4_kernel = truncated_normal_var(name='res_kernel4', shape=[3, 3, 256, 512], dtype=tf.float32)
            res4_bias = zero_var(name='res_bias4',shape=[512],dtype=tf.float32)
            
            res4=tf.nn.conv2d(pool3,res4_kernel,[1,1,1,1],padding='SAME')
            res4_add_bias=tf.nn.bias_add(res4,res4_bias)
            res4_relu=tf.nn.relu(res4_add_bias)
            
        with tf.variable_scope('residual2') as scope:#32x32x128
            #variables  PESOS y BIAS
            res5_kernel = truncated_normal_var(name='res_kernel5', shape=[3, 3, 512, 512], dtype=tf.float32)
            res5_bias = zero_var(name='res_bias5',shape=[512],dtype=tf.float32)
            
            res5=tf.nn.conv2d(res4_relu,res5_kernel,[1,1,1,1],padding='SAME')
            res5_add_bias=tf.nn.bias_add(res5,res5_bias)
            res5_relu=tf.nn.relu(res5_add_bias)
        
        with tf.variable_scope('residual3') as scope:#32x32x128
            #variables  PESOS y BIAS
            res6_kernel = truncated_normal_var(name='res_kernel6', shape=[3, 3, 512, 256], dtype=tf.float32)
            res6_bias = zero_var(name='res_bias6',shape=[256],dtype=tf.float32)
            
            res6=tf.nn.conv2d(res5_relu,res6_kernel,[1,1,1,1],padding='SAME')
            res6_add_bias=tf.nn.bias_add(res6,res6_bias)
            res6_relu=tf.nn.relu(res6_add_bias)
            
        #Salida 16x16x256

        
            

        with tf.variable_scope('dconv3') as scope:#16x16x256
            #variables  PESOS y BIAS
            dconv3_kernel = truncated_normal_var(name='dconv_kernel3', shape=[5, 5, 128, 256], dtype=tf.float32)

            #dconv2_relu=tf.concat([dconv2_relu, pool1], axis=3)

            dconv3=tf.nn.conv2d_transpose(res6_relu,dconv3_kernel,[bs,32,32,128],[1,2,2,1],padding='SAME')
            dconv3_relu=tf.nn.relu(dconv3)
            
         #Salida 32x32x128


        with tf.variable_scope('dconv4') as scope:#32x32x128
            #variables  PESOS y BIAS
            dconv4_kernel = truncated_normal_var(name='dconv_kernel4', shape=[5, 5, 64, 128], dtype=tf.float32)

            #dconv3_relu=tf.concat([dconv3_relu, pool0], axis=3)

            dconv4=tf.nn.conv2d_transpose(dconv3_relu,dconv4_kernel,[bs,64,64,64],[1,2,2,1],padding='SAME')
            #[1,3,3,1] indica por cuanto se multiplica las entradas para aumentar el tamaño
            dconv4_relu=tf.nn.relu(dconv4)
            
        #Salida 64x64x64
        
        with tf.variable_scope('dconv5') as scope:#64x64x64
            #variables  PESOS y BIAS
            dconv5_kernel = truncated_normal_var(name='dconv_kernel5', shape=[5, 5, 64, 64], dtype=tf.float32)

            #dconv3_relu=tf.concat([dconv3_relu, pool0], axis=3)

            dconv5=tf.nn.conv2d_transpose(dconv4_relu,dconv5_kernel,[bs,128,128,64],[1,2,2,1],padding='SAME')
            #[1,3,3,1] indica por cuanto se multiplica las entradas para aumentar el tamaño
            dconv5_relu=tf.nn.relu(dconv5)
        
        #Salida 128x128x64
        
        with tf.variable_scope('dconv6') as scope:#128x128x64
            #variables  PESOS y BIAS
            dconv6_kernel = truncated_normal_var(name='dconv_kernel6', shape=[5, 5, 32, 64], dtype=tf.float32)

            #dconv3_relu=tf.concat([dconv3_relu, pool0], axis=3)

            dconv6=tf.nn.conv2d_transpose(dconv5_relu,dconv6_kernel,[bs,256,256,32],[1,2,2,1],padding='SAME')
            #[1,3,3,1] indica por cuanto se multiplica las entradas para aumentar el tamaño
            dconv6_relu=tf.nn.relu(dconv6)
        
        #Salida 256x256x32
        
        with tf.variable_scope('dconv7_final') as scope:#128x128x64
            #variables  PESOS y BIAS
            dconv7_kernel = truncated_normal_var(name='dconv_kernel7', shape=[5, 5, 3, 32], dtype=tf.float32)

            #dconv3_relu=tf.concat([dconv3_relu, pool0], axis=3)

            dconv7=tf.nn.conv2d_transpose(dconv6_relu,dconv7_kernel,[bs,256,256,3],[1,1,1,1],padding='SAME')
            #[1,3,3,1] indica por cuanto se multiplica las entradas para aumentar el tamaño
            dconv7_relu=tf.nn.relu(dconv7)
        
        #Salida 256x256x3

        return dconv7_relu
    
    def buildAdversialModel(self,input_image):
        def truncated_normal_var(name, shape, dtype):
            return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
        def zero_var(name, shape, dtype):
            return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))
        
        with tf.variable_scope('conv0_Ad') as scope:#256x256x3
            conv0_kernel=truncated_normal_var(name="conv_0_kernel_ad",shape=[4,4,3,64],dtype=tf.float32)
            conv0_bias=zero_var(name="conv0_bias_ad",shape=[64],dtype=tf.float32)
            
            conv0=tf.nn.conv2d(input_image,conv0_kernel,strides=[1,1,1,1],padding="SAME")
            conv0_add_bias=tf.nn.bias_add(conv0,conv0_bias)
            conv0_relu=tf.nn.relu(conv0_add_bias)
            
        pool0=tf.nn.max_pool(conv0_relu,[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer_ad_0")
        
        with tf.variable_scope('conv1_Ad') as scope:#128x128x64
            #variables  PESOS y BIAS
            conv1_kernel = truncated_normal_var(name='conv_kernel1_ad', shape=[5, 5, 64, 64], dtype=tf.float32)
            conv1_bias = zero_var(name='conv_bias1_ad',shape=[64],dtype=tf.float32)

            conv1=tf.nn.conv2d(pool0,conv1_kernel,[1,1,1,1],padding='SAME') #(input,filter,stride,padding)
            conv1_add_bias=tf.nn.bias_add(conv1,conv1_bias)
            conv1_relu=tf.nn.relu(conv1_add_bias)
        
        pool1=tf.nn.max_pool(conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer_ad1")
        
        #Salida 64x64x64

        with tf.variable_scope('conv2_Ad') as scope:#64x64x64
            #variables  PESOS y BIAS
            conv2_kernel = truncated_normal_var(name='conv_kernel2_ad', shape=[5, 5, 64, 128], dtype=tf.float32)
            conv2_bias = zero_var(name='conv_bias2_ad',shape=[128],dtype=tf.float32)

            conv2=tf.nn.conv2d(pool1,conv2_kernel,[1,1,1,1],padding='SAME')
            conv2_add_bias=tf.nn.bias_add(conv2,conv2_bias)
            conv2_relu=tf.nn.relu(conv2_add_bias)

        pool2=tf.nn.max_pool(conv2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer_ad2")
        
        #Salida 32x32x128

        with tf.variable_scope('conv3_Ad') as scope:#32x32x128
            #variables  PESOS y BIAS
            conv3_kernel = truncated_normal_var(name='conv_kernel3_ad', shape=[3, 3, 128, 128], dtype=tf.float32)
            conv3_bias = zero_var(name='conv_bias3_ad',shape=[128],dtype=tf.float32)

            conv3=tf.nn.conv2d(pool2,conv3_kernel,[1,1,1,1],padding='SAME')
            conv3_add_bias=tf.nn.bias_add(conv3,conv3_bias)
            conv3_relu=tf.nn.relu(conv3_add_bias)

        pool3=tf.nn.max_pool(conv3_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer_ad3")
        
        #Salida 16x16x128
        
        with tf.variable_scope('conv4_Ad') as scope:#16x16x128
            #variables  PESOS y BIAS
            conv4_kernel = truncated_normal_var(name='conv_kernel4_ad', shape=[3, 3, 128, 256], dtype=tf.float32)
            conv4_bias = zero_var(name='conv_bias4_ad',shape=[256],dtype=tf.float32)

            conv4=tf.nn.conv2d(pool3,conv4_kernel,[1,1,1,1],padding='SAME')
            conv4_add_bias=tf.nn.bias_add(conv4,conv4_bias)
            conv4_relu=tf.nn.relu(conv4_add_bias)

        pool4=tf.nn.max_pool(conv4_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer_ad4")
        
        #Salida 8x8x256
        
        with tf.variable_scope('conv5_Ad') as scope:#16x16x128
            #variables  PESOS y BIAS
            conv5_kernel = truncated_normal_var(name='conv_kernel5_ad', shape=[3, 3, 256, 512], dtype=tf.float32)
            conv5_bias = zero_var(name='conv_bias5_ad',shape=[512],dtype=tf.float32)

            conv5=tf.nn.conv2d(pool4,conv5_kernel,[1,1,1,1],padding='SAME')
            conv5_add_bias=tf.nn.bias_add(conv5,conv5_bias)
            conv5_relu=tf.nn.relu(conv5_add_bias)

        pool5=tf.nn.max_pool(conv5_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool_layer_ad5")
        
        #Salida 4x4x512
        
        with tf.name_scope('Flatten_Ad') as scope: #pasar todos a un array
            flat_output=tf.layers.flatten(pool5)
        
        with tf.variable_scope('FC_Ad') as scope:
            shapeFO = flat_output.get_shape().as_list()
            #[entra,sale]
            matrix=truncated_normal_var(name="matrix",shape=[shapeFO[1],1],dtype=tf.float32)
            fc_bias=zero_var(name="bas_fc",shape=[1],dtype=tf.float32)
            output=tf.add(tf.matmul(flat_output,matrix),fc_bias)
            
        
        return tf.nn.sigmoid(output), output


# In[4]:





