import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class ResNet(tf.keras.Model):
    def __init__(self,in_channels, is_initial_block = False):
        super(ResNet, self).__init__()

        if is_initial_block:
            self.conv1 = keras.Sequential([layers.Conv1D(96 , kernel_size = 3 , strides = 2 ,padding = 'same', input_shape = in_channels),
                                            layers.BatchNormalization(axis = -1)])
        else:    
            self.conv1 = keras.Sequential([layers.Conv1D(96 , kernel_size = 3 , strides = 2 ,padding = 'same'),
                                             layers.BatchNormalization(axis = -1)])
        
        self.conv2 = keras.Sequential([layers.Conv1D(96 , kernel_size = 3 , strides = 1,padding = 'same'),
                                         layers.BatchNormalization(axis = -1)])
        
        self.conv3 =  keras.Sequential([layers.Conv1D(96 , kernel_size = 3 , strides = 1,padding = 'same'),
                                         layers.BatchNormalization(axis = -1)])
        
        self.conv4 = layers.ReLU()
    
    def call(self,x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = out_3 + out_1
        out_5 = self.conv4(out_4)

        return out_5

class CNN (tf.keras.Model):
    def __init__(self,in_channels):
        super(CNN, self).__init__()

        self.layer1 = ResNet(in_channels, is_initial_block = True)
        self.layer2 = ResNet(96)
        self.layer3 = ResNet(96)
        self.layer4 = ResNet(96)
        self.layer5 = ResNet(96)
        self.layer6 = layers.MaxPool1D(strides = 2)
        self.layer7 = layers.Flatten()
        self.layer8 = layers.Dense(20)
        self.layer9 = layers.Dense(20)
        self.layer10 = layers.Dense(1)

    def call (self,x):

        out_1 = self.layer1(x)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        out_6 = self.layer6(out_5)
        out_7 = self.layer7(out_6)
        out_8 = self.layer8(out_7)
        out_9 = self.layer9(out_8)
        out_10 = self.layer10(out_9)

        return out_10
  
