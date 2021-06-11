## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net_1(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #input is a 224 grey scale image, output of : (224-5)/1+1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        #apply max pooling, the outputs will be 32 image of 110
        self.pool = nn.MaxPool2d(2, 2)
        #dropout layer
        self.fc1_drop = nn.Dropout(p=0.5)
        #conv2, input is going to be, output is goung to be 110-3 +1 = 106
        self.conv2 = nn.Conv2d(32, 64, 5)
        #after max pooling , this is going to be 64 image of size 53
        
        #output now will be 128 image of 49
        self.conv3 = nn.Conv2d(64, 128, 5) #output
        #23 -> 11
        self.conv4 = nn.Conv2d(128, 64, 3)  # output
        # -> 4
        self.conv5 = nn.Conv2d(64, 64, 3) #output 256 * 4 * 4
        
        #another max pooling
        #linear layer, will be followed by a dropout
        self.linear1 = nn.Linear(64*4*4, 272)
        
        
        #final linear
        self.linear2 = nn.Linear(272, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))#220 -> 110
        x = self.fc1_drop(x)
        print(x.size())
        x = self.pool(F.relu(self.conv2(x)))#106 -> 53
        print(x.size())
        x = self.fc1_drop(x)
        print(x.size())
        x = self.pool(F.relu(self.conv3(x)))#49 ->24
        print(x.size())
        x = self.fc1_drop(x)
        x = self.pool(F.relu(self.conv4(x)))#22 ->11
        print(x.size())
        x = self.pool(F.relu(self.conv5(x)))#9->4
        print(x.size())
        x = F.relu(x.view(x.size(0), -1))
        
        x = F.relu(self.linear1(x))
        x = self.fc1_drop(x)
        print(x.size())
        x = self.linear2(x)
        print(x.size())
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    
class Net_2(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #input is a 224 grey scale image, output of : (224-5)/1+1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        #apply max pooling, the outputs will be 32 image of 110
        self.pool = nn.MaxPool2d(2, 2)
        #dropout layer
        self.fc1_drop = nn.Dropout(p=0.5)
        #conv2, input is going to be, output is goung to be 110-3 +1 = 106
        self.conv2 = nn.Conv2d(32, 64, 5)
        #after max pooling , this is going to be 64 image of size 53
        
        #output now will be 128 image of 49
        self.conv3 = nn.Conv2d(64, 128, 5) #output
        #23 -> 11
        self.conv4 = nn.Conv2d(128, 64, 3)  # output
        # -> 4
        self.conv5 = nn.Conv2d(64, 64, 3) #output 256 * 4 * 4
        
        #another max pooling
        #linear layer, will be followed by a dropout
        self.linear1 = nn.Linear(64*11*11, 272)
        
        
        #final linear
        self.linear2 = nn.Linear(272, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))#220 -> 110
        x = self.pool(F.relu(self.conv2(x)))#106 -> 53
        x = self.fc1_drop(x)
        x = self.pool(F.relu(self.conv3(x)))#49 ->24
        x = self.fc1_drop(x)
        x = self.pool(F.relu(self.conv4(x)))#22 ->11
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.fc1_drop(x)
        x = self.linear2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x

    
    
class Net_4(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #input is a 224 grey scale image, output of : (224-5)/1+1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        #apply max pooling, the outputs will be 32 image of 110
        self.pool3 = nn.MaxPool2d(3, 3) # -> 73
        self.pool2 = nn.MaxPool2d(2, 2) # -> 73
        #conv2, input is going to be, output is goung to be 110-3 +1 = 106
        self.conv2 = nn.Conv2d(32, 64, 5)
        #after max pooling , this is going to be 64 image of size 53
        self.fc1_drop = nn.Dropout(p=0.5)
        #output now will be 128 image of 49
        self.conv3 = nn.Conv2d(64, 128, 5) #output
        #23 -> 11
        self.conv4 = nn.Conv2d(128, 256, 5)  # output
        # -> 4
        
        #another max pooling
        #linear layer, will be followed by a dropout
        self.linear1 = nn.Linear(256*10*10, 272)
        
        
        #final linear
        self.linear2 = nn.Linear(272, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool2(F.relu(self.conv1(x)))#220 -> 110

        x = self.pool2(F.relu(self.conv2(x)))#106 -> 53

        x = self.pool2(F.relu(self.conv3(x)))#49 ->24

        x = self.pool2(F.relu(self.conv4(x)))#20 ->10

        x = x.view(x.size(0), -1)

        x = F.relu(self.linear1(x))

        x = self.fc1_drop(x)
        x = self.linear2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x

    
    
   
    
class Net(nn.Module):#leaky rely

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #input is a 224 grey scale image, output of : (224-5)/1+1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        #apply max pooling, the outputs will be 32 image of 110
        self.pool3 = nn.MaxPool2d(3, 3) # -> 73
        self.pool2 = nn.MaxPool2d(2, 2) # -> 73
        #conv2, input is going to be, output is goung to be 110-3 +1 = 106
        self.conv2 = nn.Conv2d(32, 64, 5)
        #after max pooling , this is going to be 64 image of size 53
        self.fc1_drop = nn.Dropout(p=0.5)
        #output now will be 128 image of 49
        self.conv3 = nn.Conv2d(64, 128, 5) #output
        #23 -> 11
        self.conv4 = nn.Conv2d(128, 256, 5)  # output
        # -> 4
        
        #another max pooling
        #linear layer, will be followed by a dropout
        self.linear1 = nn.Linear(256*10*10, 1000)
        
        
        #final linear
        self.linear2 = nn.Linear(1000, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool2(F.leaky_relu(self.conv1(x)))#220 -> 110

        x = self.pool2(F.leaky_relu(self.conv2(x)))#106 -> 53

        x = self.pool2(F.leaky_relu(self.conv3(x)))#49 ->24

        x = self.pool2(F.leaky_relu(self.conv4(x)))#20 ->10

        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.linear1(x))

        x = self.fc1_drop(x)
        x = self.linear2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x

    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #input is a 224 grey scale image, output of : (224-5)/1+1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        #apply max pooling, the outputs will be 32 image of 110
        self.pool3 = nn.MaxPool2d(3, 3) # -> 73
        self.pool2 = nn.MaxPool2d(2, 2) # -> 73
        #conv2, input is going to be, output is goung to be 110-3 +1 = 106
        self.conv2 = nn.Conv2d(32, 64, 5)
        #after max pooling , this is going to be 64 image of size 53
        self.fc1_drop = nn.Dropout(p=0.5)
        #output now will be 128 image of 49
        self.conv3 = nn.Conv2d(64, 128, 5) #output
        #23 -> 11
        self.conv4 = nn.Conv2d(128, 256, 5)  # output
        # -> 4
        
        #another max pooling
        #linear layer, will be followed by a dropout
        self.linear1 = nn.Linear(128*24*24, 1000)
        
        
        #final linear
        self.linear2 = nn.Linear(1000, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool2(F.leaky_relu(self.conv1(x)))#220 -> 110
        x = self.fc1_drop(x)
        print(x.size())
        x = self.pool2(F.leaky_relu(self.conv2(x)))#106 -> 53
        x = self.fc1_drop(x)
        print(x.size())
        x = self.pool2(F.leaky_relu(self.conv3(x)))#49 ->24
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        x = F.leaky_relu(self.linear1(x))
        print(x.size())
        x = self.fc1_drop(x)
        x = self.linear2(x)
        print(x.size())
        # a modified x, having gone through all the layers of your model, should be returned
        return x
 
    