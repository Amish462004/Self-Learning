import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Model
def identity_block(x,filters):
    f1,f2,f3=filters
    x_shortcut = x
    x=Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x=BatchNormalization(axis=3)(x)
    x=ReLU()(x)
    x=Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=ReLU()(x)
    x=Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x=BatchNormalization(axis=3)(x)
    x=Add()([x, x_shortcut])
    x=ReLU()(x)
    return x
def convolutional_block(x,filters,stride):
    f1,f2,f3=filters
    x_shortcut=x
    x=Conv2D(f1, kernel_size=(1,1), strides=(stride, stride), padding='valid')(x)
    x=BatchNormalization(axis=3)(x)
    x=ReLU()(x)
    x=Conv2D(f2, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x=BatchNormalization(axis=3)(x)
    x=ReLU()(x)
    x=Conv2D(f3, kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    x=BatchNormalization(axis=3)(x)
    x_shortcut=Conv2D(f3,kernel_size=(1,1),strides=(stride, stride),padding='valid')(x_shortcut)
    x_shortcut=BatchNormalization(axis=3)(x_shortcut)
    x=Add()([x,x_shortcut])
    x=ReLU()(x)
    return x
# Build the ResNet-50 model
def ResNet50(input_shape=(224, 224, 3), classes=1000):
    x_input=Input(input_shape)
    x=Conv2D(64,(7,7),strides=(2,2),padding='valid')(x_input)
    x=BatchNormalization(axis=3)(x)
    x=ReLU()(x)
    x=MaxPooling2D((3,3),strides=(2,2))(x)
    x=convolutional_block(x, filters=[64, 64, 256], stride=1)
    x=identity_block(x,filters=[64,64,256])
    x=identity_block(x,filters=[64,64,256])
    x=AveragePooling2D((2,2))(x)
    x=Flatten()(x)
    x=Dense(classes,activation='softmax')(x)
    model=Model(inputs=x_input,outputs=x,name='ResNet50')
    return model
model=ResNet50()
model.summary()

