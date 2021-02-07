import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

class mnistModel(Model):
    def __init__(self):
        super(mnistModel,self).__init__()
        self.d1=Dense(10,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2()
        )
        self.f1=Flatten()
    def call(self,x):
        x=self.f1(x)
        y=self.d1(x)
        return y

model=mnistModel()
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    validation_freq=20
)