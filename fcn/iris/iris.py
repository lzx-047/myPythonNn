import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import datasets

x_data=datasets.load_iris().data
y_data=datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train=x_data[:-30]
y_train=y_data[:-30]
x_test=x_data[-30:]
y_test=y_data[-30:]

x_train=tf.cast(x_train,tf.float32)
x_test=tf.cast(x_test,tf.float32)

train_db=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

w1=tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1=tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))

lr=0.1
#2020.10.05 23:55
train_loss_results=[]
test_acc=[]
epoch=10
loss_all=0

for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y=tf.matmul(x_train,w1)+b1
            y=tf.nn.softmax(y)
            y_=tf.one_hot(y_train,depth=3)
            loss=tf.reduce_mean(tf.square(y_-y))
            loss_all+=loss.numpy()
        grads=tape.gradient(loss,[w1,b1])
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])

    print("Epoch {},loss:{}".format(epoch,loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all=0

    total_conrrect,total_number=0,0
    for x_test,y_test in test_db:
        y=tf.matmul(x_test,w1)+b1
        y=tf.nn.softmax(y)
        pred=tf.argmax(y,axis=1)
        pred=tf.cast(pred,dtype=y_test.dtype)
        correct=tf.cast(tf.equal(pred,y_test),dtype=tf.int32)
        correct=tf.reduce_sum(correct)
        total_conrrect+=int(correct)
        total_number+=x_test.shape[0]

    acc=total_conrrect/total_number
    test_acc.append(acc)
    print("Test acc:",acc)
    print("--------------------")

plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_results,label="$Loss$")
plt.legend()
plt.show()

plt.title("Acc Curve")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc,label="$Accuracy$")
plt.legend()
plt.show()
