import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#加载MNIST

trainNum=60000  # 训练图片总数
testNum=10000   # 测试图片总数
trainSize=50000   # 训练时候用到的图片数量
testSize=5      # 测试时候用到的图片数量
k=4             # 距离最小的K个图片
#设置好特定参数

trainIndex=np.random.choice(55000,trainSize,replace=False)
testIndex=np.random.choice(testNum,testSize,replace=False)
print(trainIndex.shape,testIndex.shape)
#随机选取训练样本和测试样本

# 生成训练数据
trainData=mnist.train.images[trainIndex]
trainLabel=mnist.train.labels[trainIndex]
# 生成测试数据
testData=mnist.test.images[testIndex]
testLabel=mnist.test.labels[testIndex]
print('trainData.shape=',trainData.shape)
print('trainLabel.shape=',trainLabel.shape)
print('testData.shape=',testData.shape)
print('testLabel.shape=',testLabel.shape)
print('testLabel=',testLabel)

trainDataInput=tf.placeholder(shape=[None,784],dtype=tf.float32)
trainLabelInput=tf.placeholder(shape=[None,10],dtype=tf.float32)
testDataInput=tf.placeholder(shape=[None,784],dtype=tf.float32)
testLabelInput=tf.placeholder(shape=[None,10],dtype=tf.float32)
#设定参数以便使用tensorflow进行运算

f1=tf.expand_dims(testDataInput,1)      # 用expand_dim()来增加维度,将原来的testDataInput扩展成三维的,f1:(?,1,784)
f2=tf.subtract(trainDataInput,f1)       # subtract()执行相减操作，即 trainDataInput-testDataInput ,最终得到一个三维数据
f3=tf.reduce_sum(tf.abs(f2),reduction_indices=2)    # tf.abs()求数据绝对值,tf.reduce_sum()完成数据累加，把数据放到f3中
#计算每个训练样本和测试样本之间的曼哈顿距离

with tf.Session() as sess:
    p1=sess.run(f1,feed_dict={testDataInput:testData[0:testSize]})  # 取testData中的前testSize个样本来代替输入的测试数据
    print(p1)
    print('p1=',p1.shape)
    p2=sess.run(f2,feed_dict={trainDataInput:trainData,testDataInput:testData[0:testSize]})
    print('p2=',p2.shape)
    p3=sess.run(f3,feed_dict={trainDataInput:trainData,testDataInput:testData[0:testSize]})
    print('p3=',p3.shape)
    print('p3[0,0]=',p3[0,0])   # 输出第一张测试图片和第一张训练图片的距离
#使用tensorflow来运算

f4=tf.negative(f3)  # 计算f3数组中元素的负数
f5,f6=tf.nn.top_k(f4,k=4)   # f5:选取f4最大的四个值，即f3最小的四个值，f6:这四个值对应的索引
with tf.Session() as sess:
    p4=sess.run(f4,feed_dict={trainDataInput:trainData,testDataInput:testData[0:testSize]})
    print('p4=',p4.shape)
    print('p4[0,0]=',p4[0,0])
    # p5=(5,4),每一张测试图片(共5张)，分别对应4张最近训练图片，共20张
    p5,p6=sess.run((f5,f6),feed_dict={trainDataInput:trainData,testDataInput:testData[0:testSize]})
    print('p5=',p5.shape)
    print('p6=',p6.shape)
    print('p5:',p5,'\n','p6:',p6)
#选择距离最小的K个图片

f7=tf.gather(trainLabelInput,f6)    # 根据索引找到对应的标签值
f8=tf.reduce_sum(f7,reduction_indices=1)    # 累加维度1的数值
f9=tf.argmax(f8,dimension=1)        # 返回的是f8中的最大值的索引号
# 执行
with tf.Session() as sess:
    p7=sess.run(f7,feed_dict={trainDataInput:trainData,testDataInput:testData[0:testSize],trainLabelInput:trainLabel})
    print('p7=',p7.shape)
    print('p7:',p7)
    p8=sess.run(f8,feed_dict={trainDataInput:trainData,testDataInput:testData[0:testSize],trainLabelInput:trainLabel})
    print('p8=',p8.shape)
    print('p8:',p8)
    p9=sess.run(f9,feed_dict={trainDataInput:trainData,testDataInput:testData[0:testSize],trainLabelInput:trainLabel})
    print('p9=',p9.shape)
    print('p9:',p9)
#计算每个类型出现的频率

with tf.Session() as sess:
    p10=np.argmax(testLabel[0:testSize],axis=1)    # 如果p9=p10，代表正确
    print(('p10:',p10))
#为每一个样本作出最终的测试值


j=0
for i in range(0,testSize):
    if p10[i]==p9[i]:
        j=j+1
# 输出准确率
print('accuracy=',j*100/testSize,'%')
#计算准确度