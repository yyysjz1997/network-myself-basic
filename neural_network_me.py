import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * x_data * 0.5+ x_data * 0.6 + 0.1

Weights1 = tf.Variable(tf.random_uniform([1],0,1.0))
Weights = tf.Variable(tf.random_uniform([1],0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights1 * x_data * x_data + Weights *x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(3000):
    sess.run(train)
    if step % 100 == 0:
        print(step,sess.run(Weights1),sess.run(Weights),sess.run(biases))

'''
'''
#coding:utf-8
#两层简单的神经网络（全连接)
import tensorflow as tf
#定义输入和参数
#用placeholder实现输入定义（sess.run中喂一组数据）
x = tf.placeholder(tf.float32,shape=(None,2))
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#定义前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
#y用会话计算结果
with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  print ("y is:\n",sess.run(y,feed_dict={x:[[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
  print ("w1:\n",sess.run(w1))
  print ("w2:\n",sess.run(w2))

'''



'''
tf.set_random_seed(1)
np.random.seed(1)

# fake data
n_data = np.ones((100, 2))
x0 = np.random.normal(2*n_data, 1)      # class0 x shape=(100, 2)
y0 = np.zeros(100)                      # class0 y shape=(100, 1)
x1 = np.random.normal(-2*n_data, 1)     # class1 x shape=(100, 2)
y1 = np.ones(100)                       # class1 y shape=(100, 1)
x = np.vstack((x0, x1))  # shape (200, 2) + some noise
y = np.hstack((y0, y1))  # shape (200, )

print(x)
# plot data
plt.scatter(x[:, 1], x[:, 0], c=y, s=100, lw=0, cmap='RdYlGn')
plt.show()
'''
 #coding:utf-8
#0导入模块 ，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 30
seed = 2
#基于seed产生随机数
rdm = np.random.RandomState(seed)
#随机数返回300行2列的矩阵，表示300组坐标点（x0,x1）作为输入数据集
X = rdm.randn(300,2)
#从X这个300行2列的矩阵中取出一行,判断如果两个坐标的平方和小于2，给Y赋值1，其余赋值0
#作为输入数据集的标签（正确答案）
Y_ = [int(x0*x0 + x1*x1 <2) for (x0,x1) in X]
#遍历Y中的每个元素，1赋值'red'其余赋值'blue'，这样可视化显示时人可以直观区分
Y_c = [['red' if y else 'blue'] for y in Y_]
#对数据集X和标签Y进行shape整理，第一个元素为-1表示，随第二个参数计算得到，第二个元素表示多少列，把X整理为n行2列，把Y整理为n行1列
X = np.vstack(X).reshape(-1,2)
Y_ = np.vstack(Y_).reshape(-1,1)
print (X)
print (Y_)
print (Y_c)
#用plt.scatter画出数据集X各行中第0列元素和第1列元素的点即各行的（x0，x1），用各行Y_c对应的值表示颜色（c是color的缩写）
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.show()

#定义神经网络的输入、参数和输出，定义前向传播过程
def get_weight(shape, regularizer):
  w=tf.Variable(tf.random_normal(shape), dtype=tf.float32)
  tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
  return w

def get_bias(shape):
  b = tf.Variable(tf.constant(0.01, shape=shape))
  return b

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2,11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1)+b1)

w2 = get_weight([11,1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2)+b2

#定义损失函数
loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

#定义反向传播方法：不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  STEPS = 40000
  for i in range(STEPS):
    start = (i*BATCH_SIZE) % 300
    end = start + BATCH_SIZE
    sess.run(train_step, feed_dict={x:X [start:end], y_:Y_[start:end]})
    if i % 2000 == 0:
      loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
      print("After %d steps, loss is: %f" %(i, loss_mse_v))
#xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成二维网格坐标点
  xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
#将xx , yy拉直，并合并成一个2列的矩阵，得到一个网格坐标点的集合
  grid = np.c_[xx.ravel(), yy.ravel()]
#将网格坐标点喂入神经网络 ，probs为输出
  probs = sess.run(y, feed_dict={x:grid})
#probs的shape调整成xx的样子
  probs = probs.reshape(xx.shape)
  print ("w1:\n",sess.run(w1))
  print ("b1:\n",sess.run(b1))
  print ("w2:\n",sess.run(w2))
  print ("b2:\n",sess.run(b2))
plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

#定义反向传播方法：包含正则化 train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  STEPS = 40000
  for i in range(STEPS):
    start = (i*BATCH_SIZE) % 300
    end = start + BATCH_SIZE
    sess.run(train_step, feed_dict={x: X[start:end], y_:Y_[start:end]})
    if i % 2000 == 0:
      loss_v = sess.run(loss_total, feed_dict={x:X,y_:Y_})
      print("After %d steps, loss is: %f" %(i, loss_v))
  xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
  grid = np.c_[xx.ravel(), yy.ravel()]
  probs = sess.run(y, feed_dict={x:grid})
  probs = probs.reshape(xx.shape)
  print ("w1:\n",sess.run(w1))
  print ("b1:\n",sess.run(b1))
  print ("w2:\n",sess.run(w2))
  print ("b2:\n",sess.run(b2))

plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()