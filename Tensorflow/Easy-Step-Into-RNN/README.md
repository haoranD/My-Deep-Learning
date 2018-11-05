
## Step into RNN

### - Forward

正向传播过程
循环神经网络的输入是时序相关的，因此其输入与输入可以描述为 h_1 , ..... ,h_T 、 y_1, ..... ,y_T ，为了保持时序信息，最简单的RNN函数形式为：

\begin{matrix} h_t=f(x_t,h_{t-1})\\ \rightarrow h_t=tanh(concat[x_t, h_{t-1}]\cdot W+b)\ \rightarrow tanh(x_t\cdot W1+h_{t-1}\cdot W2+b) \end{matrix} (1.1)

其中x_t的形式为[BATCHSIZE, Features1]，h_t的形式为[BatchSize, Features2]。多层rnn网络可以在输出的基础上继续加入RNN函数：

\begin{matrix} h^l_t=f(x_t,h^l_{t-1})\ h^{l+1}t=f(h^l_t, h^{l+1}{t-1}) \end{matrix} (1.2)

### -Backward
RNN函数反向传播过程与全链接网络类似：

\begin{matrix} e^l_{t-1}=\frac{\partial loss}{\partial h^l_{t-1}}=\frac{\partial loss}{\partial h^l_{t}}\circ f'(x_t,h^l_{t-1})\frac{\partial(x_t\cdot W1+h_{t-1}\cdot W2+b)}{\partial h_{t-1}}=e^l_t f' W2&(a)\\ \Delta W1=\sum_t (x_t)^T\cdot (e_t^lf')&(b)\\ \Delta W2=\sum_t (h_{t-1})^T\cdot (e_t^lf')&(c)\\ e^{l}{t}=\frac{\partial loss}{\partial h^{l}{t}}=\frac{\partial loss}{\partial h^{l+1}{t}}\circ f'(h_t^l,h^{l+1}{t-1})\frac{\partial(h_t^l\cdot W1+h_{t-1}\cdot W2+b)}{\partial h_{t}^l}=e^{l+1}_t f' W1&(d) \end{matrix} (1.3)

1.3-a称之为时间反向传播算法BPTT，1.3-c为层间传播。可训练参数为1.3-bc。实际上传统的RNN网络与全链接网络并无不同。只是添加了时间反向传播项。




```python
import numpy as np
import tensorflow as tf

class RNNCell():
    def __init__(self, insize=12, outsize=6, type="BASIC"):
        self.outsize = outsize
        self.insize = insize
        self.w = np.random.uniform(-0.1, 0.1, [insize+outsize, outsize])
        self.b = np.random.uniform(-0.1, 0.1, [outsize])
        self.outputs = []
        self.inputs = []
        self.states = []
    def tanh(self, x):
        epx = np.exp(x)
        enx = np.exp(-x)
        return (epx-enx)/(epx+enx)
    def __call__(self, x, s):
        self.inputs.append(x)
        self.inshape = np.shape(x)
        self.states.append(s)
        inx = np.concatenate([x, s], axis=1)
        out = np.dot(inx, self.w) + self.b
        self.outputs.append(out)
        out = self.tanh(out)
        return out, out
    def assign(self, w, b):
        self.w = w
        self.b = b
    def zero_state(self, batch_size):
        return np.zeros([batch_size, self.outsize])
    def get_error(self, error):
        self.error = error
    def d_tanh(self, x):
        e2x = np.exp(2 * x)
        return 4 * e2x / (1 + e2x) ** 2
    def backward(self):
        self.back_error = [np.zeros(self.inshape) for itr in range(len(self.outputs))]
        dw = np.zeros_like(self.w)
        db = np.zeros([self.outsize])
        w1 = self.w[:self.insize, :]
        w2 = self.w[self.insize:, :]
        for itrs in range(len(self.outputs)-1, -1, -1):
            if len(self.error[itrs]) == 0:
                continue
            else:
                err = self.error[itrs]
            for itr in range(itrs, -1, -1):
                h1 = self.outputs[itr]
                h0 = self.states[itr]
                x = self.inputs[itr]
                d_fe = self.d_tanh(h1)
                #print("es", np.shape(err), itr)
                err = d_fe * err
                dw[:self.insize, :] += np.dot(x.T, err)
                dw[self.insize:, :] += np.dot(h0.T, err)
                db += np.sum(err, axis=0)
                self.back_error[itr] += np.dot(err, w1.T)
                #print(np.shape(self.back_error))
                err = np.dot(err, w2.T)
        self.dw = dw
        self.db = db
        return dw, db
    def loss(self, y):
        self.error = []
        for itr in range(len(self.outputs)):
            self.error.append([])
        self.error[-1] = 2 * (self.tanh(self.outputs[-1]) - y)
        self.error[-2] = 2 * (self.tanh(self.outputs[-2]) - y)


class MultiRNNCells():
    def __init__(self, rnn_cells):
        print(rnn_cells)
        self.cells = rnn_cells
        self.cont = 0
    def __call__(self, x, s):
        state = []
        out = x
        for idx in range(len(self.cells)):
            out, st = self.cells[idx](out, s[idx])
            state.append(st)
        self.cont += 1
        return out, state
    def tanh(self, x):
        epx = np.exp(x)
        enx = np.exp(-x)
        return (epx-enx)/(epx+enx)
    def loss(self, y):
        self.error = []
        for itr in range(self.cont):
            self.error.append([])
        self.error[-1] = 2 * (self.tanh(self.cells[-1].outputs[-1]) - y)
        self.error[-2] = 2 * (self.tanh(self.cells[-1].outputs[-2]) - y)
    def backward(self):
        error = self.error
        dws = []
        for itr in range(len(self.cells)-1, -1, -1):
            self.cells[itr].get_error(error)
            self.cells[itr].backward()
            error = self.cells[itr].back_error
            dws.append(self.cells[itr].dw)
            dws.append(self.cells[itr].db)
        return tuple(dws)
    def apply_gradient(self, eta=0.1):
        dws = []
        for itr in range(len(self.cells)-1, -1, -1):
            self.cells[itr].w -= self.cells[itr].dw
            self.cells[itr].b -= self.cells[itr].db
            dws.append(self.cells[itr].dw)
            dws.append(self.cells[itr].db)
        return tuple(dws)
```


```python
batch_size = 1
max_time = 10
indata = tf.placeholder(dtype=tf.float64, shape=[batch_size, 10, 3])
# 两层RNN网络
cell = rnn.MultiRNNCell([rnn.BasicRNNCell(3) for itr in range(2)], state_is_tuple=True)
state = cell.zero_state(batch_size, tf.float64)
outputs = []
states = []
# 获取每一步输出，与状态
for time_step in range(max_time):
    (cell_output, state) = cell(indata[:, time_step, :], state)
    outputs.append(cell_output)
    states.append(state)
y = tf.placeholder(tf.float64, shape=[batch_size, 3])
# 定义loss函数
loss = tf.square(outputs[-1]-y) + tf.square(outputs[-2]-y)
opt = tf.train.GradientDescentOptimizer(1)
# 获取可训练参数
weights = tf.trainable_variables()
# 计算梯度
grad = opt.compute_gradients(loss, weights)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 获取变量值与梯度
w1, b1, w2, b2 = sess.run(weights)
dw1, db1, dw2, db2 = sess.run(grad, feed_dict={indata:np.ones([batch_size, 10, 3]), y:np.ones([batch_size, 3])})
dw1 = dw1[0]
db1 = db1[0]
dw2 = dw2[0]
db2 = db2[0]
rnn1 = RNNCell(3, 3)
rnn1.assign(w1, b1)
rnn2 = RNNCell(3, 3)
rnn2.assign(w2, b2)
state = []
state.append(rnn1.zero_state(batch_size))
state.append(rnn2.zero_state(batch_size))
rnn = MultiRNNCells([rnn1, rnn2])
indata = np.ones([batch_size, 10, 3])
for time_step in range(max_time):
    (cell_output, state) = rnn(indata[:, time_step, :], state)
    print(cell_output)
print("TF Gradients", np.mean(dw1), np.mean(db1), np.mean(dw2), np.mean(db2))
rnn.loss(np.ones([batch_size, 3]))
dw2, db2, dw1, db1 = rnn.backward()
print("NP Gradinets", np.mean(dw1, np.mean(db1), np.mean(dw2), np.mean(db2))
```

Thanks for https://zhuanlan.zhihu.com/p/37025766


```python

```
