# coding: utf-8
import tensorflow as tf

class Att_mGRUcell(object):
    def __init__(self, input, premise, d_input, d_premise, d_cell, d_att, initializer, l2=False, init_h=None):
        # var
        self.input = input  # shape: [n_steps, n_samples, d_input],输入前对原始数据进行了变化处理
        self.premise = premise
        self.d_input = d_input
        self.d_premise = d_premise
        self.d_cell = d_cell
        self.d_att = d_att
        self.initializer = initializer

        self.type = 'gru' # Just for the using of 'tf.scan'

        if init_h is None:  # 判断是否提供了hidden_state的初始值
            # the shape of init_h is [n_samples, d_cell]
            self.init_h = tf.matmul(self.input[0, :, :], tf.zeros([self.d_input, self.d_cell]))
            self.previous = self.init_h # initial state

        # parameters, each of which has W_x W_h b
        self.rgate = self.Gate()
        self.zgate = self.Gate()
        self.hh = self.Gate()

        # to speed up computation. W_x: [d_input+d_premise, 3*d_cell], W_h: [d_cell, 3*d_cell], b: [3*d_cell,]
        # W_x = [W_xr, W_xz, W_xc]
        # W_h = [W_hr, W_hz, W_hc]
        # b = ( [b_r.T, b_z.T, b_c.T] ).T
        self.W_x = tf.concat([self.rgate[0], self.zgate[0], self.hh[0]], axis=1)  # 按行对齐
        self.W_h = tf.concat([self.rgate[1], self.zgate[1], self.hh[1]], axis=1)
        self.b = tf.concat([self.rgate[2], self.zgate[2], self.hh[2]], axis=0)  # 按列对齐

        # Query weight
        self.W_c = tf.get_variable('W_c', [self.d_input, self.d_att], tf.float32)

        # Attention weights(additive attention)
        self.v = tf.get_variable('v', [self.d_att], tf.float32)
        self.W_q = tf.get_variable('W_q', [self.d_att, self.d_att], tf.float32)
        self.W_k = tf.get_variable('W_k', [self.d_premise, self.d_att], tf.float32)
        self.W_m = tf.get_variable('W_m', [self.d_cell, self.d_att], tf.float32)

        if l2:
            lst_W = [self.W_x, self.W_h, self.W_c, self.W_q, self.W_k, self.W_m]
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in lst_W]) * 1e-4

    def Gate(self, bias=0.001):
        # Since we will use gate multiple times, let's code a class for reusing
        Wx = self.initializer([self.d_input + self.d_premise, self.d_cell])
        Wh = self.initializer([self.d_cell, self.d_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.d_cell]), trainable=True)
        return Wx, Wh, b

    def Slice_W(self, W, n):
        # split W's after computing
        return W[:, n * self.d_cell: (n + 1) * self.d_cell] # 选取给定输入的特定列

    def Step(self, prev_h, current_state_right):
        # Query vector, [-1, n_hidden], based on the state_right
        q = tf.nn.tanh(tf.matmul(current_state_right, self.W_c))

        # Attention weights(additive attention), based on premise, q and the preceding hidden state of mLSTM
        temp = tf.nn.tanh(tf.expand_dims(tf.tensordot(q, self.W_q, axes=1), axis=1)
                          + tf.tensordot(self.premise, self.W_k, axes=1)
                          + tf.expand_dims(tf.tensordot(prev_h, self.W_m, axes=1), axis=1))
        scores = tf.tensordot(temp, self.v, axes=1)  # [-1, seq_length, d_att] 'dot' [d_att,] => [-1, seq_length]
        alphas = tf.nn.softmax(scores)  # [-1, seq_length]

        # Context vector, weighted-sum by the second dim
        context = tf.reduce_sum(self.premise * tf.expand_dims(alphas, axis=2), axis=1)  # [-1, d_premise]

        current_x = tf.concat([context, current_state_right], axis=1)  # the input of mLSTM, [-1, d_input+d_premise]

        # computing all gates, 包含四个子网络的结果
        states_x = tf.matmul(current_x, self.W_x) + self.b  # [-1, 3*d_cell]
        states_h = tf.matmul(prev_h, self.W_h)

        # computing (4steps)
        r = tf.nn.sigmoid(self.Slice_W(states_x, 0) + self.Slice_W(states_h, 0)) # [-1, d_cell]
        z = tf.nn.sigmoid(self.Slice_W(states_x, 1) + self.Slice_W(states_h, 1))
        hh = tf.nn.tanh(self.Slice_W(states_x, 2) + r * self.Slice_W(states_h, 2)) # [-1, d_cell]
        current_h = (1-z) * prev_h + z * hh
        return current_h