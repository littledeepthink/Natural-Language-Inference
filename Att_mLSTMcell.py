# coding: utf-8
import tensorflow as tf

class Att_mLSTMcell(object):
    def __init__(self, input, premise, d_input, d_premise, d_cell, d_att, initializer, f_bias=1.0, l2=False,
                 cell_act=tf.tanh, init_h=None, init_c=None):
        # var
        self.input = input  # shape: [n_steps, n_samples, d_input],输入前对原始数据进行了变化处理
        self.premise = premise
        self.d_input = d_input
        self.d_premise = d_premise
        self.d_cell = d_cell
        self.d_att = d_att
        self.initializer = initializer
        self.f_bias = f_bias  # 遗忘门的偏置项单独设定,取值较大
        self.cell_act = cell_act  # 将当期时刻cell_state的信息取值转换到区间[-1,1]上

        self.type = 'lstm' # Just for the using of 'tf.scan'

        if init_h is None and init_c is None:  # 判断是否提供了cell_state和hidden_state的初始值
            # If init_h and init_c are not provided, initialize them
            # the shape of init_h and init_c is [n_samples, d_cell]
            self.init_h = tf.matmul(self.input[0, :, :], tf.zeros([self.d_input, self.d_cell]))
            self.init_c = self.init_h
            self.previous = tf.stack([self.init_h, self.init_c])

        # parameters, each of which has W_x W_h b
        self.fgate = self.Gate(bias=self.f_bias)
        self.igate = self.Gate()
        self.ogate = self.Gate()
        self.cell = self.Gate()

        # to speed up computation. W_x: [d_input+d_premise, 4*d_cell], W_h: [d_cell, 4*d_cell], b: [4*d_cell,]
        # W_x = [W_xf, W_xi, W_xo, W_xc]
        # W_h = [W_hf, W_hi, W_ho, W_hc]
        # b = ( [b_f.T, b_i.T, b_o.T, b_c.T] ).T
        self.W_x = tf.concat([self.fgate[0], self.igate[0], self.ogate[0], self.cell[0]], axis=1)  # 按行对齐
        self.W_h = tf.concat([self.fgate[1], self.igate[1], self.ogate[1], self.cell[1]], axis=1)
        self.b = tf.concat([self.fgate[2], self.igate[2], self.ogate[2], self.cell[2]], axis=0)  # 按列对齐

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
        return W[:, n * self.d_cell: (n + 1) * self.d_cell]  # 选取给定输入的特定列

    def Step(self, previous_h_c_tuple, current_state_right):

        # to split hidden state and cell
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)

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
        gates = tf.matmul(current_x, self.W_x) + tf.matmul(prev_h, self.W_h) + self.b  # [-1, 4*d_cell]

        # computing (6steps)
        # forget Gate
        f = tf.sigmoid(self.Slice_W(gates, 0))  # 提取gates的前d_cell列, 上一时刻cell_state的信息保留比例
        # input gate
        i = tf.sigmoid(self.Slice_W(gates, 1))  # 更新信息的比例
        # output Gate
        o = tf.sigmoid(self.Slice_W(gates, 2))  # 输出信息的比例
        # new cell info
        c = tf.tanh(self.Slice_W(gates, 3))  # 更新的信息
        # current cell
        current_c = f * prev_c + i * c  # element-wise multiplication
        # current hidden state
        current_h = o * self.cell_act(current_c)

        return tf.stack([current_h, current_c])