# coding: utf-8
import tensorflow as tf
import numpy as np
from Att_mLSTMcell import Att_mLSTMcell
from Att_mGRUcell import Att_mGRUcell
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper

class AttmRNN(object):
    def __init__(self, seq_length, n_vocab, n_embed, n_hidden, n_classes, batch_size, learning_rate, optimizer, m_type):
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.m_type = m_type # 'Att_mLSTM' or 'Att_mGRU'

        self.X_left = tf.placeholder(tf.int32, [None, self.seq_length], 'X_left_true')
        self.X_right = tf.placeholder(tf.int32, [None, self.seq_length], 'X_right_true')
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], 'y_true')
        self.left_actual_length = tf.placeholder(tf.int32, [None], 'left_actual_length')
        self.right_actual_length = tf.placeholder(tf.int32, [None], 'right_actual_length')
        self.char_embed_matrix = tf.placeholder(tf.float32, [self.n_vocab, self.n_embed], 'char_embed_matrix')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        self.initializer = self.orthogonal_initializer
        self.logits = self.model()
        self.loss_val = self.loss()
        self.train_op = self.train(self.optimizer, clip_norm=5)

        tf.add_to_collection('train_mini', self.train_op) # 便于后续finetune时的导入

        label_pred = tf.argmax(self.logits, 1, name='label_pred')
        label_true = tf.argmax(self.y, 1, name='label_true')

        T_or_F = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(T_or_F, tf.float32), name='Accuracy')

    def model(self):
        self.Embedding = tf.get_variable('Embedding', [self.n_vocab, self.n_embed], tf.float32)
        if self.is_train is True:
            self.Embedding.assign(self.char_embed_matrix)
        self.embeded_left = tf.nn.embedding_lookup(self.Embedding, self.X_left)
        self.embeded_right = tf.nn.embedding_lookup(self.Embedding, self.X_right)

        lstm_cell = LSTMCell(self.n_hidden)
        lstm_drop_cell = lambda : DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)

        lstm_drop_f_cell = lstm_drop_cell()
        lstm_drop_b_cell = lstm_drop_cell()

        bilstm_outputs_left = tf.nn.bidirectional_dynamic_rnn(lstm_drop_f_cell,
                                                              lstm_drop_b_cell,
                                                              self.embeded_left,
                                                              sequence_length=self.left_actual_length,
                                                              dtype=tf.float32)
        bilstm_outputs_right = tf.nn.bidirectional_dynamic_rnn(lstm_drop_f_cell,
                                                              lstm_drop_b_cell,
                                                              self.embeded_right,
                                                              sequence_length=self.right_actual_length,
                                                              dtype=tf.float32)
        outputs_left, final_state_left  = bilstm_outputs_left
        outputs_right, final_state_right = bilstm_outputs_right

        # The semantic representation of premise and hypothesis
        self.bilstm_left = tf.concat(outputs_left, axis=2)
        # null_embed = tf.zeros([self.batch_size, 1, self.n_hidden*2]) # Insert a zero vector of new word 'NULL'
        # premise = tf.concat([null_embed, self.bilstm_left], axis=1)
        premise = self.bilstm_left

        self.bilstm_right = tf.concat(outputs_right, axis=2)
        hypothesis = tf.transpose(self.bilstm_right, perm=[1,0,2])

        if self.m_type == 'Att_mLSTM':
            # Attention + mLSTM
            self.att_mrnn_cell = Att_mLSTMcell(input=hypothesis, premise=premise, d_input=self.n_hidden * 2,
                                               d_premise=self.n_hidden * 2, d_cell=self.n_hidden, d_att=self.n_hidden,
                                               initializer=self.initializer, f_bias=1.0, l2=False)
        elif self.m_type == 'Att_mGRU':
            # Attention + mGRU
            self.att_mrnn_cell = Att_mGRUcell(input=hypothesis, premise=premise, d_input=self.n_hidden*2,
                                              d_premise=self.n_hidden*2, d_cell=self.n_hidden, d_att=self.n_hidden,
                                              initializer=self.initializer, l2=False, init_h=None)

        states_h = self.RNN(cell=self.att_mrnn_cell, cell_b=None, merge='concat') # Steps first !
        self.final_state_h = states_h[-1,:,:] # [-1, n_hidden]

        # output
        W_output = tf.get_variable('W_output', [self.n_hidden, self.n_classes], tf.float32)
        b_output = tf.get_variable('b_output', [self.n_classes], tf.float32)
        logits = tf.nn.xw_plus_b(self.final_state_h, W_output, b_output, name='logits')
        return logits

    def orthogonal_initializer(self, shape, scale=1.0):
        # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        return tf.Variable(scale * q[:shape[0], :shape[1]], dtype=tf.float32, trainable=True)

    # 计算并输出Att_mLSTMcell的hidden_state
    def RNN(self, cell, cell_b=None, merge='concat'):
        """
        Note that the input shape should be [n_steps, n_sample, d_input]   !!!!!
        and the output shape will also be [n_steps, n_sample, d_cell].
        If the original data has a shape of [n_sample, n_steps, d_input],
        use 'inputs_T = tf.transpose(inputs, perm=[1,0,2])'.                !!!!!
        """
        # The simplest version of 'scan' repeatedly applies the callable 'fn' to a sequence of elements from first to last.
        # The elements are made of the tensors unpacked from 'elems' on dimension 0.
        # forward rnn loop
        # If an ‘initializer’ is provided, then the output of ‘fn’ must have the same structure as ‘initializer’;
        # and the first argument of ‘fn’ must match this structure.
        hstates = tf.scan(fn=cell.Step,
                          elems=cell.input,
                          initializer=cell.previous,
                          name='hstates')
        if cell.type == 'lstm':
            hstates = hstates[:, 0, :, :]  # 提取出hidden_state部分

        # reverse the input sequence
        if cell_b is not None:
            input_b = tf.reverse(cell.input, axis=[0])  # 将第一个维度倒序

            # backward rnn loop
            b_hstates_rev = tf.scan(fn=cell_b.Step,
                                    elems=input_b,
                                    initializer=cell_b.previous,
                                    name='b_hstates')
            if cell_b.type == 'lstm':
                b_hstates_rev = b_hstates_rev[:, 0, :, :]

            b_hstates = tf.reverse(b_hstates_rev, axis=[0])  # 将第一个维度变换回原始顺序

            if merge == 'sum':
                hstates = hstates + b_hstates
            elif merge == 'concat':
                hstates = tf.concat(values=[hstates, b_hstates], axis=2)
        return hstates

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope('cost'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits, name='cross_entropy')
            loss = tf.reduce_mean(losses, name='loss_val')
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss += l2_loss
        return loss

    def train(self, optimizer, clip_norm=5):
        if optimizer == 'Adam':
            Opt = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer == 'RMSProp':
            Opt = tf.train.RMSPropOptimizer(self.learning_rate)
        elif optimizer == 'Momentum':
            Opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        elif optimizer == 'SGD':
            Opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        grads, vars = zip(*Opt.compute_gradients(self.loss_val))
        clip_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
        train_op = Opt.apply_gradients(zip(clip_grads, vars))
        return train_op