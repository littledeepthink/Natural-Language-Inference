# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper

class BiLSTMAtt(object):
    def __init__(self, seq_length, n_vocab, n_embed, n_hidden, n_classes, batch_size, learning_rate, optimizer):
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.X_left = tf.placeholder(tf.int32, [None, self.seq_length], 'X_left_true')
        self.X_right = tf.placeholder(tf.int32, [None, self.seq_length], 'X_right_true')
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], 'y_true')
        self.left_actual_length = tf.placeholder(tf.int32, [None], 'left_actual_length')
        self.right_actual_length = tf.placeholder(tf.int32, [None], 'right_actual_length')
        self.char_embed_matrix = tf.placeholder(tf.float32, [self.n_vocab, self.n_embed], 'char_embed_matrix')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        self.logits = self.model()
        self.loss_val = self.loss()
        self.train_op = self.train(self.optimizer, clip_norm=5)

        tf.add_to_collection('train_mini', self.train_op)

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
        self.bilstm_right = tf.concat(outputs_right, axis=2)

        # Subtraction of two semantic representation
        # self.sub_bilstm = tf.subtract(self.bilstm_left, self.bilstm_right) # [-1, seq_length, n_hidden*2]

        # final_state = self.sub_bilstm[:, -1, :] # the final step state of BiLSTM

        # Extract the most important semantic information from the right text using the global_max_pooling
        bilstm_right = tf.expand_dims(self.bilstm_right, axis=-1)
        pool = tf.nn.max_pool(bilstm_right, ksize=[1, self.seq_length, 1, 1], strides=[1,1,1,1],
                              padding='VALID', name='max_pool')
        self.state_right = tf.reshape(pool, shape=[-1, self.n_hidden*2])

        # Query vector, [-1, n_hidden], based on the state_right
        W_c = tf.get_variable('W_c', [self.n_hidden*2, self.n_hidden], tf.float32)
        q = tf.nn.tanh(tf.matmul(self.state_right, W_c))

        # Attention weights(additive attention), based on bilstm_left and q
        v = tf.get_variable('v', [self.n_hidden], tf.float32)
        W_q = tf.get_variable('W_q', [self.n_hidden, self.n_hidden], tf.float32)
        W_k = tf.get_variable('W_k', [self.n_hidden*2, self.n_hidden], tf.float32)
        temp = tf.nn.tanh(tf.expand_dims(tf.tensordot(q, W_q, axes=1), axis=1) + tf.tensordot(self.bilstm_left, W_k, axes=1))
        scores = tf.tensordot(temp, v, axes=1) # [-1, seq_length, n_hidden] 'dot' [n_hidden,] => [-1, seq_length]
        alphas = tf.nn.softmax(scores) # [-1, seq_length]

        # Context vector, weighted-sum by the second dim
        context = tf.reduce_sum(self.bilstm_left * tf.expand_dims(alphas, axis=-1), axis=1) # [-1, n_hidden*2]

        # Attention vector
        W_att1 = tf.get_variable('W_att1', [self.n_hidden, self.n_hidden], dtype=tf.float32)
        W_att2 = tf.get_variable('W_att2', [self.n_hidden*2, self.n_hidden], dtype=tf.float32)
        all_of_states = tf.nn.tanh(tf.matmul(q, W_att1) + tf.matmul(context, W_att2))  # [-1, n_hidden]

        # output
        W_output = tf.get_variable('W_output', [self.n_hidden, self.n_classes], tf.float32)
        b_output = tf.get_variable('b_output', [self.n_classes], tf.float32)
        logits = tf.nn.xw_plus_b(all_of_states, W_output, b_output, name='logits')
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope('cost'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits, name='cross_entropy')
            loss = tf.reduce_mean(losses, name='loss_val')
            weights = [v for v in tf.trainable_variables() if ('W' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
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
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.loss_val, tvars), clip_norm=clip_norm)
        train_op = Opt.apply_gradients(zip(grads, tvars))
        return train_op


if __name__ == '__main__':
    model = BiLSTMAtt(seq_length=100, n_vocab=16116, n_embed=100, n_hidden=50, n_classes=2, batch_size=64,
                      learning_rate=0.001, optimizer='Adam')
    w = model.weights
    print(w, len(w))