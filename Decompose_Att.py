# coding: utf-8
import tensorflow as tf

class DecomposeAtt(object):
    def __init__(self, len_left, len_right, n_vocab, n_embed, n_units_intra, n_units_att, n_units_compare,
                 n_units_agg, n_classes, optimizer, learning_rate):
        self.len_left = len_left
        self.len_right = len_right
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        # self.n_unit_proj = n_unit_proj
        self.n_units_intra = n_units_intra
        self.n_units_att = n_units_att
        self.n_units_compare = n_units_compare
        self.n_units_agg = n_units_agg
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.X_left = tf.placeholder(tf.int32, [None, self.len_left], name='X_left')
        self.X_right = tf.placeholder(tf.int32, [None, self.len_right], name='X_right')
        self.left_m = tf.placeholder(tf.float32, [None, self.len_left], name='left_m')
        self.right_m = tf.placeholder(tf.float32, [None, self.len_right], name='right_m')
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], name='y')
        self.char_embed_matrix = tf.placeholder(tf.float32, [self.n_vocab, self.n_embed], name='char_embed_matrix')
        self.dropout_prob = tf.placeholder(tf.float32, name='dropout')

        self.logits = self.model()
        self.loss_val = self.loss()
        self.train_op = self.train(self.optimizer, clip_norm=5)

        tf.add_to_collection('train_mini', self.train_op)

        label_pred = tf.argmax(self.logits, 1, name='label_pred')
        label_true = tf.argmax(self.y, 1, name='label_true')

        T_or_F = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(T_or_F, tf.float32), name='Accuracy')

    def model(self):
        """
        Build the computation graph, return the logits
        """
        # def linear_project(input, n_unit, reuse_weights=False):
        #     with tf.variable_scope('linear_proj', reuse=reuse_weights) as self.linear_proj:
        #         initializer = tf.variance_scaling_initializer(scale=1.0, mode='fan_avg', distribution='normal') # Xavier
        #         projected = tf.layers.dense(input, n_unit, kernel_initializer=initializer)
        #     return projected

        def feedforward(input, n_units, scope=None, reuse_weights=False, initializer=None):
            """
            :param n_units: list of length 2, containing the number of units in each layer
            """
            scope = scope or 'feedforward'
            with tf.variable_scope(scope, reuse=reuse_weights):
                if initializer is None:
                    initializer = tf.random_normal_initializer(0.0, 1.0)
                    with tf.variable_scope('layer1'):
                        drop = tf.nn.dropout(input, self.dropout_prob)
                        dense1 = tf.layers.dense(drop, n_units[0], tf.nn.relu, kernel_initializer=initializer)
                    with tf.variable_scope('layer2'):
                        drop = tf.nn.dropout(dense1, self.dropout_prob)
                        dense2 = tf.layers.dense(drop, n_units[1], tf.nn.relu, kernel_initializer=initializer)
            return dense2

        def intra_att(sent, sent_m, scope):
            """
            Augment the input representation(embedding) with intra-sentence attention to encode compositional
            relationships between words within each sentence.
            """
            with tf.variable_scope(scope) as self.intra_scope:
                repr = feedforward(sent, self.n_units_intra, self.intra_scope)
                # Mask: [-1, len_sent, len_sent]
                m = tf.multiply(tf.expand_dims(sent_m, axis=2), tf.expand_dims(sent_m, axis=1))

                # Unnormailized attention: [-1, len_sent, len_sent]
                raw_att = tf.matmul(repr, tf.transpose(repr, [0, 2, 1]))
                raw_att = tf.multiply(raw_att, m)

                # Normalization
                att = tf.exp(raw_att - tf.reduce_max(raw_att, axis=2, keep_dims=True))
                att = tf.multiply(att, tf.expand_dims(sent_m, axis=1)) # mask
                att = tf.divide(att, tf.reduce_sum(att, axis=2, keep_dims=True))
                att = tf.multiply(att, m) # mask

                context = tf.matmul(att, sent) # [-1, len_sent, n_embed]
                output = tf.concat([sent, context], axis=2) # concatenation, [-1, len_sent, n_embed*2]
            return output

        def attention(a, b):
            """
            Apply attention mechanism between the two sents in a decomposable way
            """
            with tf.variable_scope('att_scope') as self.att_scope:
                repr1 = feedforward(a, self.n_units_att, self.att_scope)
                repr2 = feedforward(b, self.n_units_att, self.att_scope, reuse_weights=True)
                # Mask: [-1, len_left, len_right]
                m1_m2 = tf.multiply(tf.expand_dims(self.left_m, axis=2), tf.expand_dims(self.right_m, axis=1))

                # Unnormalized attention: [-1, len_left, len_right]
                raw_att = tf.matmul(repr1, tf.transpose(repr2, [0, 2, 1]))
                raw_att = tf.multiply(raw_att, m1_m2)

                # Normalization for alpha and beta
                att_left = tf.exp(raw_att - tf.reduce_max(raw_att, axis=2, keep_dims=True))
                att_right = tf.exp(raw_att - tf.reduce_max(raw_att, axis=1, keep_dims=True))
                # mask
                att_left = tf.multiply(att_left, tf.expand_dims(self.left_m, axis=1))
                att_right = tf.multiply(att_right, tf.expand_dims(self.right_m, axis=2))
                # softmax
                att_left = tf.divide(att_left, tf.reduce_sum(att_left, axis=2, keep_dims=True))
                att_right = tf.divide(att_right, tf.reduce_sum(att_right, axis=1, keep_dims=True))
                # mask
                att_left = tf.multiply(att_left, m1_m2)
                att_right = tf.multiply(att_right, m1_m2)

                alpha = tf.matmul(att_left, b, name='alpha') # [-1, len_left, n_embed*2]
                beta = tf.matmul(tf.transpose(att_right, [0, 2, 1]), a, name='beta')
            return alpha, beta


        def compare(sent, soft_align, reuse_weights=False):
            """
            Apply feedforward to the concatenation of sent and its soft alignment
            """
            with tf.variable_scope('compare_scope', reuse=reuse_weights) as self.compare_scope:
                input = tf.concat([sent, soft_align], axis=2)
                output = feedforward(input, self.n_units_compare, self.compare_scope, reuse_weights)
            return output

        def aggregate(v1, v2):
            """
            Aggregate the representations induced from two sents to get the logits, using to calculate the loss
            """
            v1_sum = tf.reduce_sum(v1, axis=1) # sum by the second dim
            v2_sum = tf.reduce_sum(v2, axis=1)

            input = tf.concat([v1_sum, v2_sum], axis=1)

            with tf.variable_scope('agg_scoe') as self.agg_scope:
                hidden = feedforward(input, self.n_units_agg, self.agg_scope)
                output = tf.layers.dense(hidden, self.n_classes, name='logits')
            return output

        with tf.name_scope('embed'):
            self.Embedding = tf.Variable(self.char_embed_matrix, trainable=True, name='Embedding')
            self.embeded_left = tf.nn.embedding_lookup(self.Embedding, self.X_left)
            self.embeded_right = tf.nn.embedding_lookup(self.Embedding, self.X_right)

        with tf.name_scope('Att_Compare_Agg'):
            repr_left = intra_att(self.embeded_left, self.left_m, 'intra_left') # [-1, len_left, n_embed*2]
            repr_right = intra_att(self.embeded_right, self.right_m, 'intra_right')

            alpha, beta = attention(repr_left, repr_right) # [-1, len_left, n_embed*2]

            v_left = compare(repr_left, alpha) # [-1, len_left, n_units_compare[-1]]
            v_right = compare(repr_right, beta, reuse_weights=True)

            logits = aggregate(v_left, v_right) # [-1, n_classes]

        return logits

    def loss(self, l2_lambda=0.0001):
        if self.n_classes == 2:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        elif self.n_classes > 2:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        loss = tf.reduce_mean(cross_entropy, name='loss_val')
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name]) * l2_lambda
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
        else:
            ValueError('Unkown optimizer: {0}'.format(optimizer))
        grads, vars = zip(*Opt.compute_gradients(self.loss_val))
        cliped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
        train_op = Opt.apply_gradients(zip(cliped_grads, vars))
        return train_op


if __name__ == '__main__':

    model =  DecomposeAtt(len_left=100, len_right=100, n_vocab=100000, n_embed=100, n_units_intra=[100, 50],
                          n_units_att=[100, 50], n_units_compare=[100, 100], n_units_agg=[100, 50], n_classes=2)
    model.loss()







