# coding: utf-8
from BiLSTM_Att import BiLSTMAtt
from Att_mRNN import AttmRNN
import os
import numpy as np
import tensorflow as tf
from preprocess import load_pretrained_wordvector

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_index = 0
epoch = 1
def next_batch(X_left, X_right, y, left_a_l, right_a_l, batch_size):
    global epoch
    global batch_index

    start = batch_index
    n_example = X_left.shape[0]
    batch_index += batch_size

    if batch_index >= n_example:
        epoch += 1 # run the new epoch
        batch_index = 0
        start = batch_index
        batch_index += batch_size
        rand = [i for i in range(n_example)]
        np.random.shuffle(rand)
        X_left = X_left[rand]
        X_right = X_right[rand]
        y = y[rand]
        left_a_l = left_a_l[rand]
        right_a_l = right_a_l[rand]

    assert batch_size < n_example
    end = batch_index

    return X_left[start: end], X_right[start: end], y[start: end], left_a_l[start: end], right_a_l[start: end]

def train(seq_length, n_vocab, n_embed, n_hidden, n_classes, batch_size, learning_rate, optimizer, left_train,
          right_train, y_train, left_a_l_train, right_a_l_train, is_dev, left_dev, right_dev, y_dev, left_a_l_dev,
          right_a_l_dev, dropout=0.5, char_embed_matrix=None, train_epochs=100, m_type='BiLSTM_Att'):
    with tf.Session() as sess:
        # Instantiate model
        if m_type == 'BiLSTM_Att':
            bilstmatt = BiLSTMAtt(seq_length, n_vocab, n_embed, n_hidden, n_classes,
                                  batch_size, learning_rate, optimizer)
        elif m_type == 'Att_mLSTM':
            bilstmatt = AttmRNN(seq_length, n_vocab, n_embed, n_hidden, n_classes,
                                 batch_size, learning_rate, optimizer, m_type)
        elif m_type == 'Att_mGRU':
            bilstmatt = AttmRNN(seq_length, n_vocab, n_embed, n_hidden, n_classes,
                                batch_size, learning_rate, optimizer, m_type)
        # Initialize Save
        saver = tf.train.Saver(max_to_keep=5)
        saver_path = os.getcwd() + '\checkpoint\\' + m_type
        print('Start Training......')
        sess.run(tf.global_variables_initializer())

        # Feed data & Training & Visualization
        train_losses, dev_losses = [], []
        train_accuracies, dev_accuracies = [], []
        step_range = []
        display_step = 10
        total_step = 0
        min_loss = 100
        while epoch <= train_epochs:
            batches = next_batch(left_train, right_train, y_train, left_a_l_train, right_a_l_train, batch_size)
            X_left_batch, X_right_batch, y_batch, left_a_l_batch, right_a_l_batch = batches
            step = batch_index // batch_size
            if step % display_step ==0:
                feed_dict = {bilstmatt.X_left: X_left_batch, bilstmatt.X_right: X_right_batch, bilstmatt.y: y_batch,
                             bilstmatt.dropout_keep_prob: dropout, bilstmatt.is_train: True,
                             bilstmatt.char_embed_matrix: char_embed_matrix,
                             bilstmatt.left_actual_length: left_a_l_batch,
                             bilstmatt.right_actual_length: right_a_l_batch}
                train_loss, train_accuracy = sess.run([bilstmatt.loss_val, bilstmatt.accuracy], feed_dict=feed_dict)
                if (is_dev):
                    feed_dict = {bilstmatt.X_left: left_dev, bilstmatt.X_right: right_dev, bilstmatt.y: y_dev,
                                 bilstmatt.dropout_keep_prob: 1.0, bilstmatt.char_embed_matrix: char_embed_matrix,
                                 bilstmatt.is_train: None,
                                 bilstmatt.left_actual_length: left_a_l_dev,
                                 bilstmatt.right_actual_length: right_a_l_dev}
                    dev_loss, dev_accuracy = sess.run([bilstmatt.loss_val, bilstmatt.accuracy], feed_dict=feed_dict)
                    print('Epoch %d: train_loss / dev_loss => %.4f / %.4f for step %d' % (epoch, train_loss, dev_loss, step))
                    print('Epoch {0[0]}: train_accuracy / dev_accuracy => {0[1]:.2%} / {0[2]:.2%} for step {0[3]}'.format(
                        (epoch, train_accuracy, dev_accuracy, step)))

                    if dev_loss < min_loss:
                        saver.save(sess, saver_path + '\\vali_loss_{:.4f}.ckpt'.format(dev_loss))
                        min_loss = dev_loss

                    dev_losses.append(dev_loss)
                    dev_accuracies.append(dev_accuracy)

                else:
                    print('Epoch %d: train_loss => %.4f for step %d' % (epoch, train_loss, step))
                    print('Epoch {0[0]}: train_accuracy => {0[1]:.2%} for step {0[2]}'.format(epoch, train_accuracy, step))
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                total_step += display_step
                step_range.append(total_step)

            # train on batch
            feed_dict = {bilstmatt.X_left: X_left_batch, bilstmatt.X_right: X_right_batch, bilstmatt.y: y_batch,
                         bilstmatt.dropout_keep_prob: dropout, bilstmatt.is_train: True,
                         bilstmatt.char_embed_matrix: char_embed_matrix,
                         bilstmatt.left_actual_length: left_a_l_batch,
                         bilstmatt.right_actual_length: right_a_l_batch}
            sess.run(bilstmatt.train_op, feed_dict=feed_dict)

    sess.close()


if __name__ == '__main__':
    fold_path = os.getcwd() + '/related_data'

    # Load pre-trained word_embedding
    char_embed_path = fold_path + '/char_embed_matrix.npy'
    if os.path.exists(char_embed_path):
        char_embed_matrix = np.load(char_embed_path)
    else:
        wv_path = fold_path + '/wiki_100_utf8.txt'
        vocab, embed = load_pretrained_wordvector(wv_path)
        char_embed_matrix = np.asarray(embed, dtype='float32')
        np.save(char_embed_path, char_embed_matrix)

    # Load train&dev data
    lst1 = ['_train.npy', '_dev.npy']
    lst2 = ['/left', '/right', '/y']
    data_loaded = (np.load(fold_path + pre + name) for name in lst1 for pre in lst2)
    left_train, right_train, y_train, left_dev, right_dev, y_dev = data_loaded
    print(left_train.shape)

    lst3 = ['/left_a_l', '/right_a_l']
    a_l_loaded = (np.load(fold_path + pre + name) for name in lst1 for pre in lst3)
    left_a_l_train, right_a_l_train, left_a_l_dev, right_a_l_dev = a_l_loaded

    train(seq_length=100, n_vocab=16116, n_embed=100, n_hidden=50, n_classes=2, batch_size=64, learning_rate=0.001,
          is_dev=True, optimizer='Adam', left_train=left_train, right_train=right_train, y_train=y_train,
          left_a_l_train=left_a_l_train, right_a_l_train=right_a_l_train, left_dev=left_dev, right_dev=right_dev,
          y_dev=y_dev, left_a_l_dev=left_a_l_dev, right_a_l_dev=right_a_l_dev, dropout=0.5,
          char_embed_matrix=char_embed_matrix, train_epochs=500, m_type='BiLSTM_Att')