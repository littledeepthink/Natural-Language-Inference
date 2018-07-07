# coding: utf-8
import tensorflow as tf
from BiLSTM_Att import BiLSTMAtt
import numpy as np
import os
from preprocess import *

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

def finetune(left_train, right_train, y_train, left_a_l_train, right_a_l_train, is_dev, left_dev, right_dev, y_dev,
             left_a_l_dev, right_a_l_dev, batch_size, dropout=0.5,train_epochs=100, m_type='BiLSTM_Att'):
    saver_path = os.getcwd() + '\checkpoint\\' + m_type
    ckpt = tf.train.get_checkpoint_state(saver_path)
    # 获取导入图的saver, 便于后面的restore
    saver_restore = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

    # 此时默认的图就是导入的图
    graph_restore = tf.get_default_graph()

    # 从导入图中获取需要的tensor: placeholder
    X_left = graph_restore.get_tensor_by_name('X_left_true:0')
    X_right = graph_restore.get_tensor_by_name('X_right_true:0')
    y = graph_restore.get_tensor_by_name('y_true:0')
    left_actual_length = graph_restore.get_tensor_by_name('left_actual_length:0')
    right_actual_length = graph_restore.get_tensor_by_name('right_actual_length:0')
    dropout_keep_prob = graph_restore.get_tensor_by_name('dropout_keep_prob:0')
    # 计算图中的char_embed_matrix和is_train用于刚开始训练时加载预训练好的字向量，在finetune时无需使用
    # char_embed_matrix = graph_restore.get_tensor_by_name('char_embed_matrix:0')
    # is_train = graph_restore.get_tensor_by_name('is_train:0')
    loss = graph_restore.get_tensor_by_name('cost/loss_val:0')
    acc = graph_restore.get_tensor_by_name('Accuracy:0')
    # 从导入图中获取需要的operation
    train_op = tf.get_collection('train_mini')[0]
    with tf.Session() as sess:
        # Restore the trained model
        model_file = tf.train.latest_checkpoint(saver_path)
        saver_restore.restore(sess, model_file)
        # Feed data & Training
        display_step = 10
        total_step = 0
        min_loss = float(model_file[-11:-5]) # 设置新的min_loss阈值
        print(min_loss)
        saver = tf.train.Saver(max_to_keep=5) # 设置新的Saver
        while epoch <= train_epochs:
            batches = next_batch(left_train, right_train, y_train, left_a_l_train, right_a_l_train, batch_size)
            X_left_batch, X_right_batch, y_batch, left_a_l_batch, right_a_l_batch = batches
            step = batch_index // batch_size
            if step % display_step == 0:
                feed_dict = {X_left: X_left_batch, X_right: X_right_batch, y: y_batch, dropout_keep_prob: dropout,
                             left_actual_length: left_a_l_batch, right_actual_length: right_a_l_batch}
                train_loss, train_accuracy = sess.run([loss, acc], feed_dict=feed_dict)
                if (is_dev):
                    feed_dict = {X_left: left_dev, X_right: right_dev, y: y_dev, dropout_keep_prob: 1.0,
                                 left_actual_length: left_a_l_dev, right_actual_length: right_a_l_dev}
                    dev_loss, dev_accuracy = sess.run([loss, acc], feed_dict=feed_dict)
                    print('Epoch %d: train_loss / dev_loss => %.4f / %.4f for step %d' % (
                    epoch, train_loss, dev_loss, step))
                    print(
                        'Epoch {0[0]}: train_accuracy / dev_accuracy => {0[1]:.2%} / {0[2]:.2%} for step {0[3]}'.format(
                            (epoch, train_accuracy, dev_accuracy, step)))

                    if dev_loss < min_loss:
                        saver.save(sess, saver_path + '\\vali_loss_{:.4f}.ckpt'.format(dev_loss))
                        min_loss = dev_loss
                else:
                    print('Epoch %d: train_loss => %.4f for step %d' % (epoch, train_loss, step))
                    print('Epoch {0[0]}: train_accuracy => {0[1]:.2%} for step {0[2]}'.format(epoch, train_accuracy,
                                                                                              step))
                total_step += display_step

            # train on batch
            feed_dict = {X_left: X_left_batch, X_right: X_right_batch, y: y_batch, dropout_keep_prob: dropout,
                         left_actual_length: left_a_l_batch, right_actual_length: right_a_l_batch}
            sess.run(train_op, feed_dict=feed_dict)

    sess.close()

if __name__ == '__main__':
    folder_path = os.getcwd() + '/related_data'

    # Load pre-trained word_embedding
    wv_path = folder_path + '\wiki_100_utf8.txt'
    vocab, embed = load_pretrained_wordvector(wv_path)
    char_embed_matrix = np.asarray(embed, dtype='float32')
    char2index = {w: i for i, w in enumerate(vocab, 1)}
    char2index['<UNK>'] = 0
    index2char = {i: c for c, i in char2index.items()}

    label2index = {'不矛盾': 0, '矛盾': 1}

    # Insert the new data to finetune the trained model
    filename = folder_path + '\\test_data.xlsx'
    # left_data, right_data, label = get_preli_data(filename, u'Sheet2', char2index, label2index)
    left_data, right_data, label = get_preli_data(filename, u'全量', char2index, label2index,
                                                  left_idx=3, right_idx=4, label_idx=6)
    left_new, right_new, y_new = get_standard_data(left_data, right_data, label, max_length=100)
    print(left_new.shape)
    left_a_l_new = get_actual_length(left_data)
    right_a_l_new = get_actual_length(right_data)

    # Load train&dev data
    lst1 = ['_dev.npy']
    lst2 = ['/left', '/right', '/y']
    data_loaded = (np.load(folder_path + pre + name) for name in lst1 for pre in lst2)
    left_dev, right_dev, y_dev = data_loaded
    print(left_dev.shape)
    lst3 = ['/left_a_l', '/right_a_l']
    a_l_loaded = (np.load(folder_path + pre + name) for name in lst1 for pre in lst3)
    left_a_l_dev, right_a_l_dev = a_l_loaded

    finetune(left_train=left_new, right_train=right_new, y_train=y_new, left_a_l_train=left_a_l_new,
             right_a_l_train=right_a_l_new, is_dev=True, left_dev=left_dev, right_dev=right_dev, y_dev=y_dev,
             left_a_l_dev=left_a_l_dev, right_a_l_dev=right_a_l_dev, batch_size=128, dropout=0.5,
             train_epochs=500, m_type='BiLSTM_Att')


    # Remarks:
    # Learning_rate的指数衰减
    # initial_lr = 0.001
    # global_step = tf.Variable(0, trainable=False, name='global_step')
    # learning_rate = tf.train.exponential_decay(initial_lr,
    #                                            global_step=global_step,
    #                                            decay_steps=decay_steps,
    #                                            staircase=True,
    #                                            decay_rate=0.1)
    # 在finetune时指定只更新部分变量
    # var_list = tf.contrib.framework.get_variables('scope_name')  # 获取指定scope下的变量
    # Opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    # train_op = Opt.minimize(loss, global_step=global_step, var_list=var_list)  # 只更新指定的variables