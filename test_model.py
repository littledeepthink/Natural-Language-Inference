# coding: utf-8
from BiLSTM_Att import BiLSTMAtt
from Att_mRNN import AttmRNN
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
import os
from preprocess import load_pretrained_wordvector
from keras.preprocessing.sequence import pad_sequences
from preprocess import *

def standard(texts, char2index, max_length):
    lst = []
    length = []
    for text in texts:
        temp = [char2index.get(s, 0) for s in text.strip() if s != ' ']
        lst.append(temp)
        length.append(len(temp))
    arr = pad_sequences(lst, maxlen=max_length, dtype='int32', padding='post', truncating='post', value=0)
    a_l = np.array(length)
    return arr, a_l

def predict(seq_length, n_vocab, n_embed, n_hidden, n_classes, batch_size, learning_rate, optimizer, left_test,
            right_test, y_test, left_a_l_test, right_a_l_test, index2char, dropout=1.0, m_type='BiLSTM_Att'):
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
        saver_path = os.getcwd() + '\checkpoint\\' + m_type
        saver = tf.train.Saver(max_to_keep=5)
        model_file = tf.train.latest_checkpoint(saver_path)
        # model_file = saver_path + '\\vali_loss_0.0014.ckpt'
        saver.restore(sess, model_file)

        print('Start Testing...')
        feed_dict = {bilstmatt.X_left: left_test, bilstmatt.X_right: right_test, bilstmatt.y: y_test,
                     bilstmatt.dropout_keep_prob: dropout,
                     bilstmatt.left_actual_length: left_a_l_test,
                     bilstmatt.right_actual_length: right_a_l_test}
        acc_test, logits_test = sess.run([bilstmatt.accuracy, bilstmatt.logits], feed_dict=feed_dict)

        y_pred = np.argmax(logits_test, 1)
        y_true = np.argmax(y_test, 1)
        f1_test = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_true))
        auc_test = roc_auc_score(y_true, y_pred, average='weighted')
        for id in range(len(y_true)):
            if y_true[id] !=  y_pred[id]:
                left_text = ''.join([index2char[idx] for idx in left_test[id]])
                right_text = ''.join([index2char[idx] for idx in right_test[id]])
                print('Left_text: {0[0]} / Right_text: {0[1]}'.format([left_text, right_text]))
                print('The true label is {0[0]} / The pred label is {0[1]}'.format([y_true[id], y_pred[id]]))
        print('The test accuracy / f1 / auc: {0[0]:.2%} / {0[1]:.4f} / {0[2]:.4f}'.format((acc_test, f1_test, auc_test)))

if __name__ == '__main__':
    folder_path = os.getcwd() + '\\related_data'

    # Test: use the pre-splited test data
    # lst = ['\left_test.npy', '\\right_test.npy', '\y_test.npy']
    # left_test, right_test, y_test = (np.load(folder_path + name) for name in lst)
    # print(y_test.shape)
    #
    # # Load pre-trained word_embedding
    # char_embed_path = folder_path + '\char_embed_matrix.npy'
    # if os.path.exists(char_embed_path):
    #     char_embed_matrix = np.load(char_embed_path)
    # else:
    #     wv_path = folder_path + '\wiki_100_utf8.txt'
    #     vocab, embed = load_pretrained_wordvector(wv_path)
    #     char_embed_matrix = np.asarray(embed, dtype='float32')
    #     np.save(char_embed_path, char_embed_matrix)

    # char to index
    wv_path = folder_path + '\wiki_100_utf8.txt'
    vocab, embed = load_pretrained_wordvector(wv_path)
    char_embed_matrix = np.asarray(embed, dtype='float32')
    char2index = {w: i for i, w in enumerate(vocab, 1)}
    char2index['<UNK>'] = 0
    index2char = {i: c for c, i in char2index.items()}

    # # the actual length of test data
    # lst = ['\left_a_l_test.npy', '\\right_a_l_test.npy']
    # left_a_l_test, right_a_l_test = (np.load(folder_path + name) for name in lst)


    # Test: use the extra test data
    label2index = {'不矛盾': 0, '矛盾': 1}

    filename = folder_path + '\\test_data.xlsx'
    left_data, right_data, label = get_preli_data(filename, u'全量', char2index, label2index, left_idx=3, right_idx=4, label_idx=6)
    left_a_l_test = get_actual_length(left_data)
    right_a_l_test = get_actual_length(right_data)
    left_test, right_test, y_test = get_standard_data(left_data, right_data, label, max_length=100)

    predict(seq_length=100, n_vocab=16116, n_embed=100, n_hidden=50, n_classes=2, batch_size=64, learning_rate=0.001,
            optimizer='Adam', left_test=left_test, right_test=right_test, y_test=y_test, left_a_l_test=left_a_l_test,
            right_a_l_test=right_a_l_test, index2char=index2char, dropout=1.0, m_type='BiLSTM_Att')
