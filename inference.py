# coding: utf-8
from BiLSTM_Att import BiLSTMAtt
import numpy as np
import tensorflow as tf
import os
from preprocess import load_pretrained_wordvector
from keras.preprocessing.sequence import pad_sequences
from preprocess import *
"""
输入两个(pair)字符串格式的文本，预测两者矛盾与否，将结果输出。
"""
def standard(text, char2index, max_length):
    lst = []
    temp = [char2index.get(s, 0) for s in text.strip() if s != ' ']
    lst.append(temp)
    arr = pad_sequences(lst, maxlen=max_length, dtype='int32', padding='post', truncating='post', value=0)
    a_l = np.array([len(temp)])
    return arr, a_l

def inference(sent1, sent2, seq_length=100, n_vocab=16116, n_embed=100, n_hidden=50, n_classes=2,
              batch_size=64, learning_rate=0.001, optimizer='Adam', dropout=1.0):
    folder_path = os.getcwd() + '\\related_data'
    wv_path = folder_path + '\wiki_100_utf8.txt'
    vocab, embed = load_pretrained_wordvector(wv_path)
    char_embed_matrix = np.asarray(embed, dtype='float32')
    char2index = {w: i for i, w in enumerate(vocab, 1)}
    char2index['<UNK>'] = 0
    left_test, left_a_l_test = standard(sent1, char2index, 100)
    right_test, right_a_l_test = standard(sent2, char2index, 100)
    index2label = {0: '不矛盾', 1: '矛盾'}
    with tf.Session() as sess:
        # Instantiate model
        bilstmatt = BiLSTMAtt(seq_length, n_vocab, n_embed, n_hidden, n_classes,
                              batch_size, learning_rate, optimizer)
        saver_path = os.getcwd() + '\checkpoint\\BiLSTM_Att'
        saver = tf.train.Saver(max_to_keep=5)
        model_file = tf.train.latest_checkpoint(saver_path)
        saver.restore(sess, model_file)

        feed_dict = {bilstmatt.X_left: left_test, bilstmatt.X_right: right_test, bilstmatt.dropout_keep_prob: dropout,
                     bilstmatt.left_actual_length: left_a_l_test, bilstmatt.right_actual_length: right_a_l_test}
        logits_test = sess.run(bilstmatt.logits, feed_dict=feed_dict)
        y_pred = np.argmax(logits_test, 1)
        return index2label[y_pred[0]]

if __name__ == '__main__':

    sent1 = '本产品的起价适用持中国居民身份证或中国护照的游客，持其他国家或地区证件的游客请选择对应的选项补足差额'
    sent2 = '本产品接受非大陆籍客人预定'

    result = inference(sent1, sent2)
    print(result)