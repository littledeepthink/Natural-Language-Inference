# coding: utf-8
from xlrd import open_workbook
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split

def load_pretrained_wordvector(filename):
    vocab, embed = [], []
    file = open(filename, 'r', encoding='utf-8')
    id = 0
    for line in file.readlines():
        temp = line.strip().split()
        if id > 0:
            vocab.append(temp[0])
            embed.append(temp[1:])
        else:
            n_embed = int(temp[1])
            # vocab.append('unk')
            embed.append([0] * n_embed)
        id += 1
    # print('Great! Loaded word_embedding successfully !')
    file.close()
    return vocab, embed

def get_preli_data(excel_filename, sheet, char2index, label2index, left_idx=7, right_idx=8, label_idx=10):
    wb = open_workbook(excel_filename)
    data = wb.sheet_by_name(sheet_name=sheet)
    left_text, right_text = [], []
    label = []
    for num in range(1, data.nrows):
        row = data.row_values(num)
        if row:
            left_des = [char2index.get(s, 0) for s in row[left_idx].strip() if s != ' ']
            right_des = [char2index.get(s, 0) for s in row[right_idx].strip() if s != ' ']
            tag = label2index[row[label_idx]]
            left_text.append(left_des)
            right_text.append(right_des)
            label.append([tag])  # if OneHotVector, append 'list'; else ‘scalar’
            # exchange the order of left_des/right_des
            left_text.append(right_des)
            right_text.append(left_des)
            label.append([tag])
    print('Read data successfully!!')
    return left_text, right_text, label

def get_actual_length(data):
    length = []
    for lst in data:
        lg = min(len(lst), 100)
        length.append(lg)
    return np.array(length)

def count(data, length):
    ct = []
    num = 0
    for lst in data:
        ct.append(len(lst))
        if len(lst) > length:
            num += 1
    print('max langth is {}'.format(max(ct)))  # 293
    print('min langth is {}'.format(min(ct)))  # 4
    print('The ratio of >length is {:.2%}'.format(num/len(ct)))

def get_standard_data(left_data, right_data, y_data, max_length):
    left_arr = pad_sequences(left_data, maxlen=max_length, dtype='int32', padding='post', truncating='post', value=0)
    right_arr = pad_sequences(right_data, maxlen=max_length, dtype='int32', padding='post', truncating='post', value=0)
    enc = OneHotEncoder(n_values=2, dtype='float32')
    y_arr = enc._fit_transform(y_data).toarray() # OneHotVector
    # y_arr = np.array(y_data, dtype='int32')
    return left_arr, right_arr, y_arr

def data_split(left_arr, right_arr, y_arr):
    X_arr = np.hstack([left_arr, right_arr])
    X_train, X_dev, y_train, y_dev = train_test_split(X_arr, y_arr, test_size = 0.15, random_state = 22)
    left_train, right_train = X_train[:, :100], X_train[:,100:]
    left_dev, right_dev = X_dev[:, :100], X_dev[:,100:]
    return left_train, right_train, y_train, left_dev, right_dev, y_dev

def data_split_ver2(X_arr, y_arr):
    X_train, X_dev, y_train, y_dev = train_test_split(X_arr, y_arr, test_size = 0.15, random_state =22)
    return X_train, y_train, X_dev, y_dev

if __name__ == '__main__':
    folder_path = os.getcwd() + '\\related_data'

    wv_path = folder_path + '\wiki_100_utf8.txt'
    vocab, embed = load_pretrained_wordvector(wv_path)
    char2index = {w: i for i, w in enumerate(vocab, 1)}
    char2index['<UNK>'] = 0
    index2char = {i: c for c, i in char2index.items()}
    print(len(vocab), len(embed)) # 16115, 16116
    print(index2char[0], embed[2])

    label2index = {'不矛盾':0, '矛盾':1}

    filename = folder_path + '\contradiction_data.xlsx'
    left_data, right_data, label = get_preli_data(filename, u'Sheet2', char2index, label2index)
    print(len(left_data))
    # print(set(label))

    # count(left_text, 100); count(right_text, 100)
    left_a_l = get_actual_length(left_data)
    right_a_l = get_actual_length(right_data)

    all_a_l = data_split_ver2(left_a_l, right_a_l)
    # print(all_a_l[2].shape)
    id = 0
    for name in ['_train.npy', '_dev.npy']:
        for pre in ['\left_a_l', '\\right_a_l']:
            np.save(folder_path + pre + name, all_a_l[id])
            id += 1

    left_arr, right_arr, y_arr = get_standard_data(left_data, right_data, label, max_length=100)

    all_data = data_split(left_arr, right_arr, y_arr)
    # print(all_data[2].shape)

    id = 0
    for name in ['_train.npy', '_dev.npy']:
        for pre in ['\left', '\\right', '\y']:
            np.save(folder_path + pre  + name, all_data[id])
            id += 1