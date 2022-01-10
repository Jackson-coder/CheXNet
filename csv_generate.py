# -*- coding:utf-8 -*-
"""
author: win10
date: 2021-11-26
"""
import csv

CLASS_NAMES = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3, 'Mass': 4, 'Nodule': 5,
               'Pneumonia': 6, 'Pneumothorax': 7, 'Consolidation': 8, 'Edema': 9, 'Emphysema': 10, 'Fibrosis': 11,
               'Pleural_Thickening': 12, 'Hernia': 13}

readFile1 = open("train_val_list.txt", "r", newline='')
reader1 = readFile1.readlines()
train_data_length = len(reader1)
for i in range(train_data_length):
    reader1[i] = reader1[i].strip('\n')

readFile2 = open("Data_Entry_2017.csv", "r", newline='')
reader2 = csv.reader(readFile2)

csvFile = open('train.csv', "a", newline='')
writer = csv.writer(csvFile)
writer.writerow(['Image_Index', 'Finding'])

train_pos = 0
for row in reader2:
    if reader1[train_pos] == row[0]:
        # print(reader1[train_pos], row[0])
        flags = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        classes = row[1].split('|')
        for c in classes:
            if c == 'No Finding':
                continue
            flags[CLASS_NAMES[c]] = '1'
        flags.insert(0, row[0])
        writer.writerow([f for f in flags])
        train_pos += 1
        if train_pos == train_data_length:
            break
