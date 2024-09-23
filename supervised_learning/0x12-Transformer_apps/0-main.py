#!/usr/bin/env python3

Dataset = __import__('0-dataset').Dataset
import tensorflow as tf

data = Dataset()

# Print a Portuguese and English sentence from the training dataset
for pt, en in data.data_train.take(1):
    print("Portuguese Sentence (Train):", pt.numpy().decode('utf-8'))
    print("English Sentence (Train):", en.numpy().decode('utf-8'))

# Print a Portuguese and English sentence from the validation dataset

for pt, en in data.data_valid.take(1):
    print("Portuguese Sentence (Valid):", pt.numpy().decode('utf-8'))
    print("English Sentence (Valid):", en.numpy().decode('utf-8'))

# Print the types of tokenizers
print("Type of Portuguese tokenizer:", type(data.tokenizer_pt))
print("Type of English tokenizer:", type(data.tokenizer_en))