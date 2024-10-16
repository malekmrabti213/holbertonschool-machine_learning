#!/usr/bin/env python3
"""
    FINAL
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """performs semantic search on a corpus of documents:"""
    # fetch file
    reference = []
    reference.append(sentence)
    dirs = os.listdir(corpus_path)
    for file in dirs:
        if not file.endswith('.md'):
            continue
        with open(corpus_path+'/'+file, 'r', encoding='utf-8') as f:
            reference.append(f.read())
    e = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    embeddings = e(reference)
    corr = np.inner(embeddings, embeddings)
    close = np.argmax(corr[0, 1:])

    return reference[close + 1]



def q_a(question, reference):
    """
        finds a snippet of text within a reference document
        to answer a question

    :param question: string, question to answer
    :param reference: string, reference document from which find answer

    :return: string, answer
        if no answer: None
    """
    # load tokenizer & model
    qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word'
                                                '-masking-finetuned-squad',
                                                clean_up_tokenization_spaces=True)
    qa_model = (
        hub.load("https://www.kaggle.com/models/seesee/bert/"
                "TensorFlow2/uncased-tf2-qa/1"))
    
    # tokenize inputs
    q_tokenized = qa_tokenizer.tokenize(question)
    ref_tokenized = qa_tokenizer.tokenize(reference)

    ###########################
    # PREPROCESS INPUT TOKENS #
    ###########################

    # concatenate token with special tokens
    tokens = ['[CLS]'] + q_tokenized + ['[SEP]'] + ref_tokenized + ['[SEP]']

    # conversion in numerical ids
    input_word_ids = qa_tokenizer.convert_tokens_to_ids(tokens)

    # create tensor mask : 1 to consider this token
    input_mask = [1] * len(input_word_ids)
    # mask to separate question - reference
    input_type_ids = ([0] * (1 + len(q_tokenized) + 1) +
                      [1] * (len(ref_tokenized) + 1))

    # conversion in TensorFlow tensor
    input_word_ids, input_mask, input_type_ids = (
        map(lambda t: tf.expand_dims(
            tf.convert_to_tensor(t,
                                 dtype=tf.int32), 0),
            (input_word_ids, input_mask, input_type_ids)))

    ###########################
    #   INFERENCE WITH BERT   #
    ###########################

    # pass in the model
    outputs = qa_model([input_word_ids, input_mask, input_type_ids])
    # outputs = proba of each token to be start and end
    # of the answer in given doc

    ###########################
    #    EXTRACTION ANSWER    #
    ###########################

    # found the most probably start token (+1 because ignore CLS)
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    # found most probably end token (+1 because ignore CLS)
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    # extraction of token corresponding to answer
    answer_tokens = tokens[short_start: short_end + 1]
    if not answer_tokens:
        return None

    # conversion answer token in understandable string
    answer = qa_tokenizer.convert_tokens_to_string(answer_tokens)

    return answer


def question_answer(corpus_path):
    """
        Answers questions from multiple reference texts

    :param corpus_path: path to the corpus of reference documents

    :return: answer to the question
    """

    while True:
        question = input("Q: ")
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        # first semantic search
        reference = semantic_search(corpus_path, question)

        # second QA
        answer = q_a(question, reference)

        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
