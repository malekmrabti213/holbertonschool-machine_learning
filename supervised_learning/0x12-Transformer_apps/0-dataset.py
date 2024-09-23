# #!/usr/bin/env python3
# """ Defines `Dataset`. """
# import tensorflow as tf
# import tensorflow_datasets as tfds

# class Dataset:
#     def __init__(self):
#         """
#         Class constructor to initialize the dataset.
#         """
#         # Load the TED Talks translation dataset (Portuguese to English)
#         self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
#         self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)

#         # Prepare tokenizers for Portuguese and English from the training dataset
#         self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

#     def tokenize_dataset(self, data):
#         """
#         Creates sub-word tokenizers for the dataset using TensorFlow's TextVectorization.

#         Args:
#         - data: A tf.data.Dataset whose examples are formatted as a tuple (pt, en)
#                 where pt is a tf.Tensor containing the Portuguese sentence and
#                 en is a tf.Tensor containing the corresponding English sentence.

#         Returns:
#         - tokenizer_pt: The Portuguese sub-word tokenizer
#         - tokenizer_en: The English sub-word tokenizer
#         """
#         # Extract Portuguese and English sentences for vocabulary building
#         pt_sentences = [pt.numpy().decode('utf-8') for pt, _ in data]
#         en_sentences = [en.numpy().decode('utf-8') for _, en in data]

#         # Create TextVectorization layers
#         tokenizer_pt = tf.keras.layers.TextVectorization(standardize=None, split='whitespace', max_tokens=2**15)
#         tokenizer_en = tf.keras.layers.TextVectorization(standardize=None, split='whitespace', max_tokens=2**15)

#         # Apply the tokenizer to build the vocabulary
#         tokenizer_pt.adapt(pt_sentences)
#         tokenizer_en.adapt(en_sentences)

#         return tokenizer_pt, tokenizer_en


from transformers import BertTokenizer
import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    def __init__(self):
        """
        Class constructor to initialize the dataset.
        """
        # Load the TED Talks translation dataset (Portuguese to English)
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)

        # Initialize BERT tokenizers for both Portuguese and English with a trimmed vocabulary
        self.tokenizer_pt = self.load_and_trim_tokenizer('neuralmind/bert-base-portuguese-cased', max_tokens=2**15)
        self.tokenizer_en = self.load_and_trim_tokenizer('bert-base-uncased', max_tokens=2**15)

    def load_and_trim_tokenizer(self, model_name, max_tokens):
        """
        Load a BERT tokenizer and trim its vocabulary to the specified max number of tokens.

        Args:
        - model_name: The pre-trained model name to load the tokenizer.
        - max_tokens: Maximum number of tokens to retain in the tokenizer vocabulary.

        Returns:
        - tokenizer: The customized BERT tokenizer with a reduced vocabulary size.
        """
        tokenizer = BertTokenizer.from_pretrained(model_name)

        # Limit the vocabulary to the first `max_tokens` entries
        if len(tokenizer.vocab) > max_tokens:
            reduced_vocab = {k: v for k, v in list(tokenizer.vocab.items())[:max_tokens]}
            tokenizer.vocab = reduced_vocab
            tokenizer.ids_to_tokens = {v: k for k, v in reduced_vocab.items()}

        return tokenizer

    # def tokenize_sentence(self, sentence, tokenizer):
    #     """
    #     Tokenizes a sentence using the given BERT tokenizer.

    #     Args:
    #     - sentence: A string containing the sentence to tokenize.
    #     - tokenizer: A BERT tokenizer (Portuguese or English).

    #     Returns:
    #     - tokenized_sentence: A tensor containing the tokenized sentence.
    #     """
    #     tokenized_sentence = tokenizer(sentence, return_tensors='tf')['input_ids']
    #     return tokenized_sentence
