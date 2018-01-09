import logging
import operator
import os

import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer

# create logger
logging.basicConfig(level = logging.WARNING)
logger = logging.getLogger(__name__)
# Set logger level
logger.setLevel(logging.INFO)

# Data Directory
data_dir = 'data/training-monolingual'

# Function to read the corpus file and return a text string
def read_corpus(fname):
    """
    :param fname(str): name of the file to be imported
    :return: a list of strings
    """
    f = open(os.path.join(data_dir, fname))
    txt = f.readlines()
    return txt

## Keras Tokenizer handle most of these functions already
# # Function to remove punctuations from unicode strings
# def remove_punctuation_utf8(text):
#     """
#     :param text(str): unicode string
#     :return: unicode string
#     """
#     return re.sub(re.compile(u'[^\u4E00-\u9FA5]'), r"", text)
#
# # Function to remove all special characters from a list of strings
# def process_txt(txt, lang):
#     # Initialize the list to store cleaned txt
#     cleaned_txt = []
#     # Translator to remove all punctuation from strings
#     translator = str.maketrans(dict.fromkeys(string.punctuation))
#
#     # Loop through all the lines in the txt
#     for i, line in enumerate(txt):
#         if lang in ('zhs', 'zht'):
#             cleaned_line = remove_punctuation_utf8(line.decode('utf-8'))
#             cleaned_txt.append(cleaned_line)
#
#         elif lang in ['en', 'fr']:
#             cleaned_txt.append(line.translate(translator).lower())
#
#         if i % 10000 == 0:
#             logger.info("Processed {}/{}".format(i, len(txt)))
#
#     return cleaned_txt

def load_vectors(fname, binary):
    model = KeyedVectors.load_word2vec_format(fname, binary = binary)
    logger.info('Imported {} words pretrained vector'.format(len(model.vocab.keys())))

    return model

# Function to convert the FastText pretrained vector to a binary format for gensim
def convert_vectors(path, fname, n_words, cvt_fname):
    name, lang, fmt = fname.split('.')

    logger.info('Importing {} word vectors'.format(fname))
    model = KeyedVectors.load_word2vec_format(os.path.join(path, fname), limit = n_words)

    new_fname = ".".join([cvt_fname, lang, 'bin'])

    logger.info('Saving the word vector in binary format')
    model.save_word2vec_format(os.path.join(path, new_fname), binary = True)



def pretrained_embeddings(fname, word2vec, nb_words, embedding_dim):
    """
    Extract pretrained word2vec according to most frequently used words in a corpus
    :param fname: name of the corpus
    :param word2vec:  word2vec model from gensim
    :param nb_words: number of words in the final embedding
    :param embedding_dim: dimension of the embedding
    :return:
        pretrained word2vec of top nb_words, word2idx dictionary
    """
    txt = read_corpus(fname)

    # Create a keras tokenizer
    tokenizer = Tokenizer()
    # Fit tokenizer on text
    tokenizer.fit_on_texts(txt)

    # Extract the words count dictionary from the tokenizer
    words_dict = tokenizer.word_counts
    logger.info('Number of words in the corpus: {}'.format(len(words_dict)))

    # Check which words from the corpus exist in the word2vec model
    filtered_words_dict = {word: count for word, count in words_dict.items() if word2vec.vocab.get(word) is not None}
    logger.info('Number of corpus words in the word2vec model: {}'.format(len(filtered_words_dict)))

    # Sort and keep the top-n words common to corpus and word2vec
    sorted_words = sorted(filtered_words_dict.items(), key=operator.itemgetter(1), reverse=True)[:nb_words]

    # Extract the relevant word vector into a matrix
    embedding = np.zeros((nb_words, embedding_dim))
    # word2idx for the extracted embeddings
    word2idx = dict()
    for i, tup in enumerate(sorted_words):
        word2idx[tup[0]] = i
        embedding[i] = word2vec.word_vec(tup[0])

    return embedding, word2idx
