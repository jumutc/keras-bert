from keras_bert import get_base_dict, get_model, gen_batch_inputs
from keras_bert.bert import TOKEN_CLS, TOKEN_SEP, TOKEN_MASK, TOKEN_UNK
from multiprocessing import Pool
from keras import backend as K
from psutil import cpu_count

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import keras
import nltk
import sys

seq_len = 128
n = cpu_count() // 2

tokenize = lambda e: nltk.word_tokenize(e.lower(), sys.argv[3])
count_words = lambda x: np.sum([len(s) for s in x])


def tokenize_partition(df):
    return df.apply(lambda e: [tokenize(s.strip()) for s in e.split('.')])


input_df = pd.read_csv(sys.argv[1], error_bad_lines=False)
input_df = input_df[~input_df.duplicated(subset=['expression'])]
input_df = input_df[input_df['expression'].str.len() > 0]
input_df['expression'] = input_df['expression'].apply(tokenize)
input_df = input_df[input_df['expression'].map(np.unique).map(len) > 2]
input_df = input_df[input_df['expression'].map(len) <= seq_len // 2]

expressions = input_df['expression'].values
print("Expressions shape: %s" % expressions.shape)

wiki_df = pd.read_csv(sys.argv[2], error_bad_lines=False, header=None)
wiki_df = wiki_df.loc[wiki_df[0].str.len() > 50, 0]

pool = Pool(processes=n)
list_dfs = [wiki_df[i:i + n].copy() for i in range(0, wiki_df.shape[0], n)]
wiki_dfs = pool.map(tokenize_partition, list_dfs)
pool.close()

wiki_df = pd.concat(wiki_dfs)
wiki_df = wiki_df[wiki_df.map(len) > 1]
wiki_df = wiki_df[wiki_df.map(count_words) <= seq_len - 3]

sentence_tuples = wiki_df.values
print("Wiki sentences shape: %s" % sentence_tuples.shape)

token_dict = get_base_dict()  # A dict that contains some special tokens
token_dict_freq = dict()
token_dict_rev = dict()

for k, v in token_dict.items():
    token_dict_rev[v] = k
    token_dict_freq[k] = 1000

# Build token dictionary
for sentence_tuple in sentence_tuples:
    for sentence in sentence_tuple:
        for token in sentence:
            if token not in token_dict:
                index = len(token_dict)
                token_dict[token] = index
                token_dict_freq[token] = 1
                token_dict_rev[index] = token
            else:
                token_dict_freq[token] += 1

token_keys = [k for k in token_dict.keys() if token_dict_freq[k] >= 10]
token_dict = {k: i for i, k in enumerate(token_keys)}
token_list = list(token_dict.keys())

with open('bert_vocab.plk', 'wb') as f:
    pickle.dump(token_dict, f)


def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_tuples,
            token_dict,
            token_list,
            mask_rate=0.15,
            seq_len=seq_len,
            swap_sentence_rate=1.0,
            batch_size=16
        )


def _get_session():
    tf_config = tf.ConfigProto(
        use_per_session_threads=True,
        allow_soft_placement=True
    )
    tf_config.gpu_options.allow_growth = True
    return tf.Session(graph=tf.get_default_graph(), config=tf_config)


K.set_session(_get_session())

# Build & train the model
model = get_model(
    token_num=len(token_dict),
    embed_dim=256,
    head_num=4,
    transformer_num=6,
    seq_len=seq_len
)
model.summary()

model.fit_generator(
    generator=_generator(),
    steps_per_epoch=4000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    ],
)

model.save_weights('bert_nlg.hdf5')
np.random.shuffle(expressions)

for input in expressions[:20]:
    tokens = [TOKEN_CLS] + input + [TOKEN_SEP] + [TOKEN_MASK] * (seq_len - len(input) - 3) + [TOKEN_SEP]

    token_input = np.asarray([[token_dict.get(token, TOKEN_UNK) for token in tokens]])
    seg_input = np.asarray([[0] * (len(input) + 2) + [1] * (seq_len - len(input) - 2)])
    mask_input = np.asarray([[0] * seq_len])

    output = model.predict([token_input, seg_input, mask_input])[0]
    indices = np.argmax(output, axis=-1)[0]
    probabilities = np.max(output, axis=-1)[0]
    prob_mask = np.argwhere(probabilities > 0.5)

    print("INPUT: %s" % input)
    print("OUTPUT: %s" % [token_dict_rev[o] for o in indices])
    print("CLEANED OUTPUT: %s" % [token_dict_rev[o] for o in indices[prob_mask].flatten()])
