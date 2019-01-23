from keras_bert import get_base_dict, get_model, gen_batch_inputs_nlg
from keras_bert.bert import TOKEN_CLS, TOKEN_SEP, TOKEN_MASK

import pandas as pd
import numpy as np
import keras
import nltk
import sys

seq_len = 126
tokenize = lambda e: nltk.word_tokenize(e.lower(), sys.argv[2])
input_df = pd.read_csv(sys.argv[1], error_bad_lines=False)
input_df = input_df[~input_df.duplicated(subset=['expression'])]
input_df = input_df[input_df['expression'].str.len() > 0]
input_df['expression'] = input_df['expression'].apply(tokenize)
input_df = input_df[input_df['expression'].map(np.unique).map(len) > 2]
input_df = input_df[input_df['expression'].map(len) <= seq_len // 2]

intents = input_df['intent'].values
sentences = input_df['expression'].values
print("Dataset shape: %s" % sentences.shape)

token_dict = get_base_dict()  # A dict that contains some special tokens
token_dict_rev = dict()

for k, v in token_dict.items():
    token_dict_rev[v] = k

# Build token dictionary
for sentence in sentences:
    for token in sentence:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
            token_dict_rev[len(token_dict)] = token

token_list = list(token_dict.keys())  # Used for selecting a random word

# Build & train the model
model = get_model(
    token_num=len(token_dict),
    embed_dim=256,
    head_num=4,
    transformer_num=6,
    feed_forward_dim=256,
    seq_len=seq_len
)
model.summary()


def _generator():
    while True:
        yield gen_batch_inputs_nlg(
            sentences,
            intents,
            token_dict,
            token_list,
            mask_rate=0.1,
            seq_len=seq_len,
            swap_sentence_rate=1.0,
        )


model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    ],
)

model.save_weights('bert_nlg.hdf5')
np.random.shuffle(sentences)

for input in sentences[:20]:
    tokens = [TOKEN_CLS] + input + [TOKEN_SEP] + [TOKEN_MASK] * (seq_len - len(input) - 2)

    token_input = np.asarray([[token_dict[token] for token in tokens]])
    seg_input = np.asarray([[0] * (len(input) + 2) + [1] * (seq_len - len(input) - 2)])
    mask_input = np.asarray([[0] * seq_len])

    output = model.predict([token_input, seg_input, mask_input])[0]
    indices = np.argmax(output, axis=-1)[0]
    probabilities = np.max(output, axis=-1)[0]
    prob_mask = np.argwhere(probabilities > 0.5)

    print("INPUT: %s" % [token_dict_rev[i] for i in input])
    print("OUTPUT: %s" % [token_dict_rev[o] for o in output[0]])
    print("CLEANED OUTPUT: %s" % [token_dict_rev[o] for o in output[0][prob_mask]])
