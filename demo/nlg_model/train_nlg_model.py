from keras_bert import get_base_dict, get_model, gen_batch_inputs_nlg

import pandas as pd
import numpy as np
import keras
import nltk
import sys

seq_len=126
tokenize = lambda e: nltk.word_tokenize(e.lower(), sys.argv[2])
input_df = pd.read_csv(sys.argv[1], error_bad_lines=False)
input_df = input_df[~input_df.duplicated(subset=['expression'])]
input_df = input_df[input_df['expression'].str.len() > 0]
input_df['expression'] = input_df['expression'].apply(tokenize)
input_df = input_df[input_df['expression'].map(np.unique).map(len) > 2]
input_df = input_df[input_df['expression'].map(len) <= seq_len//2]

sentences = input_df['expression'].values
print("Dataset shape: %s" % sentences.shape)

token_dict = get_base_dict()  # A dict that contains some special tokens
# Build token dictionary
for sentence in sentences:
    for token in sentence:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word

# Build & train the model
model = get_model(
    token_num=len(token_dict),
    embed_dim=256,
    head_num=4,
    feed_forward_dim=1024,
    seq_len=seq_len,

)
model.summary()


def _generator():
    while True:
        yield gen_batch_inputs_nlg(
            sentences,
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

test_data = gen_batch_inputs_nlg(
            sentences,
            token_dict,
            token_list,
            mask_rate=0.1,
            swap_sentence_rate=1.0,
        )

inputs = test_data[0]
outputs = model.predict(inputs)

for input, output in zip(inputs, outputs):
    print("INPUT: %s -- OUTPUT: %s" % ([token_dict[i] for i in input], [token_dict[o] for o in output]))

