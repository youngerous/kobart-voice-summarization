# ref. (code) https://deeplearningcourses.com/c/deep-learning-advanced-nlp
# ref. (data) https://github.com/muik

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import os, sys
import hgtk
import pickle

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
                         Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt

#import tensorflow as tf
import tensorflow.compat.v1 as tf

# DIFF(subin)
if len(tf.config.list_physical_devices('GPU') ) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU

from datetime import datetime
import time

import warnings
warnings.filterwarnings('ignore')

###############################################################################

YEARMONTHDAY = str(datetime.fromtimestamp(time.time())).split()[0]
CUR_PATH = os.path.dirname(os.path.abspath( __file__ ))
#CUR_PATH = os.getcwd()
BACK_PATH = '/'.join(CUR_PATH.split('/')[:-3]) # back 2 times
PRETRAINED_MODEL_PATH = CUR_PATH + "/resources/2019-05-14_model.h5"
DEFAULT_COMPOSE_CODE = 'ᴥ'

params = {
    'BATCH_SIZE' : 32, #64
    'EPOCHS' : 100, #100,
    'LATENT_DIM' : 300, #256
    'LATENT_DIM_DECODER' : 300, #256 # idea: make it different to ensure things all fit together properly!
    'EMBEDDING_DIM' : 150, #100
    #'MAX_SEQUENCE_LENGTH' = 100,
    'MAX_NUM_WORDS' : None, #20000

    # below values are saved after training process
    # for using pretrained model
    'LEN_INPUT_TEXTS' : None, # 56699, # = NUM_SAMPLES
    'MAX_LEN_INPUT' : None, # 52,
    'MAX_LEN_TARGET' : None, # 47,
    'LEN_WORD2IDX_INPUTS' : None, # 33,
    'LEN_WORD2IDX_OUTPUTS' : None, # 43,
}

DEBUG_MODE = True
###############################################################################

def log(*s): # multiple args
    if DEBUG_MODE:
        print(s)

def save_pkl(data, name):
    fp = open(name, 'wb')
    pickle.dump(data, fp)
    fp.close()

def load_pkl(name):
    fp = open(name, 'rb')
    return pickle.load(fp)

def load_data(path_trans): # dataset only for this projecte (specifc form)
    #log('> Loading')
    data = []
    for filename in os.listdir(path_trans):
        full_path = path_trans + '/' + filename
        each_file = open(full_path, 'r')
        for x in each_file:
            if '#' == list(x)[0]:
                continue
            data.append(x.strip())
    return data

def eng_preprop(in_str):
    in_str = in_str.lower()
    in_str = in_str.replace(' ', '_')
    in_str = in_str.replace('-', '_')
    return in_str

def preprocessing(data):
    #log('> Preprocessing')
    def kor_preprop(in_str):
        in_str = in_str.replace(' ', '')
        in_str_decompose = hgtk.text.decompose(in_str)
        in_str_filter = [x for x in list(in_str_decompose) if x != DEFAULT_COMPOSE_CODE]
        in_str_join = ''.join(in_str_filter)
        return in_str_join
    for i, _ in enumerate(data):
        source_eng = data[i].split('\t')[0]
        target_kor = data[i].split('\t')[-1]
        data[i] = eng_preprop(source_eng) + '\t' + kor_preprop(target_kor)
    return data

def input_formatting(data):
    #log('> Input Formatting')
    input_texts = [] # sentence in original language
    target_texts = [] # sentence in target language
    target_texts_inputs = [] # sentence in target language offset by 1
    """
    < korean-go.txt >
    ... ... ...
    gahnite     가나이트
    garnetting  가네팅
    GANEFO      가네포
    garnett     가넷
    ... ... ...
    """
    #t = 0
    #for line in open(os.getcwd() + '/spa.txt'):
    for line in data:
        # only keep a limited number of samples
        #t += 1
        #if t > NUM_SAMPLES:
        #    break
        # input and target are separated by tab
        if '\t' not in line:
            continue
        # split up the input and translation
        input_text, translation = line.rstrip().split('\t')

        # make the target input and output
        # recall we'll be using teacher forcing
        target_text = ' '.join(list(translation)) + ' <eos>'
        target_text_input = '<sos> ' + ' '.join(list(translation))

        input_texts.append(' '.join(list(input_text)))
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)

    # log(">> Number of Data:", len(input_texts))
    params['LEN_INPUT_TEXTS'] = len(input_texts)
    return (input_texts, target_texts_inputs, target_texts)

def tokenizing(input_texts, target_texts_inputs, target_texts):
    #log('> Tokenizing')
    ## tokenize the inputs
    #tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer_inputs = Tokenizer(num_words=params['MAX_NUM_WORDS'], filters='') # MAX_NUM_WORDS = None
    tokenizer_inputs.fit_on_texts(input_texts)
    input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
    # get the word to index mapping for input language
    word2idx_inputs = tokenizer_inputs.word_index
    params['LEN_WORD2IDX_INPUTS'] = len(word2idx_inputs)
    #print('Found %s unique input tokens.' % len(word2idx_inputs))
    # determine maximum length input sequence
    params['MAX_LEN_INPUT'] = max(len(s) for s in input_sequences)
    # save 'tokenizer_inputs' for decoding
    save_pkl(tokenizer_inputs, CUR_PATH + '/resources/tokenizer_inputs.pkl')
    #log('>> Tokenizer_inputs is saved!')

    ## tokenize the outputs
    # tokenize the outputs
    # don't filter out special characters
    # otherwise <sos> and <eos> won't appear
    tokenizer_outputs = Tokenizer(num_words=params['MAX_NUM_WORDS'], filters='') # MAX_NUM_WORDS = None
    tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs) # inefficient, oh well
    target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
    target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
    # get the word to index mapping for output language
    word2idx_outputs = tokenizer_outputs.word_index
    params['LEN_WORD2IDX_OUTPUTS'] = len(word2idx_outputs)
    #print('Found %s unique output tokens.' % len(word2idx_outputs))
    # store number of output words for later
    # remember to add 1 since indexing starts at 1 (index 0 = unknown)
    #num_words_output = len(word2idx_outputs) + 1
    # determine maximum length output sequence
    params['MAX_LEN_TARGET'] = max(len(s) for s in target_sequences)
    # save 'tokenizer_inputs' for decoding
    save_pkl(tokenizer_outputs, CUR_PATH + '/resources/tokenizer_outputs.pkl')
    #log('>> Tokenizer_outputs is saved!')

    return (input_sequences, target_sequences_inputs, target_sequences, word2idx_inputs, word2idx_outputs)


def padding(input_sequences, target_sequences_inputs, target_sequences):
    #log('> Padding')
    # pad the sequences
    encoder_inputs = pad_sequences(input_sequences, maxlen=params['MAX_LEN_INPUT'])
    #log(">> encoder_data.shape:", encoder_inputs.shape)
    #print("encoder_data[0]:", encoder_inputs[0])

    decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=params['MAX_LEN_TARGET'], padding='post')
    #print("decoder_data[0]:", decoder_inputs[0])
    #log(">> decoder_data.shape:", decoder_inputs.shape)

    decoder_targets = pad_sequences(target_sequences, maxlen=params['MAX_LEN_TARGET'], padding='post')

    return (encoder_inputs, decoder_inputs, decoder_targets)


# DIFF(subin)
def use_pretrained_word2vec(glove_fullpath, word2idx_inputs):

    # store all the pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    #with open(os.path.join(os.getcwd()+'/glove.6B.%sd.txt' % params['EMBEDDING_DIM'])) as f:
    with open(os.path.join(glove_fullpath)) as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    print('Found %s word vectors.' % len(word2vec))

    # prepare embedding matrix
    print('Filling pre-trained embeddings...')
    if params['MAX_NUM_WORDS'] == None:
        raise ValueError('please set MAX_NUM_WORDS params. not None.')
    num_words = min(params['MAX_NUM_WORDS'], len(word2idx_inputs) + 1)
    embedding_matrix = np.zeros((num_words, params['EMBEDDING_DIM']))
    for word, i in word2idx_inputs.items():
        if i < params['MAX_NUM_WORDS']:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

class Transliterator(object):

    def __init__(self):
        ## Basic process for model
        # 아래 과정을 통해 입출력 길이를 파악해야 해야만, 네트워크 파라미터 크기를 결정할 수 있음. (필수적)

        # DIFF(subin)
        log('Loading model...')
        data = load_data(CUR_PATH + '/data') # dataset only for transliteration
        data = preprocessing(data)
        input_texts, target_texts_inputs, target_texts = input_formatting(data)
        input_sequences, target_sequences_inputs, target_sequences, word2idx_inputs, word2idx_outputs = tokenizing(input_texts, target_texts_inputs, target_texts)
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = padding(input_sequences, target_sequences_inputs, target_sequences)

        ## Variables
        self.tokenizer_inputs = load_pkl(CUR_PATH + '/resources/tokenizer_inputs.pkl')
        self.tokenizer_outputs = load_pkl(CUR_PATH + '/resources/tokenizer_outputs.pkl')
        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def _softmax_over_time(self, x):
        # make sure we do softmax over the time axis
        # expected shape is N x T x D
        assert(K.ndim(x) > 2)
        e = K.exp(x - K.max(x, axis=1, keepdims=True)) # axis=1에 주목.
        s = K.sum(e, axis=1, keepdims=True)
        return e / s

    def _stack_and_transpose(self, x): # 다시 원래의 shape로 만들기 위해.
        # 'outputs' is now a list of length Ty
        # each element is of shape (batch size, output vocab size)
        # therefore if we simply stack all the outputs into 1 tensor
        # it would be of shape T x N x D
        # we would like it to be of shape N x T x D
        # x is a list of length T, each element is a batch_size x output_vocab_size tensor
        x = K.stack(x) # is now T x batch_size x output_vocab_size tensor
        x = K.permute_dimensions(x, pattern=(1, 0, 2)) # is now batch_size x T x output_vocab_size
        return x

    # DIFF(subin): remove def train(self), not used
    
    def use_pretrained_model(self):
    # For using pre-trained model
    # duplicate version of train function except for model.fit

        # DIFF(subin)
        log('Loading pretrained_weights...')
        
        ###########
        ### ENCODER
        embedding_layer = Embedding(
                                params['LEN_WORD2IDX_INPUTS'] + 1,
                                params['EMBEDDING_DIM'],
                                #weights=[embedding_matrix],
                                input_length=params['MAX_LEN_INPUT'],
                                # trainable=True
                          )
        encoder_inputs_placeholder = Input(shape=(params['MAX_LEN_INPUT'],))
        x = embedding_layer(encoder_inputs_placeholder)
        encoder = Bidirectional(LSTM(
                                    params['LATENT_DIM'],
                                    return_sequences=True,
                                    # dropout=0.5 # dropout not available on gpu
                                )
                  )
        encoder_outputs = encoder(x)

        ###########
        ### DECODER
        # this word embedding will not use pre-trained vectors, although you could
        decoder_embedding = Embedding(
                                params['LEN_WORD2IDX_OUTPUTS'] + 1,
                                params['EMBEDDING_DIM']
                            )
        decoder_inputs_placeholder = Input(shape=(params['MAX_LEN_TARGET'],)) # teacher forcing input
        decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

        #############
        ### ATTENTION
        # Attention layers need to be global (전역 변수) because they will be repeated Ty times at the decoder
        attn_repeat_layer = RepeatVector(params['MAX_LEN_INPUT'])
        attn_concat_layer = Concatenate(axis=-1)
        attn_dense1 = Dense(10, activation='tanh')
        attn_dense2 = Dense(1, activation=self._softmax_over_time)
        attn_dot = Dot(axes=1) # to perform the weighted sum of alpha[t] * h[t]
        
        def _one_step_attention(h, st_1):
            # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
            # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

            # copy s(t-1) Tx times
            # now shape = (Tx, LATENT_DIM_DECODER)
            st_1 = attn_repeat_layer(st_1)

            # Concatenate all h(t)'s with s(t-1)
            # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM * 2)
            x = attn_concat_layer([h, st_1])

            # Neural net first layer
            x = attn_dense1(x)

            # Neural net second layer with special softmax over time
            alphas = attn_dense2(x)

            # "Dot" the alphas and the h's
            # Remember a.dot(b) = sum over a[t] * b[t]
            context = attn_dot([alphas, h])

            return context

        # define the rest of the decoder (after attention)
        decoder_lstm = LSTM(
                            params['LATENT_DIM_DECODER'],
                            return_state=True
                        )
        decoder_dense = Dense(
                            params['LEN_WORD2IDX_OUTPUTS'] + 1,
                            activation='softmax'
                        )

        initial_s = Input(shape=(params['LATENT_DIM_DECODER'],), name='s0')
        initial_c = Input(shape=(params['LATENT_DIM_DECODER'],), name='c0')
        context_last_word_concat_layer = Concatenate(axis=2) # for teacher forcing

        # Unlike previous seq2seq, we cannot get the output all in one step
        # Instead we need to do Ty steps And in each of those steps, we need to consider all Tx h's

        # s, c will be re-assigned in each iteration of the loop
        s = initial_s
        c = initial_c

        # collect outputs in a list at first
        outputs = []
        # 원래 LSTM은 내부적으로 아래와 같은 for문을 진행하지만, 여기서 우리는 Context를 계산하기 위해서 manual하게 for문을 구성함.
        for t in range(params['MAX_LEN_TARGET']): # Ty times

            ######################################################
            ## `one_step_attention` function !
            # get the context using attention
            context = _one_step_attention(encoder_outputs, s)

            # we need a different layer for each time step
            selector = Lambda(lambda x: x[:, t:t+1]) # 해당 time 벡터만 추출. 우리는 layer-wise로 코딩해야 되기 때문에 lambda를 사용.
            xt = selector(decoder_inputs_x)

            # combine
            decoder_lstm_input = context_last_word_concat_layer([context, xt])

            # pass the combined [context, last word] into the LSTM
            # along with [s, c]
            # get the new [s, c] and output
            o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

            # final dense layer to get next word prediction
            decoder_outputs = decoder_dense(o)
            outputs.append(decoder_outputs)

        # make it a layer
        stacker = Lambda(self._stack_and_transpose)
        outputs = stacker(outputs)

        ################
        ### CREATE MODEL
        self.model = Model(
            inputs=[
                encoder_inputs_placeholder,
                decoder_inputs_placeholder,
                initial_s,
                initial_c,
            ],
            outputs=outputs
        )

        # compile the model
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        ## load pretrained model
        self.model.load_weights(PRETRAINED_MODEL_PATH)

        ##### Make predictions #####
        # As with the poetry example, we need to create another model
        # that can take in the RNN state and previous word as input
        # and accept a T=1 sequence.
        #encoder_inputs_placeholder = Input(shape=(params['MAX_LEN_INPUT'],))

        # The encoder will be stand-alone
        # From this we will get our initial decoder hidden state
        # i.e. h(1), ..., h(Tx)
        self.encoder_model = Model(
                    encoder_inputs_placeholder,
                    encoder_outputs)

        # next we define a T=1 decoder model
        encoder_outputs_as_input = Input(shape=(params['MAX_LEN_INPUT'], params['LATENT_DIM'] * 2,))
        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

        # no need to loop over attention steps this time because there is only one step
        context = _one_step_attention(encoder_outputs_as_input, initial_s)

        # combine context with last word
        decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])

        # lstm and final dense
        o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
        decoder_outputs = decoder_dense(o)

        # note: we don't really need the final stack and tranpose
        # because there's only 1 output
        # it is already of size N x D
        # no need to make it 1 x N x D --> N x 1 x D
        # time dimension이 1이기 때문에 자동으로 없어짐: 따라서, stack_and_transpose함수가 필요없음.

        # create the model object
        self.decoder_model = Model(
            inputs=[
                decoder_inputs_single,
                encoder_outputs_as_input,
                initial_s,
                initial_c
            ],
            outputs=[decoder_outputs, s, c]
        )

        # map indexes back into real words
        # so we can view the results
        #idx2word_eng = {v:k for k, v in word2idx_inputs.items()}
        #idx2word_trans = {v:k for k, v in word2idx_outputs.items()}

        # DIFF(subin)
        log('Successfully loaded...')
        
    def compose_hangul(self, in_str):
        # https://zetawiki.com/wiki/...
        kor_vowel_list = "ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ".split()
        temp_list = [DEFAULT_COMPOSE_CODE]
        temp_input_list = in_str[::-1].split()
        for i, x in enumerate(temp_input_list):
            #print(i, x)
            if i >= 2:
                if temp_input_list[i-2] in kor_vowel_list:
                    temp_list.append(DEFAULT_COMPOSE_CODE)
                temp_list.append(temp_input_list[i])
            else:
                temp_list.append(temp_input_list[i])
        #print(temp_list)
        out_str = hgtk.text.compose(temp_list[::-1])
        return out_str

    def decode_sequence(self, input_seq):
        # preprocessing & tokenizing & padding for input_seq
        input_seq = eng_preprop(input_seq)
        input_seq = ' '.join(list(input_seq))
        input_seq = self.tokenizer_inputs.texts_to_sequences([input_seq]) # it is array!
        input_seq = pad_sequences(input_seq, maxlen=params['MAX_LEN_INPUT'])

        # Encode the input as state vectors.
        enc_out = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first character of target sequence with the start character.
        # NOTE: tokenizer lower-cases all words
        target_seq[0, 0] = self.tokenizer_outputs.word_index['<sos>'] # word2idx_outputs

        # if we get this we break
        eos = self.tokenizer_outputs.word_index['<eos>'] # word2idx_outputs

        # [s, c] will be updated in each loop iteration
        s = np.zeros((1, params['LATENT_DIM_DECODER']))
        c = np.zeros((1, params['LATENT_DIM_DECODER']))

        # Create the translation
        output_sentence = []
        output_prob_dist = []
        for _ in range(params['MAX_LEN_TARGET']):
            o, s, c = self.decoder_model.predict([target_seq, enc_out, s, c])

            output_prob_dist.append(max(o.flatten()))

            # Get next word
            idx = np.argmax(o.flatten())

            # End sentence of EOS
            if eos == idx:
                break

            word = ''
            if idx > 0:
                word = {v:k for k, v in self.tokenizer_outputs.word_index.items()}[idx] # idx2word_trans
                output_sentence.append(word)

            # Update the decoder input
            # which is just the word just generated
            target_seq[0, 0] = idx

        return (self.compose_hangul(' '.join(output_sentence)), np.average(output_prob_dist))

##########################################################################################
# 기본적으로 이 프로그램은 import를 통해서 실행.
if __name__ == "__main__":
    print('Please use this via import process!')
    pass

else: # when being used for import!
    pass
    #transliterator = TransliterateEngKor()
    #encoder_model, decoder_model = transliterator.full_process_using_pretrained_model()

    #model = transliterator.desgin_model_for_train()
    #transliterator.load_pretrained_model(model, "_saved/model.h5")
    #encoder_model, decoder_model = transliterator.design_model_for_predict()

    #data = load_data(BACK_PATH + '/assets/transliteration') # dataset only for transliteration
    #data = preprocessing(data)
    #input_texts, target_texts_inputs, target_texts = input_formatting(data)
    #input_sequences, target_sequences_inputs, target_sequences, word2idx_inputs, word2idx_outputs = tokenizing(input_texts, target_texts_inputs, target_texts)
    #encoder_inputs, decoder_inputs, decoder_targets = padding(input_sequences, target_sequences_inputs, target_sequences)

    #print('>> Basic process is done!')

    #transliterator = TransliterateEngKor()
    #model = transliterator.desgin_model_for_train()
    #transliterator.train(model, encoder_inputs, decoder_inputs, decoder_targets) # training
