
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sentences = [
             'Hey! how are you doing?',
             'I am doing great! What about you',
             'Well! I had some stuff to do but now I am free, thanks for asking'
]
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)        


word = tokenizer.word_index

sentence_sequences = tokenizer.texts_to_sequences(sentences)



test_sequence = [
                 'Hey, everyone how are you doing ?',
                 'Everyone: We are good, lets just party',
]
test_data_sequences = tokenizer.texts_to_sequences(test_sequence)
pad_seq = pad_sequences(sequences)
pad_test_seq = pad_sequences(test_data_sequences)

print(word)
print(sentence_sequences)
print(f'Test sequences: \n {test_data_sequences}')
print(f'Padded sequences for sentences:\n {pad_seq}')
print(f'Padded sequecnes for test data:\n {pad_test_seq}')



