from keras.layers import LSTM,Input,Dense,Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

max_vocab_size=20000
embedding_dim=50
max_sequence_length=100
epochs=2000  
batch_size=128
latent_dim=25

input_texts=[]
output_texts=[]
with open("robert_frost.txt") as f:
    for line in f:
        line=line.rstrip()
        if not line:
            continue
        
        input_text="<sos> "+line
        output_text=line+" <eos>"
        input_texts.append(input_text)
        output_texts.append(output_text)
all_lines=input_texts+output_texts

tokenizer=Tokenizer(num_words=max_vocab_size,filters='')
tokenizer.fit_on_texts(all_lines)
input_sequences=tokenizer.texts_to_sequences(input_texts)
output_sequences=tokenizer.texts_to_sequences(output_texts)

max_length=max(len(s) for s in input_sequences)
max_sequence_length=min(max_length,max_sequence_length)

input_sequences=pad_sequences(input_sequences,maxlen=max_sequence_length)
output_sequences=pad_sequences(output_sequences,maxlen=max_sequence_length)

word2idx=tokenizer.word_index

idx2word={v:k for k,v in word2idx.items()}

max_words=min(len(word2idx)+1,max_vocab_size)


targets=np.zeros((len(input_sequences),max_sequence_length,max_words))
for i,output in enumerate(output_sequences):
    for t,o in enumerate(output):
        if o>0:
            targets[i,t,o]=1

embedding_matrix=np.zeros((max_words,embedding_dim))
word_vec={}
with open("glove.6B.50d.txt",encoding="utf-8") as f:
    for line in f:
        line=line.split()
        w=line[0]
        v=np.asarray(line[1:],np.float32)
        word_vec[w]=v
for words,i in word2idx.items():
    if i<max_words:
        vector=word_vec.get(words)
        if vector is not None:
            embedding_matrix[i]=vector
embedding_layer=Embedding(max_words,embedding_dim,weights=[embedding_matrix],trainable=False)

input_=Input(shape=(max_sequence_length,))
initial_h=Input(shape=(latent_dim,))
initial_c=Input(shape=(latent_dim,))
x=embedding_layer(input_)
lstm=LSTM(latent_dim,return_sequences=True,return_state=True)
x,_,_=lstm(x,initial_state=[initial_h,initial_c])
dense=Dense(max_words,activation='softmax')
output=dense(x)
model=Model([input_,initial_h,initial_c],output)
model.compile(loss="categorical_crossentropy",metrics=["accuracy"],optimizer=Adam(lr=0.1))
z=np.zeros((len(input_sequences),latent_dim))
model.fit([input_sequences,z,z],targets,validation_split=0.1,epochs=epochs,batch_size=batch_size)

input2=Input(shape=(1,))
x=embedding_layer(input2)
x,h,c=lstm(x,initial_state=[initial_h,initial_c])
output=dense(x)

sample_model=Model([input2,initial_h,initial_c],[output,h,c])

for i in range(4):
    test=np.array([[1]])
    
    h=np.zeros((1,latent_dim))
    c=np.zeros((1,latent_dim))
    K=" "

    for i in range(max_sequence_length):
        o,h,c=sample_model.predict([test,h,c])
        
        probs=o[0,0]
        probs[0]=0
        
        n=np.random.choice(len(probs), p=probs)
        if n==2:
            break
        K+=" "+idx2word.get(n)
        test[0,0]=n
    print(K)
    
    