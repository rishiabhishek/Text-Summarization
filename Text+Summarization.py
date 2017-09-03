
# coding: utf-8

# In[1]:


# Imports 
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
import re
from collections import Counter
import operator
from tensorflow.python.layers.core import Dense


# In[2]:


def read_reviews():
    reviews = pd.read_csv("../Datasets/Reviews/Reviews.csv")
    reviews = reviews.dropna()
    reviews = reviews.drop(["Id","ProductId","UserId","ProfileName","HelpfulnessNumerator","HelpfulnessDenominator","Score","Time"]
                 ,axis=1)
    return reviews

reviews = read_reviews()
reviews.head()


# In[3]:


reviews[reviews.isnull().any(axis=1)] # All cells have values


# In[4]:


# Cleaning and Normalizing the text and summaries
# Some contraction to expansion
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}
def normalization(review,remove_stopwords=False):
    text = review.lower()
    clean_text = []
    for word in text.split():
        if word in contractions:
            clean_text.append(contractions[word])
        else:
            clean_text.append(word)
    text = " ".join(clean_text)
    
    # Format words and remove unwanted characters
#     text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'https', ' ', text)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br', ' ', text)
    text = re.sub(r'/>', ' ', text)
    text = re.sub(r'>', ' ', text)
    text = re.sub(r'<', ' ', text)
    text = re.sub(r'`', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


# In[5]:


normalization(reviews.Text[713])


# In[6]:


def clean_reviews(texts):
    return [normalization(text) for text in texts]


# In[7]:


summary = clean_reviews(reviews.Summary)
text = clean_reviews(reviews.Text)


# In[8]:


print("None count in Summary ",sum(x is None for x in summary))
print("None count in Text ",sum(x is None for x in text))
print(len(summary),len(text))


# In[9]:


#Counting the words in Text and summary and remove words having count less than threshold
def get_word_count(texts,summaries,threshold=20):
    '''
    Params: Tests , Summaries ,threshold = 20
    Return : word count dict
    '''
    tokens = []
    for text in texts:
        tokens.extend(text.split())
    for summary in summaries:
        tokens.extend(summary.split())
    counts = Counter(tokens)
    reduced_count = {word:i for word,i in counts.items() if i >= threshold}
    return reduced_count


# In[10]:


count = get_word_count(text,summary)


# In[11]:


count


# In[12]:


def get_vocab(word_counts):
    '''
    Param: word_counts
    Return: Vocab,vocab_to_int,int_to_vocab
    '''
    vocab = set(word_counts.keys())
    
    vocab_to_int = {}
    int_to_vocab = {}
    
    codes = ["<UNK>","<PAD>","<EOS>","<GO>"]
    for i,code in enumerate(codes):
        vocab_to_int[code] = i

    for i,word in enumerate(vocab,4):
        vocab_to_int[word] = i
        
    int_to_vocab = {i:word for word,i in vocab_to_int.items()}
    return vocab,vocab_to_int,int_to_vocab


# In[13]:


vocab,vocab_to_int,int_to_vocab = get_vocab(count)


# In[14]:


print(len(vocab),len(vocab_to_int),len(int_to_vocab))


# In[15]:


# Using pre-trained Conceptnet Numberbatch's Embeddings (https://github.com/commonsense/conceptnet-numberbatch)
def get_word_embeddings():
    embeddings = {}
    with open('../Datasets/embeddings/numberbatch-en-17.06.txt',encoding='utf-8') as em:
        for embed in em:
            em_line = embed.split(' ')
            if len(em_line) > 2: # First line of file is no. of words , number of dimensions
                word = em_line[0]
                embedding = np.array(em_line[1:])
                embeddings[word] = embedding
    print('Word embeddings:', len(embeddings))
    return embeddings


# In[16]:


CN_embeddings = get_word_embeddings()


# In[17]:


not_in_embeddings = [word for word in vocab if word not in CN_embeddings]


# In[18]:


print("No. of words not in Ebeddings : ",len(not_in_embeddings))


# In[19]:


def create_embedding_matrix(int_to_vocab,embeddings,embedding_dim = 300):
    '''
    Params : int_to_vocab, embeddings, embedding_dim
    Return : embedding matrix
    '''
    # Generating empty numpy matrix
    embeding_matrix = np.zeros([len(vocab_to_int),embedding_dim])
    embeding_matrix = embeding_matrix.astype(np.float32)
    
    #Generating random embeddings for words not in CN embeddings
    for i,word in int_to_vocab.items():
        if word in embeddings:
            embeding_matrix[i] = embeddings[word]
        else:
            embeding_matrix[i] = np.array(np.random.normal(embedding_dim))
    return embeding_matrix


# In[20]:


embeding_matrix = create_embedding_matrix(int_to_vocab,CN_embeddings)


# In[21]:


print(len(embeding_matrix),len(vocab_to_int))


# In[22]:


def encode_source_target(sources, targets, vocab_to_int):
    '''
    Params : Sources, Targets, vocab_to_int
    Return :encoded_sources, encoded_targets
    '''
    encoded_sources = []
    encoded_targets = []
    for source in sources:
        encod_ent = []
        for word in source.split():
            if word in vocab_to_int:
                encod_ent.append(vocab_to_int[word])
            else:
                encod_ent.append(vocab_to_int["<UNK>"])
        encoded_sources.append(encod_ent)
    
    for target in targets:
        encod_ent = []
        for word in target.split():
            if word in vocab_to_int:
                encod_ent.append(vocab_to_int[word])
            else:
                encod_ent.append(vocab_to_int["<UNK>"])
        encoded_targets.append(encod_ent)
        
    return encoded_sources, encoded_targets


# In[23]:


encoded_sources, encoded_targets = encode_source_target(text,summary,vocab_to_int)


# In[24]:


print(len(encoded_sources),len(text))


# ### Model

# In[25]:


# Building Input Placeholders
def model_inputs():
    '''
    Returns : input_,target,learning_rate,keep_prob,source_seq_length,target_seq_length,max_target_seq_length
    '''
    input_ = tf.placeholder(dtype=tf.int32,shape=(None,None),name="inputs")
    target = tf.placeholder(dtype=tf.int32,shape=(None,None),name="target")
    
    learning_rate = tf.placeholder(dtype=tf.float32,name="learning_rate")
    keep_prob = tf.placeholder(dtype=tf.float32,name="keep_prob")
    
    source_seq_length = tf.placeholder(dtype=tf.int32,shape=(None,),name="source_seq_length")
    target_seq_length = tf.placeholder(dtype=tf.int32,shape=(None,),name="target_seq_length")
    
    max_target_seq_length = tf.reduce_max(target_seq_length,name="max_target_seq_length")
    return input_,target,learning_rate,keep_prob,source_seq_length,target_seq_length,max_target_seq_length


# In[26]:


#Process decoder input
def process_decoder_input(target_data,vocab_to_int,batch_size):
    
    strided_target = tf.strided_slice(target_data,(0,0),(batch_size,-1),(1,1))
    go = tf.fill(value=vocab_to_int["<GO>"],dims=(batch_size,1))
    decoder_input = tf.concat((go,strided_target),axis=1)
    return decoder_input


# In[27]:


# Create LSTM cells
def get_lstm(rnn_size,keep_prob=0.7):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    return drop


# In[80]:


def encoding_layer(embeded_rnn_input,rnn_size,keep_prob,num_layers,batch_size,source_sequence_length):

    #     forward lstm layer
    cell_fw = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size,keep_prob) for _ in range(num_layers)])

    #     backward lstm layer
    cell_bw = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size,keep_prob) for _ in range(num_layers)])
    
    ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=embeded_rnn_input,
                                    sequence_length=source_sequence_length,dtype=tf.float32)
                                                                     
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
    
    encoder_states = []
    
    for i in range(num_layers):
        if isinstance(encoder_fw_state[i],tf.contrib.rnn.LSTMStateTuple):
            encoder_state_c = tf.concat(values=(encoder_fw_state[i].c,encoder_bw_state[i].c),axis=1,name="encoder_fw_state_c")
            encoder_state_h = tf.concat(values=(encoder_fw_state[i].h,encoder_bw_state[i].h),axis=1,name="encoder_fw_state_h")
            encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
        elif isinstance(encoder_fw_state[i], tf.Tensor):
            encoder_state = tf.concat(values=(encoder_fw_state[i], encoder_bw_state[i]), axis=1, name='bidirectional_concat')
        
        encoder_states.append(encoder_state)
        
    return encoder_outputs,encoder_states


# In[67]:


def training_decoder(dec_embed_input,decoder_cell,encoder_state, output_layer,
                     target_sequence_length,max_target_length):
    
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,helper,initial_state=encoder_state,
                                              output_layer=output_layer)
    
    final_outputs, final_state,_ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,impute_finished=True,
                                                     maximum_iterations=max_target_length)
    
    return final_outputs


# In[68]:


def inference_decoder(embeddings,decoder_cell,encoder_state,output_layer,vocab_to_int,
                      max_target_length,batch_size):
    
    start_tokens = tf.tile(tf.constant(dtype=tf.int32,value=[vocab_to_int["<GO>"]]),
                           multiples=[batch_size],name="start_tokens")
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                      start_tokens=start_tokens,
                                                      end_token=vocab_to_int["<EOS>"])
    
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,helper,initial_state=encoder_state,
                                              output_layer=output_layer)
    
    final_output, final_state,_ = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True,
                                                     maximum_iterations=max_target_length)
    return final_output


# In[69]:


def decoding_layer(target_inputs,encoder_state,embedding,vocab_to_int,rnn_size,target_sequence_length,max_target_length,
                   batch_size,num_layers):
    
    vocab_len = len(vocab_to_int)
    decoder_cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(num_layers)])
    output_layer = Dense(vocab_len,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    
    embed = tf.nn.embedding_lookup(embedding,target_inputs)
    
    with tf.variable_scope("decoding"):
        
        training_logits = training_decoder(embed,decoder_cell,encoder_state,output_layer,
                                         target_sequence_length,max_target_length)
    
        
    with tf.variable_scope("decoding",reuse=True):
        
        inference_logits = inference_decoder(embeddings,decoder_cell,encoder_state,output_layer,vocab_to_int,
                                          max_target_length,batch_size)
    
    return training_logits, inference_logits


# In[70]:


def seq2seq_model(source_input,target_input,embeding_matrix,vocab_to_int,source_sequence_length,
                  target_sequence_length,max_target_length, rnn_size,keep_prob,num_layers,batch_size):
    '''
    Params : source_input,target_input,embeding_matrix,vocab_to_int,source_sequence_length,
                  target_sequence_length,max_target_length, rnn_size,keep_prob,num_layers,batch_size
    
    Return : training_logits, inference_logits
    '''
    embedings = embeding_matrix
    embed = tf.nn.embedding_lookup(embedings,source_input)
    
    encoder_output,encoder_states = encoding_layer(embed,rnn_size,keep_prob,num_layers,
                                                   batch_size,source_sequence_length)
    
    training_logits, inference_logits = decoding_layer(target_input,encoder_states,embedings,
                                                                vocab_to_int,rnn_size,target_sequence_length,
                                                                max_target_length,batch_size,num_layers)
    
    return training_logits, inference_logits


# ### Batching

# In[57]:


# Sorting the text and summary for better padding
# sort based on length of length of text
def sort_text_summary(texts,summaries):
    text_length = [(i,text,len(text)) for i,text in enumerate(texts)]
    text_length.sort(key=operator.itemgetter(2))
    
    sorted_text = [text for i,text,length in text_length]
    sorted_summary = []
    for i,text,length in text_length:
        sorted_summary.append(summaries[i])
    return sorted_text,sorted_summary


# In[58]:


sorted_text, sorted_summary = sort_text_summary(encoded_sources,encoded_targets)


# In[59]:


sorted_summary[:5]


# In[60]:


# Padding batches
def pad_sentence_batch(sentence_batch):
    max_length = max([len(sent) for sent in sentence_batch])
    padded_sentences = []
    for sent in sentence_batch:
        sent_len = len(sent)
        if len(sent) < max_length:
            padded_sentences.append(sent + [vocab_to_int["<PAD>"] for _ in range(max_length - sent_len)])
        else:
            padded_sentences.append(sent)
    return padded_sentences


# In[61]:


def get_batches(encoded_sources, encoded_targets, batch_size):
    
    '''
    Params : encoded_sources, encoded_targets, batch_size
    Return : text_batch,summary_batch,source_seq_len,target_seq_len
    '''
    
    sorted_text, sorted_summary = sort_text_summary(encoded_sources,encoded_targets)
    
    batch_count = len(sorted_text)//batch_size
    
    for i in range(batch_count):
        start = i * batch_size
        end = start + batch_size
        
        text_batch = np.array(pad_sentence_batch(sorted_text[start:end]))
        summary_batch = np.array(pad_sentence_batch(sorted_summary[start:end]))
        
        source_seq_len = [len(sent) for sent in text_batch]
        target_seq_len = [len(sent) for sent in summary_batch]
        
        yield (text_batch,summary_batch,source_seq_len,target_seq_len)
        


# In[62]:


# Hyperparametrs
epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.01
keep_probability = 0.75


# In[81]:


# Build Graph

train_graph = tf.Graph()
with train_graph.as_default():
    
    # Load the model inputs   
    input_,target,learning_rate,keep_prob,source_seq_length,target_seq_length,max_target_seq_length = model_inputs()
    
    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(input_,target,embeding_matrix,vocab_to_int,source_seq_length,target_seq_length,
                  max_target_seq_length,rnn_size,keep_probability,num_layers,batch_size)
    
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
    
    masks = tf.sequence_mask(target_seq_length, max_target_seq_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(inference_logits,target,masks)
        optimizer=tf.train.AdamOptimizer(learning_rate)
        
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# In[ ]:




