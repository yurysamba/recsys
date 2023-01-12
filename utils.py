import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers
AUTOTUNE = tf.data.AUTOTUNE

def create_vocab(df):
    """
        Returns a dictionary and an inverse dictionary with an int reference for each item
    """
    
    vocab = {}
    inv_vocab = {} # inverse vocabulary
    distinct_items = list(set(np.concatenate((df['input'].values, df['output'].values))))
    for i in range(0, len(distinct_items)):
        vocab[distinct_items[i]] = i
        inv_vocab[i] = distinct_items[i]
        
    return vocab, inv_vocab

def generate_negative_samples(train_df, k=5, pwr_val=3/4):
    """
    ### Negative Sampling
          - The ratio between positive and negative samples is 1:2∼5 for large text and 1:5∼20 for small text.
          - For Negative Sampling, we define a noise distribution from which we extract some items with a certain probability
              - Instead of using the actual frequencies to define this probability function. 
                  We readjust the distribution to increase the probabilities of taking an item that has a lower frequency in our dataset
              -More details: https://www.tensorflow.org/tutorials/text/word2vec
                  Returns the training dataset after applying Negative Sampling.
                  This will turn the task into a logistic regression
        
        - For Negative Sampling, we define a noise distribution from which we extract some items with a certain probability
            - Instead of using the actual frequencies to define this probability function. 
            - We readjust the distribution to increase the probabilities of taking 
              an item that has a lower frequency in our dataset
            - More details: https://www.tensorflow.org/tutorials/text/word2vec
            
        Parameters:
        train_df: contains the pair of items: input, output
        k: number of negative samples to generate. Suggestion: 5-20
        pwr_val: hyper-paramter used to recalculate the distribution
                default value is 3/4. 1/4, 2/4 Could also be good candidate
    """
    
    observed_items = train_df.values.flatten() # returns all observed items in a list 
    freq = pd.Series(observed_items).value_counts() # returns the frequency of the items
    
    # building the noise distribution
    freq_adj = (freq ** (pwr_val)) 
    noise_dstr = freq_adj/freq_adj.sum()
    
    ### NOW WE WILL USE THIS NOISE DISTRIBUTION TO GENERATE NEGATIVE SAMPLES
    
    ## for each row in the dataset, generate k negative sample and add to an array
    arr = [] # empty array
    targets = [] 
    contexts = [] 
    labels = []
    for positive_x in train_df.values:
        # positive_x[0] --> target word
        # positive_x[1] --> context word
        # 1 is the label --> correct

        target_item = positive_x[0]
        targets.append(target_item) # append the target

        arr.append([target_item, positive_x[1], 1]) # append the positive sample

        # generate k number of negative samples
        negative_samples = np.random.choice(noise_dstr.index, 
                                                k,
                                             replace=False,
                                             p=noise_dstr.values)
        #print(positive_x[1])
        #print(negative_samples.tolist())
        
        
        contexts.append([positive_x[1]]+negative_samples.tolist())
        #print([positive_x[1]]+negative_samples.tolist())

        # append these k negative samples
        for negative_sample in negative_samples:
            arr.append([target_item, negative_sample, 0])
        
        # generating the labels
        label = [1] + [0] * k
        labels.append(label)
        
    return np.array(arr), np.array(targets), np.array(contexts), np.array(labels)


def prune_rare_words(df, min_count):
  """
    Takes as an input a dataframe with 2 columns: input and output
    Returns the same dataframe filtered for observations with a count higher than the defined value for min_count
  
    Parameter:
      df: dataframe with input, output
      min_count: occurence of pair of airport in the dataset
  """
  df['input_output'] = df['input'].astype(str) + "_" + df['output'].astype(str) # create an identifier: will make it easier when it comes to rejoining the data
  df_count = df.groupby(['input_output']).size().reset_index().rename(columns={0:'counts'}) # count the occurence of the input_output observation
  df_count = df_count[df_count.counts>=min_count].reset_index(drop=True) # filter for observations that happen more than the min_count
  df_pruned = pd.merge(df_count, df, how='inner', on=['input_output']).drop('input_output', axis=1) # filter df so that we only include relevant appearance
  
  obs_pruned = len(df)-len(df_pruned)  
  print("Using a treshold of " + str(min_count) + ": "+ str(round((obs_pruned/len(df))*100,2)) + "%" + " of observations were pruned")

  return df_pruned[['input','output']]

def subsample(df, treshold=0.005):
    """
    Subsample all observations that appear more often then the defined treshold.
    The value of the treshold should depend on the observed frequencies in the dataset.

    Parameters:
        df: Pandas Dataframe that contains the set of input, output airports
        treshold: default value is 0.005. 

    """
    df_count = df.groupby(['input', 'output']).size().reset_index().rename(columns={0:'counts'})
    df_count['freq'] = df_count.counts/df_count.counts.sum() # computing the frequency
    df_count['discard_prob'] = (1-np.sqrt((treshold/df_count['freq'])))# defining the probability of discarding the pair of airport 
    df_count['discard_prob'] = np.where(df_count['discard_prob']<=0, 0, df_count['discard_prob']) # return a 0% prob. of discarding when the pair of airport appears less than the treshold
    df_count['adj_counts'] = ((1-df_count['discard_prob'])*df_count['counts']).astype(int) # returns the number of time it should appear in dataset

    arr = [] # array to append observations according to their adjusted counts
    for index, row in df_count.iterrows(): # iterate through the rows
        for i in range(0, int(row['adj_counts'])): # append the input, output as often as the adjusted count
            arr.append([row['input'], row['output']])

    df = pd.DataFrame(arr, columns=['input', 'output'])

    return df



class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, n_negative_samples):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                          embedding_dim,
                                          input_length=1,
                                          name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                           embedding_dim,
                                           input_length=n_negative_samples+1)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots


