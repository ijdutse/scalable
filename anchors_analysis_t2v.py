from collections import defaultdict
import pandas as pd
import numpy as np
import preprocessor as p
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyprind
pbar = pyprind.ProgBar(21)
import time
import matplotlib.pyplot as plt
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from itertools import zip_longest as zipper
import seaborn as sns
sns.set()


class Tweet2Vec():
    """A simple class to read a file of preprocessed tweets in batches, train a doc2vec model and then compute
    the similarities of random tweets. This is similar the earlier approch that utilises cosine similarity"""

    def get_tweets_stream(self, filesource):
        """accepts filesource and return stream of posting time and tweets as generator object"""
        self.filesource = filesource
        with open(self.filesource, 'r') as csv:
            next(csv)
            for line in csv:
                ptime, tweet = line[0:5], line[5:-1] # this returns both posting times and tweets
                #tweet = line[22:-1]
                yield ptime.strip(' ,'), tweet.strip(' ,') # remove trailing whitespace or comma

    def tweets_mini_batch(self, tweets_stream, batch_size=3000):
        """get tweets in batches and assign tag to each tweet"""
        self.tweets_stream = tweets_stream
        self.batch_size = batch_size
        ptimes = [] # stores posting times as list
        tweets = [] # stores list of tweets in the batch
        try:
            for _ in range(batch_size):
                ptime, tweet = next(tweets_stream)
                ptimes.append(ptime)
                tweets.append(tweet)
        except StopIteration:
            return None, None
        for ptime, tweet in zip(ptimes,tweets):
            #tweet_sentence=TaggedDocument(words=tweet.split(),tags=[' '.join(tweet.split()[:2]),'P_'+str(ptime)])
            tweet_sentence=TaggedDocument(words=tweet.split(),tags=['P_'+str(ptime)])
            yield tweet_sentence

    def tweet2vec(self, tagged_tweets):
        """accepts tagged tweets for training Doc2Vec model on tweets collection"""
        self.tagged_tweets = tagged_tweets
        model = Doc2Vec(vector_size = 50, window = 3, min_count = 1, workers = 11, alpha = 0.025, epochs = 10)
        model.build_vocab(tagged_tweets)
        tic=time.time()
        model.train(tagged_tweets, total_examples = model.corpus_count, epochs = model.epochs)
        toc = time.time()
        print('Training completed successfully!')
        model.save('trained_doc2vec_model.pkl')# save the trained Doc2Vec
        return model
# second class to build on the previous class:


class Tweet2VecSim():
    """this class accepts collection of tweets in dataframe and compute similarities between each pair of tweets"""
    #def __init__(self, df):
     #   self.df = df
    def pair_sim(self, df):
        """this function accepts a csv file containing tweets and corresponding posting times to compute
        pairwise similarity between each pair"""
        self.df = df
        extracts = {'PostingTime':[],'Anchors':[],'SimScores':[]}
        anchor_index = 0
        other_index = 1
        for anchor_time in df.TimeInSec[anchor_index:]:
            anchor_tag = 'P_'+str(anchor_time)
            if other_index == len(df.TimeInSec)-1: #stop execution on reaching the end
                break
            for other_time in df.TimeInSec[other_index:]:
                other_tag = 'P_'+str(other_time)
                time_diff = other_time - anchor_time
                cosim = model.docvecs.similarity(anchor_tag, other_tag)#compute similarity between tweets pair
                extracts['PostingTime'].append(time_diff)
                extracts['Anchors'].append((anchor_tag, other_tag))
                extracts['SimScores'].append(round(cosim,3))
            other_index+=1 #increment other_tweet index
        tf = pd.DataFrame(extracts)
        tf['Similar?'] = tf.SimScores.apply(lambda x: 'Yes' if x>5.0 else 'No') #tf.to_csv('tweet2vec.csv')
        return tf

    def bin_results(self, df):
        """this function accepts a dataframe from pair_sim function and returns summarised results in bins"""
        self.df = df
        time_bins = np.linspace(df.PostingTime.min(), df.PostingTime.max(), num=101,dtype=np.int64)
        item_bin_index = np.digitize(df.PostingTime, time_bins) # get the index of each item in the unbinned data
        df['ItemBinIndex'] = item_bin_index # add the column containing the time bin of each data item
        # to include the actual bin for each time:
        time_bins_list = time_bins.tolist() # convert time_bins to list from np.ndarray to support appending
        periods = [] # stores binned posting times as periods
        for index in df.ItemBinIndex:
            periods.append(time_bins_list[index])
            time_bins_list.append(time_bins_list[index])#replace item back tolist to avoid exhausting thelist prem..
        df['TimeInterval'] = periods #df.to_csv('t2v_binned_test_file1.csv') # save updated file

        # store binned values for further transformation
        binned_values = defaultdict(list) # create a default dictionary to store bins and corresponding scores
        for p, s in zip(df.TimeInterval, df.SimScores):
            binned_values[int(p)].append(s)

        # create a new data structure, BINS, to contain summarised values for each bin in bins_values
        bin_summary = defaultdict(list)
        for item, key in zip(range(len(binned_values)),binned_values.keys()):
            similar = []
            dissimilar = []
            for score in binned_values[key]:
                if score > 0.5: # try different threshold values e.g. 0.75
                    similar.append(score)
                else:
                    dissimilar.append(score)
            bin_summary[key].append((len(np.array(binned_values[key])),round(np.array(binned_values[key]).mean(),\
                                    3),round(np.array(similar).mean(),3),round(np.array(dissimilar).mean(),3),\
                                   len(similar),len(dissimilar)))
        summary_bin = pd.DataFrame(bin_summary).T #convert to Dataframe and transpose the file before saving:

        # extract relevant metrics and save final outputs for visualisations:
        summarised_bin = {'PostingTime':[],'N_Items':[],'MeanScores':[],'MeanSimScores':[],'MeanDisScores':[],\
                                     'N_SimItems':[],'N_DisItems':[]}
        for key in bin_summary.keys():
            summarised_bin['PostingTime'].append(key)
            for item in bin_summary[key]:
                summarised_bin['N_Items'].append(item[0])
                summarised_bin['MeanScores'].append(item[1])
                summarised_bin['MeanSimScores'].append(item[2])
                summarised_bin['MeanDisScores'].append(item[3])
                summarised_bin['N_SimItems'].append(item[4])
                summarised_bin['N_DisItems'].append(item[5])
        t2v_bins_df = pd.DataFrame(summarised_bin) # convert to dataframe
        #save final outputs file for visualisations: #t2v_bins_df.to_csv('t2v_summarised_bins.csv')
        return t2v_bins_df

    def plot_bins(self, binned_df):
        """This function accepts the final output i.e. summarised bins for various visualisations using plt/sns"""
        self.binned_df = binned_df
        #some visuals:
        sns.pairplot(binned_df,hue='PostingTime', size=5.7, vars=['PostingTime','MeanSimScores'])
        #sns.pairplot(binned_df,hue='PostingTime',size=2.5)
        #sns.pairplot(binned_df,hue='MeanSimScores',size=2.5) # using all variables in the datframe
        plt.title('Mean Scores and relative posting time of most similar tweets in a frame')
        # PairGrid plot:
        g = sns.PairGrid(binned_df, vars=['PostingTime', 'MeanSimScores'],
                     hue='PostingTime', palette='RdBu_r')
        g.map(plt.scatter, alpha=0.8) #g.add_legend()
        plt.show()

"""if __name__=='__main__':
    f = pd.read_csv('subject-specific_test_tweets_sec.csv')
    f = f[:1500]
    q = Tweet2VecSim()
    pair_sims = q.pair_sim(f)
    binned_results = q.bin_results(pair_sims)
    print(binned_results)
    # visualise output: visuals = q.plot_bins(binned_results)"""


if __name__=='__main__':
    p = Tweet2Vec()
    stream = p.get_tweets_stream('subject-specific_test_tweets_sec.csv')
    tagged_tweets = p.tweets_mini_batch(stream, 20000)
    model = p.tweet2vec(tagged_tweets)
    # save the trained Doc2Vec: model.save('trained_doc2vec_model.pkl')
    print(model.docvecs.most_similar('P_47115'))
    print(model.docvecs.similarity('P_47115','P_47098'))
    # compute pairwise similarities:
    #df = pd.read_csv('subject-specific_test_tweets_sec.csv')
    #gf = p.pair_sim(df3)
    #visualise binned results

    # Using the second class:
    f = pd.read_csv('subject-specific_test_tweets_sec.csv')
    f = f[:1500]
    q = Tweet2VecSim()
    pair_sims = q.pair_sim(f)
    binned_results = q.bin_results(pair_sims)
    print(binned_results)
    visuals = q.plot_bins(binned_results)
