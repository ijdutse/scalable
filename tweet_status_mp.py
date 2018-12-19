#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import matplotlib.ticker
import itertools
from random import randint
from itertools import zip_longest as zipper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
sns.set()
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
import time
mp = Pool()


class TweetStatus(object):
    """Python class to compute similarity between n tweets in a finite window that can be scaled to accommodate tweets efficiently.
    Initialise/trigger the class to create instances by supplying path to the file containing your data and the
    the window size or batch size to return"""

    def get_data_list(self, datapath, chunksize):
        self.datapath = datapath
        self.chunksize = chunksize

        datachunks = pd.read_csv(datapath, chunksize = chunksize)# read data in chunks, specify by user ...
        datasize = len(pd.read_csv(datapath)) # size of the file for computing epochs
        epochs = int(datasize/chunksize) # get the number of epochs to read the file as function of size/chunksize...
        listed_data_tuples = [] # store list of all data tuples to be extracted from the main file ...
        for e in range(epochs):
            data_tuple = []
            try:
                data_chunk = next(datachunks) #print(len(data_chunk))
                values = data_chunk.values
                #data_tuple = []
                for ptime, tweet, period, fav, fol, friend in zip(values[:,0],values[:,1],values[:,2],\
                     values[:,3],values[:,4],values[:,5]):
                    data_tuple.append((ptime,tweet.strip(' ,'), period, fav, fol, friend))
            except StopIteration:
                break
            listed_data_tuples.append(data_tuple)

        return listed_data_tuples


    def get_tweet_status(self, data_tuple):
        """This function takes list of tweets and return set of windows. A frame is a unit of computation that can take very
        large file, break it into finite windows and return the associated metrics in each window. Each window consist of list of Top Scores
        and a dictionary of tracker consisting of Top scores for each anchor tweet and the corresponding number of the top scores
        The frame also contains top scores and corresponding indices in each window"""

        self.data_tuple = data_tuple
        frame_size = 10
        window_size = int(len(data_tuple)/frame_size) # size of each widnow in the frame
        window_tweets = np.array_split(data_tuple, frame_size) # split tweets into window tweets according to the frame size, default is 3
        frame = {} # initialise empty frame to store all window instances and associated values
        dfs = pd.DataFrame()
        k = 0 # initialise stopping criteria
        while k < frame_size:
            for window, window_tweet in zip(range(window_size),window_tweets):
                frame['Window_'+str(window)] = {'AnchorID':[],'AnchorTweet':[],'OtherTweet':[],'PostingTimes':[], 'RelativeTime':[],\
                'PairIndices':[],'CoSim':[],'Discards':[],'FavCount':[],'FolCount':[],'Friends':[],'Period':[]}#,'Counter':[]}
                anchor_index = 0
                other_index = 1
                while other_index < len(window_tweet)-1:
                    discarded_pair = 0 # keeps track of number of pairs with simialrity of 0.0 or 1.0
                    retained_pair = 0 # keeps track of number of pairs greater than 0.0 and less than 1.0
                    tracked_index = 2# track all other indices
                    for anchor_tweet, other_tweet in zipper(window_tweet[anchor_index:], window_tweet[other_index:]):
                        try:
                            tweet_pair = [] #  only stores the pair of tweets for simialrity computation. This is being created and deleted continously
                            anchor_tweet = window_tweet[anchor_index] # make the anchor tweet constant for each iteration in the window
                            tweet_pair.append(anchor_tweet[1]), tweet_pair.append(other_tweet[1])# stores pairs for simialrity computation only
                            vectorizer = TfidfVectorizer(max_features= 3,lowercase = False)
                            vectorised_pair = vectorizer.fit_transform(tweet_pair) # convert tweet_pair to numeric using tfidf scheme
                            cosim = np.round(cosine_similarity(vectorised_pair.toarray()[0].reshape(1,-1), vectorised_pair.toarray()[1].reshape(1,-1)).flatten(),2)
                            if cosim[0] == 0.0 or cosim[0] == 1.0:
                                discarded_pair +=1 # keep track of less important scores
                                continue
                            frame['Window_'+str(window)]['AnchorID'].append('A_'+str(anchor_index)) # stores the id of the anchor
                            frame['Window_'+str(window)]['AnchorTweet'].append(anchor_tweet[1]) # stores the anchor tweet
                            frame['Window_'+str(window)]['OtherTweet'].append(other_tweet[1]) # stores the tweet to compare with
                            frame['Window_'+str(window)]['PostingTimes'].append((anchor_tweet[0],other_tweet[0])) # stores posting times of the pair
                            frame['Window_'+str(window)]['RelativeTime'].append((int(other_tweet[0])-int(anchor_tweet[0]))) # relative posting time/time difference
                            frame['Window_'+str(window)]['PairIndices'].append((anchor_index,tracked_index)) # stores the indices of the pair
                            frame['Window_'+str(window)]['CoSim'].append(cosim[0]) # stores the indices of the pair
                            frame['Window_'+str(window)]['Discards'].append(discarded_pair) # stores the indices of the pair
                            frame['Window_'+str(window)]['FavCount'].append(anchor_tweet[3]) # stores the favourite counts history
                            frame['Window_'+str(window)]['FolCount'].append(anchor_tweet[4]) # stores the number of followers
                            frame['Window_'+str(window)]['Friends'].append(anchor_tweet[5]) # stores the number of friends ... specific time of the day also needed
                            frame['Window_'+str(window)]['Period'].append(anchor_tweet[2]) # period of the day, e.g. morning or afternoon
                            retained_pair +=1
                            tracked_index+=1 # update the index of the inner tweet being compared with the anchor
                        except:
                            continue

                    anchor_index +=1 # pick the next tweet as the next anchor
                    other_index +=1 # shrink the window size by a factor of 1
            k+=1 # update stopping criteria:
            #dfs = pd.DataFrame() # instantiate empty dataframe to store all windows in the frame ...
            for key in frame.keys(): # update the dataframe ....
                df = pd.DataFrame(frame[key])
                dfs = dfs.append(df)
        return dfs

# MAIN ... run all:
if __name__=='__main__':
    trigger = TweetStatus()
    data_tuples = trigger.get_data_list('sbt_all_extracts_numeric_columns_del.csv', chunksize=201)
    ###################################################

    # MULTIPROCESSING/USING MULTIPLE CORES ...
    mp = Pool()
    start = time.time()
    dfs = pd.DataFrame()
    df = mp.map(trigger.get_tweet_status,data_tuples)
    dfs = dfs.append(df)
    #mp.close()
    #mp.join()
    print(len(dfs))
    print(dfs)
    stop = time.time() - start
    print(stop)

    # SEQUENTIAL/USING SINGLE CORE:
    """start1 = time.time()
    dfs = pd.DataFrame()
    for data in data_tuples:
        df = trigger.get_tweet_status(data)
        dfs = dfs.append(df)
    print(len(dfs))
    print(dfs)
    stop1 = time.time() - start1
    print(stop1)"""
