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


class Static(object):
    """Python class to compute similarity between n tweets in a finite window that can be scaled to accommodate tweets efficiently.
    Initialise/trigger the class to create instances by supplying path to the file containing your data and the
    the window size or batch size to return"""

    def __init__(self, dataframe, window_size): # defines the constructor:
         self.dataframe = dataframe
         self.window_size = window_size

    def tweets_stream(self):
        """ THis function iteratively yield tweets from a csv file containing tweets"""
        with open(self.dataframe, 'r') as csv:
            next(csv) # this skips the header in the file
            for line in csv:
                time, tweet = line[0:5], line[5:-1] # tuple of posting times converted to seconds and tweets
                #time, tweet = line[0:9], line[9:-1] # tuple of posting times converted to seconds and scaled logathrimically
                #time, tweet = line[0:9],line[19:-1] # try this
                yield time, tweet

    def tweets_batch(self,stream, window_size = 50):
        """This function utilises the tweets_stream generator function to return m number of tuples of posting times and tweets as list"""
        self.stream = stream
        self.window_size = window_size
        time_tweet = [] # stores posting time and tweets in batches
        try:
            for _ in range(self.window_size):
                time, tweet = next(stream)
                time_tweet.append((time, tweet))
        except StopIteration:
            return None, None
        return time_tweet

    def frame(self, tweets_list, frame_size = 3):
        """This function takes list of tweets and return set of windows. A frame is a unit of computation that can take very
        large file, break it into finite windows and return the associated metrics in each window. Each window consist of list of Top Scores
        and a dictionary of tracker consisting of Top scores for each anchor tweet and the corresponding number of the top scores
        The frame also contains top scores and corresponding indices in each window"""
        self.tweets_list = tweets_list # list of tweets to be broken into n windows
        self.frame_size = frame_size # number of windows in each frame
        window_size = int(len(tweets_list)/frame_size) # size of each widnow in the frame
        window_tweets = np.array_split(tweets_list, frame_size) # split tweets into window tweets according to the frame size, default is 3
        frame = {} # initialise empty frame to store all window instances and associated values
        k = 0 # initialise stopping criteria
        while k < frame_size:
            for window, window_tweet in zip(range(window_size),window_tweets):
                frame['Window_'+str(window)] = {'All Scores':[],'Counter':[], 'Score Tracker':[], 'Top Scores':[]} # for each window store its top scores, tuple of top scores and indices
                anchor_index = 0
                other_index = 1
                while other_index < len(window_tweet)-1:
                    discarded_pair = 0 # keeps track of number of pairs with simialrity of 0.0 or 1.0
                    retained_pair = 0 # keeps track of number of pairs greater than 0.0 and less than 1.0
                    tracked_index = 2# track all other indices
                    for anchor_tweet, other_tweet in zipper(window_tweet[anchor_index:], window_tweet[other_index:]):
                        tweet_pair = [] #  only stores the pair of tweets for simialrity computation. This is being created and deleted continously
                        if other_tweet in window_tweet[1:][-1]: # avoids TypeError on reaching the final tweet in the list
                            break
                        anchor_tweet = window_tweet[anchor_index] # make the anchor tweet constant for each iteration in the window
                        tweet_pair.append(anchor_tweet[1]), tweet_pair.append(other_tweet[1])# stores pairs for simialrity computation only
                        vectorizer = TfidfVectorizer(max_features= 3,lowercase = False)
                        vectorised_pair = vectorizer.fit_transform(tweet_pair) # convert tweet_pair to numeric using tfidf scheme
                        cosim = np.round(cosine_similarity(vectorised_pair.toarray()[0].reshape(1,-1), vectorised_pair.toarray()[1].reshape(1,-1)).flatten(),2)
                        if cosim[0] == 0.0 or cosim[0] == 1.0:
                            discarded_pair +=1
                        elif cosim[0] > 0.85:
                            #try:# avoid posting times with some commas
                            frame['Window_'+str(window)]['Top Scores'].append(((anchor_index,tracked_index),(anchor_tweet[0],other_tweet[0]),cosim[0]))
                            #except:
                            #    continue
                            #frame['Window_'+str(window)]['Top Scores'].append(((anchor_index,tracked_index),(anchor_tweet[0],other_tweet[0]),cosim[0])) # update top scores wrt to the anchor
                        else:# anchor_tweet[0] and other_tweet[0] refers to the posting times used in computing relative posting time
                            #try:
                            frame['Window_'+str(window)]['All Scores'].append(((anchor_index,tracked_index), (anchor_tweet[0],other_tweet[0]),cosim[0])) # updtae the frame data structure with the anchor tweet and
                            #except:
                            #    continue
                            retained_pair +=1
                        tracked_index+=1 # update the index of the inner tweet being compared with the anchor
                    frame['Window_'+str(window)]['Counter'].append((anchor_index, retained_pair)) # updtae the frame data structure with the anchor tweet and
                    # update the score Tracker
                    """q = np.array(frame['Window_'+str(window)]['All Scores']) # convert list of scores in pos. in the tuple to np array for computational ease
                    while q.shape[0]>1: # execute as long as length of the array is > 1
                        frame['Window_'+str(window)]['Score Tracker'].append((q.max(),q.argmax())) # track top scores and their indices in windows
                        q = np.delete(q, q.argmax())""" #  delete used score and index
                    # update indices of anchor and other tweets
                    anchor_index +=1 # pick the next tweet as the next anchor
                    other_index +=1 # shrink the window size by a factor of 1
            # update stopping criteria:
            k+=1
        return frame

# VISUALISATION FUNCTIONS:
    # for anchors in a window
    def get_all_anchors(self, frame):
        """This function returns all anchors & corresponding posting time and similarity
        magnitude with others in each window and select random anchors for visualisation"""
        self.frame = frame
        anchors = {}
        anchors_frame = {'Window':[],'Anchors':[],'Posting Time':[], 'Scores':[],'Similar?':[]}
        for window in frame:
            anchors[window] = {} # initialise anchors in each window of the frame to empty dictionary
            for i in range(len(frame[window]['All Scores'])): # for the number of anchors in each window
                anchors[window]['Anchor_'+str(i)] = [] # initalise each anchor to empty list to store time and score
                for item in frame[window]['All Scores']:
                    anchors[window]['Anchor_'+str(i)].append((int(item[1][1])-int(item[1][0]),item[2]))
        # extract relative posting time and scores for random anchors in the frame to analyse/visualise:
        for w in range(randint(0,len(frame))):# w for random window and
            for a in range(randint(0, 127)): #  a for random anchor
                relative_time = []
                sim_score = []
                try: # avoid empty anchors terminating the process
                    for time,score in zip(anchors['Window_'+str(w)]['Anchor_'+str(a)], anchors['Window_'+str(w)]['Anchor_'+str(a)]):
                        relative_time.append(time[0])
                        sim_score.append(score[1])
                        #create dataframe with:
                        anchors_frame['Window'].append(w)
                        anchors_frame['Anchors'].append(a)
                        anchors_frame['Posting Time'].append(time[0])
                        anchors_frame['Scores'].append(score[1])
                        if score[1]> 0.5:
                            anchors_frame['Similar?'].append('Yes')
                        else:
                            anchors_frame['Similar?'].append('No')
                except:
                    continue
        anchors_df = pd.DataFrame(anchors_frame)
        return anchors_df

    # function to bin outputs:
    def time_binning(self, df):
        """This function accepts a dataframe created from the execution of the frame function in the class"""
        self.df = df
        #create posting time bins:
        time_bins = np.linspace(df['Posting Time'].min(), df['Posting Time'].max(), 101,dtype=np.int64)
        item_bin_index = np.digitize(df['Posting Time'], time_bins) # get the index of each item in the unbinned data
        df['ItemBinIndex'] = item_bin_index # add the column containing the time bin of each data item

        # associate each bin with the actual time period/interval:
        time_bins_list = time_bins.tolist() # convert time_bins to list from np.ndarray to support appending
        periods = [] # stores binned posting times as periods
        for index in df.ItemBinIndex:
            periods.append(time_bins_list[index])
            time_bins_list.append(time_bins_list[index])#replace item back to ist to avoid exhausting the list b4 end
        df['TimeInterval'] = periods

        binned_values = defaultdict(list) # create a default dictionary to store bins and corresponding scores
        for p, s in zip(df.TimeInterval, df.Scores):
            binned_values[int(p)].append(s)

        # create a new data structure, to contain summarised values for each bin in bins_values
        bin_summary = defaultdict(list)
        for item, key in zip(range(len(binned_values)),binned_values.keys()):
            similar = []
            dissimilar = []
            for score in binned_values[key]:
                if score > 0.5:
                    similar.append(score)
                else:
                    dissimilar.append(score)
            bin_summary[key].append((len(np.array(binned_values[key])),round(np.array(binned_values[key]).mean(),3),\
                                    round(np.array(similar).mean(),3),round(np.array(dissimilar).mean(),3),\
                                   len(similar),len(dissimilar)))

        # extract relevant metrics for visualisations:
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
        # convert extracts to dataframe for ease of interaction and visualisations:
        bins_df = pd.DataFrame(summarised_bin)
        return bins_df

    def plot_bins(self, bins_df):
        """This function accepts the output from time_binning as summarised bins"""
        self.bins_df = bins_df
        # some visuals:
        #sns.pairplot(bins_df,hue='PostingTime',size=2.5)
        sns.pairplot(bins_df,hue='PostingTime', size=5.7, vars=['PostingTime','MeanSimScores'])
        plt.title('Mean Scores and relative posting time of most similar tweets in a frame')
        #sns.pairplot(bins_df,hue='MeanSimScores',size=2.5) # using all variables in the datframe
        # PairGrid plot:
        """g = sns.PairGrid(bins_df, vars=['PostingTime', 'MeanSimScores'],
                     hue='PostingTime', palette='RdBu_r')
        g.map(plt.scatter, alpha=0.8)
        g.add_legend()"""
        plt.show()


# MAIN ... run all:
if __name__=='__main__':
    """The main function to initialise/trigger the Scalable class to make its methods available"""
    trigger = Static('subject-specific_test_tweets_sec.csv',33) # using time in seconds
    stream = trigger.tweets_stream() # stream of tweets to pull out m batches
    batch = trigger.tweets_batch(stream, window_size=290)#, 313) # batch of tweets from stream of tweets
    frame = trigger.frame(batch, frame_size=4) #  many windows and multiple anchor tweets for each window
    all_anchors = trigger.get_all_anchors(frame) # plot similarity between a  random anchor and other tweets and corresponding posting times
    bins = trigger.time_binning(all_anchors) # summarise data via binning
    # VISUALISE RESULTS:
    view_bins = trigger.plot_bins(bins)
