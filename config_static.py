#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import itertools
from itertools import zip_longest as zipper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
                    for anchor_tweet, other_tweet in zipper(window_tweet[anchor_index:], window_tweet[other_index:]):
                        tweet_pair = [] #  only stores the pair of tweets for simialrity computation. This is being created and deleted continously
                        if other_tweet in window_tweet[1:][-1]: # avoids TypeError on reaching the final tweet in the list
                            break
                        anchor_tweet = window_tweet[anchor_index] # make the anchor tweet constant for each iteration in the window
                        tweet_pair.append(anchor_tweet[1]), tweet_pair.append(other_tweet[1])# stores pairs for simialrity computation only
                        vectorizer = TfidfVectorizer(lowercase = False)
                        vectorised_pair = vectorizer.fit_transform(tweet_pair) # convert tweet_pair to numeric using tfidf scheme
                        cosim = np.round(cosine_similarity(vectorised_pair.toarray()[0].reshape(1,-1), vectorised_pair.toarray()[1].reshape(1,-1)).flatten(),2)
                        if cosim[0] == 0.0 or cosim[0] == 1.0:
                            discarded_pair +=1
                        elif cosim[0] > 0.5:
                            frame['Window_'+str(window)]['Top Scores'].append((anchor_tweet[0],cosim[0])) # update top scores wrt to the anchor
                        else:
                            frame['Window_'+str(window)]['All Scores'].append((anchor_index, anchor_tweet[0],cosim[0])) # updtae the frame data structure with the anchor tweet and
                            retained_pair +=1

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


# VISUALISATIONS:
    def plot_line(self, frame):
        self.frame = frame
        """This function visualises the pairwise simialrity scores for a number of windows in a frame
        using a line plot"""
        for key in frame:
            scores = [] # store top scores for every window in the frame
            for score in frame[key]['All Scores']:
                scores.append(score)
            #for score in frame[key]['Top Scores']:
            #    scores.append(score)
            plt.plot(range(len(scores)),scores)
            # format the x-axis to print integers at certain intervals:
            locator = matplotlib.ticker.MultipleLocator(10)
            plt.gca().xaxis.set_major_locator(locator)
            #formatter =  matplotlib.ticker.StrMethodFormatter("{x:0f}")
            #plt.gca().xaxis.set_major_formatter(formatter)
            plt.xlabel('Index')#, rotation=90)
            plt.ylabel('Scores')
            plt.title('Similarity between anchor and other tweets in '+key)
            plt.xticks(rotation=90)
            plt.show()

    def plot_bar(self, frame):
        """This function visualises scores and corresponding indices for each window in a frame"""
        self.frame = frame
        for window in frame:
            scores = [] # stores the top scores in each window in the frame
            indices = [] #  stores indices of the scores
            for tracked in frame[window]['Score Tracker']:
                scores.append(tracked[0])
                indices.append(tracked[1])
            plt.bar(indices, scores)
            plt.xlabel('Index')
            plt.ylabel('Score')
            plt.title('Most similar scores and indices in '+window)
            plt.xticks(rotation=90)
            plt.show()

    def plot_scatter(self, frame):
        self.frame = frame
        for window in frame:
            time_in_sec = [] # stores posting times
            sim_scores = [] # stores similarity of pairs during  posting time
            visited_items = set() # avoid duplicate values due to nested loops
            const = 3 # useful in scaling the size parameter in the scatter plot (optional)
            for item in frame[window]['All Scores']:
                if item in visited_items:
                    continue
                # check for possible occurence of ',' in time and convert to float at index 1 before appending
                if (item[1][0] == ',') and (item[1][-1]==','):
                    time_in_sec.append(float(item[1][1:-1]))
                elif (item[1][0]==','):
                    time_in_sec.append(float(item[1][1:]))
                elif (item[1][-1]==','):
                    time_in_sec.append(float(item[1][:-1]))
                else:
                    time_in_sec.append(float(item[1]))
                sim_scores.append(item[2]) # stores similarity score of pairs at index 2 in the tuple
                visited_items.add(item) # add item to set of visited items
            # visualise output as line plot and scatter plot:
            #plt.plot(time_in_sec, sim_scores) # all posting times and all scores
            #sizes = [sim_scores.count(score)**2*const for score in sim_scores] #make multiple scores standout(optional)
            sizes = [sim_scores.count(score) for score in sim_scores]
            #plt.scatter(time_in_sec, sim_scores, s=sizes) # scatter plot with standout scores
            plt.scatter(time_in_sec, sim_scores, s = sizes)
            plt.xlabel('Posting Time (s)')
            plt.ylabel('Scores')
            plt.title('Similarity between anchor and other tweets in '+window)
            plt.xticks(rotation=90)
            plt.show()


# MAIN ... run all:
if __name__=='__main__':
    """The main function to initialise/trigger the Scalable class to make its methods available"""
    #trigger = Scalable('test.csv',100) # the window_size here is being overshadowed by the window_size in tweets_batch function
    #trigger = Static('test_X.csv',101) # recent name of the class to reflect the the windowing type i.e. Static
    trigger = Static('test_X1.csv',30) # normalisedtime set as index
    stream = trigger.tweets_stream() # stream of tweets to pull out m batches
    batch = trigger.tweets_batch(stream, window_size=160)#, 313) # batch of tweets from stream of tweets
    #sim_scores = trigger.tweets_sim(batch) # print(sim_scores)
    #window = trigger.frame(batch) # single window and many anchor tweets
    frame = trigger.frame(batch, frame_size=4) #  many windows and multiple anchor tweets for each window
    #view_result0 = trigger.view_results(sim_scores) # visualise simialriy scores
    #print(frame)
    #line_plot = trigger.plot_line(frame) # visualise simialriy scores in form of line plot
    #bar_chart = trigger.plot_bar(frame) # visualise simialriy scores in form of bar chart
    scatter_plot = trigger.plot_scatter(frame) # visualise outcome as scatter plot
