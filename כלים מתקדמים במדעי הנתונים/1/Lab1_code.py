#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nest_asyncio
nest_asyncio.apply()

import twint

import pandas as pd
import tweepy 
from pymongo import MongoClient

import pandas as pd
import numpy as np
import tweepy 
from pymongo import MongoClient


consumer_key='HHy2xCHiAYKeNoBldCveExVr6'
consumer_secret='zhOqAuL0gACGTdbTXNWs5iEr3v9HHvItCg95JvFZSwesNocyuj'
access_token='1227238114621394950-71yLs1e2sFkLfqdKe6267X9Zoh1DtB'
access_token_secret='0xOG1l28QomDza8qIxXimxqL8fShNQ4bTH3mTsdDz7wPz'

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth, wait_on_rate_limit=True)
api.verify_credentials()


from tkinter import *

r = Tk()
r.title('Find Tweet')
r.geometry("1600x900")

kav = Label(r, text='                ')
kav.grid(row = 0, column = 0)

myLabel = Label(r, text='\nTWEEPY API\nLets download some tweets :)', font=("Arial", 25))
myLabel.grid(row = 0, column = 2)
kav = Label(r, text='------------------------------' + 
                 '--------------------------------------------------------------')
kav.grid(row = 1, column = 2)

kav = Label(r, text='      ')
kav.grid(row = 1, column = 4)

########################################################

myLabel_2 = Label(r, text='\nTWINT API\nLets download some tweets :)', font=("Arial", 25))
myLabel_2.grid(row = 0, column = 6)
kav_2 = Label(r, text='------------------------------' + 
                 '--------------------------------------------------------------')
kav_2.grid(row = 1, column = 6)

# Collect tweets

def collect_tweets(title, count, language):
    
    tweets = tweepy.Cursor(api.search_tweets,
                  q=title,
                  lang=language).items(count)
    return tweets
        
        
def convert_tweet_to_tuple(tweet):
    
    screen_name = tweet.user.screen_name
    full_name = tweet.author.name
    retweet_count = tweet.retweet_count
    followers_count = tweet.user.followers_count
    screen_name = tweet.user.screen_name
    content = tweet.text
    description = tweet.user.description
    
    post_tuple = (screen_name, 
                  full_name,
                  followers_count, 
                  description,
                  content,
                  retweet_count)
    
    return post_tuple


def save_to_db(db_name, tweets_df):
    
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    
    mydatabase = client['tweet_DB']

    # Access collection of the database
    mycollection=mydatabase[db_name]

    rec = tweets_df.to_dict('records')

    # inserting the data in the database
    mycollection.insert_many(rec)

def inputPrint():
    
    tweets = collect_tweets(e1.get(), int(e2.get()), e3.get())
    post_tuples = []

    for tweet in tweets:
        post_tuple = convert_tweet_to_tuple(tweet)
        post_tuples.append(post_tuple)
        
    tweets_df = pd.DataFrame(post_tuples, columns=['username', 'full_name',  'followers', 'description', 'tweet', 'retweets'])
        
    save_to_db(e1.get(), tweets_df)


def followers():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    mydatabase = client['tweet_DB']
    
    mycollection=mydatabase[e4.get()]
    t = list(mycollection.find())
    df = pd.DataFrame(t)
    
    fol = max(df['followers'])    
    user = df[df['followers'] == fol]
    
    myLabel1 = Label(r, text='Username: @' + user['username'].values[0] + 
                     '\n\nNum of Followers: ' + str(fol), bg='lightblue')
    
    myLabel1.grid(row = 12, column = 2)
    
def retweets():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    mydatabase = client['tweet_DB']
    
    mycollection=mydatabase[e4.get()]
    t = list(mycollection.find())
    df = pd.DataFrame(t)
    
    fol = max(df['retweets'])    
    user = df[df['retweets'] == fol]
    
    myLabel1 = Label(r, text='Username: @' + user['username'].values[0] + 
                     '\n\nNum of Retweets: ' + str(fol) + 
                     '\n\nTweet: ' + user['tweet'].values[0], bg='lightblue')
    
    myLabel1.grid(row = 12, column = 2)

    
def tweets():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    mydatabase = client['tweet_DB']
    
    mycollection=mydatabase[e4.get()]
    t = list(mycollection.find())
    df = pd.DataFrame(t)
    
    index = pd.Index(df['tweet'])
    size = index.value_counts()
    
    user = df[df['tweet'] == size.index[0]]
    
    myLabel1 = Label(r, text='Username: @' + user['username'].values[0] + 
                     '\n\nNum of Repeats: ' + str(size.values[0]) + 
                     '\n\nTweet: ' + size.index[0], bg='lightblue')
    
    myLabel1.grid(row = 12, column = 2)
    
##########################################################################  
    
def twint_input(city, lim):
    
    name = str(city) + ".csv"
    c = twint.Config()
    c.Limit = lim
    c.Store_csv = True
    c.Output = name
    c.Near = city
    c.Hide_output = True
    c.Count = True
    c.Stats = True
    twint.run.Search(c)

    test_df = pd.read_csv(name)
    
    return test_df
    
def save_by_city():

    client = MongoClient()
    client = MongoClient('localhost', 27017)

    mydatabase = client['tweet_DB']
    
    city = str(e1_2.get())
    
    test_df = twint_input(city, int(e2_2.get()))
                
    mycollection=mydatabase[city]

    rec = test_df.to_dict('records')

    mycollection.insert_many(rec)
    

def twint_likes():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    mydatabase = client['tweet_DB']
    
    mycollection=mydatabase[e4_2.get()]
    t = list(mycollection.find())
    df = pd.DataFrame(t)
    
    fol = max(df['likes_count'])    
    user = df[df['likes_count'] == fol]
    
    myLabel1 = Label(r, text='Username: @' + user['username'].values[0] + 
                     '\n\nNum of Likes: ' + str(fol)+ 
                     '\n\nTweet: ' + user['tweet'].values[0], bg='lightblue')
    
    myLabel1.grid(row = 12, column = 6)
    
def twint_retweets():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    mydatabase = client['tweet_DB']
    
    mycollection=mydatabase[e4_2.get()]
    t = list(mycollection.find())
    df = pd.DataFrame(t)
    
    fol = max(df['retweets_count'])    
    user = df[df['retweets_count'] == fol]
    
    myLabel1 = Label(r, text='Username: @' + user['username'].values[0] + 
                     '\n\nNum of Retweets: ' + str(fol) + 
                     '\n\nTweet: ' + user['tweet'].values[0], bg='lightblue')
    
    myLabel1.grid(row = 12, column = 6)

    
def twint_tweets():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    mydatabase = client['tweet_DB']
    
    mycollection=mydatabase[e4_2.get()]
    t = list(mycollection.find())
    df = pd.DataFrame(t)
    
    index = pd.Index(df['tweet'])
    size = index.value_counts()
    
    user = df[df['tweet'] == size.index[0]]
    
    myLabel1 = Label(r, text='Username: @' + user['username'].values[0] + 
                     '\n\nNum of Repeats: ' + str(size.values[0]) + 
                     '\n\nTweet: ' + size.index[0], bg='lightblue')
    
    myLabel1.grid(row = 12, column = 6)

    
e1 = Entry(r, width=50, bg='lightblue', borderwidth=5)    
e1.insert(0,'Tweet Topic')
e1.grid(row = 2, column = 2)

e2 = Entry(r, width=50, bg='lightblue', borderwidth=5)    
e2.insert(0,'Number of Tweets')
e2.grid(row = 3, column = 2)

e3 = Entry(r, width=50, bg='lightblue', borderwidth=5)    
e3.insert(0,'Language')
e3.grid(row = 4, column = 2)

##############################################################################

e1_2 = Entry(r, width=50, bg='lightblue', borderwidth=5)    
e1_2.insert(0,'City')
e1_2.grid(row = 2, column = 6)

e2_2 = Entry(r, width=50, bg='lightblue', borderwidth=5)    
e2_2.insert(0,'Limit')
e2_2.grid(row = 3, column = 6)


button = Button(r, text='SEARCH', width=15,height=5, pady=10, padx=10, fg='white', bg='green', command=inputPrint)
button.grid(row = 5, column = 1)

button2 = Button(r, text='EXIT', width=15,height=5, pady=10, padx=10, fg='white', bg='red', command=r.destroy)
button2.grid(row = 5, column = 3)


button_2 = Button(r, text='SEARCH', width=15,height=5, pady=10, padx=10, fg='white', bg='green', command=save_by_city)
button_2.grid(row = 5, column = 5)

button2_2 = Button(r, text='EXIT', width=15,height=5, pady=10, padx=10, fg='white', bg='red', command=r.destroy)
button2_2.grid(row = 5, column = 7)

#######################################################

kav2 = Label(r, text='\n------------------------------' + 
                 '--------------------------------------------------------------\n')
kav2.grid(row = 6, column = 2)


myLabel2 = Label(r, text='So, lets find some tweets :)', font=("Arial", 15))
myLabel2.grid(row = 7, column = 2)

e4 = Entry(r, width=50, bg='lightblue', borderwidth=5)    
e4.insert(0,'Find by Topic')
e4.grid(row = 8, column = 2)

kav3 = Label(r, text='\n------------------------------' + 
                 '--------------------------------------------------------------\n')
kav3.grid(row = 9, column = 2)

followers = Button(r, text='MAX \nFOLLOWERS', width=15,height=5, pady=10, padx=10, fg='red', bg='yellow', command=followers)
followers.grid(row = 10, column = 1)

retweets = Button(r, text='MAX \nRETWEETS', width=15,height=5, pady=10, padx=10, fg='red', bg='yellow', command=retweets)
retweets.grid(row = 10, column = 2)

tweets = Button(r, text='MAX \nREPEATED \nTWEETS', width=15,height=5, pady=10, padx=10, fg='red', bg='yellow', command=tweets)
tweets.grid(row = 10, column = 3)

kav4 = Label(r, text='\n------------------------------' + 
                 '--------------------------------------------------------------\n\n')
kav4.grid(row = 11, column = 2)

########################################################

kav2_2 = Label(r, text='\n------------------------------' + 
                 '--------------------------------------------------------------\n')
kav2_2.grid(row = 6, column = 6)


myLabel2_2 = Label(r, text='So, lets find some tweets :)', font=("Arial", 15))
myLabel2_2.grid(row = 7, column = 6)

e4_2 = Entry(r, width=50, bg='lightblue', borderwidth=5)    
e4_2.insert(0,'Find by City')
e4_2.grid(row = 8, column = 6)

kav3_2 = Label(r, text='\n------------------------------' + 
                 '--------------------------------------------------------------\n')
kav3_2.grid(row = 9, column = 6)

followers_2 = Button(r, text='MAX \nLIKES', width=15,height=5, pady=10, padx=10, fg='red', bg='yellow', command=twint_likes)
followers_2.grid(row = 10, column = 5)

retweets_2 = Button(r, text='MAX \nRETWEETS', width=15,height=5, pady=10, padx=10, fg='red', bg='yellow', command=twint_retweets)
retweets_2.grid(row = 10, column = 6)

tweets_2 = Button(r, text='MAX \nREPEATED \nTWEETS', width=15,height=5, pady=10, padx=10, fg='red', bg='yellow', command=twint_tweets)
tweets_2.grid(row = 10, column = 7)

kav4_2 = Label(r, text='\n------------------------------' + 
                 '--------------------------------------------------------------\n\n')
kav4_2.grid(row = 11, column = 6)


r.mainloop()

