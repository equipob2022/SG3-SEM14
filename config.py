import pandas as pd
import asyncio
import twint
import os

def load_tweet(keyword, limit):
  c = twint.Config()
  c.Search = keyword
  c.Limit = limit
  c.Lang = 'en'
  c.Store_csv = True
  #por numero de likes
  c.Min_likes = 20
  #por numero de retweets
  c.Min_retweets = 20
  #guardar los tweets en un dataframe
  #twets populares
  c.Popular_tweets = True
  c.Pandas = True

  asyncio.set_event_loop(asyncio.new_event_loop())
  twint.run.Search(c)
  df = twint.storage.panda.Tweets_df
  return df