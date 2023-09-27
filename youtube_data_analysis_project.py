#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


comments=pd.read_csv(r"C:\Users\User\Downloads\UScomments.csv", error_bad_lines=False)


# In[3]:


comments.head()


# In[4]:


comments.isnull().sum()


# In[5]:


comments.dropna(inplace=True)


# In[6]:


comments.isnull().sum()


# In[7]:


get_ipython().system('pip install textblob')


# In[8]:


from textblob import TextBlob


# In[9]:


comments.head()


# In[10]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity


# In[11]:


polarity=[]
for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[12]:


len(polarity)


# In[13]:


comments['polarity'] = polarity


# In[14]:


comments.head(5)


# In[15]:


filter1=comments['polarity']==1


# In[16]:


comments_positive=comments[filter1]


# In[17]:


comments[filter1]


# In[18]:


filter2=comments['polarity']==-1


# In[19]:


comments_negative=comments[filter2]


# In[ ]:





# In[ ]:





# In[20]:


comments_positive.head(5)


# In[21]:


get_ipython().system('pip install wordcloud')


# In[22]:


from wordcloud import WordCloud,STOPWORDS


# In[23]:


set(STOPWORDS)


# In[24]:


comments['comment_text']


# In[25]:


type(comments['comment_text'])


# In[26]:


total_comments_positive=''.join(comments_positive['comment_text'])


# In[27]:


wordcloud=WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_positive)


# In[28]:


plt.imshow(wordcloud)
plt.axis('off')


# In[29]:


total_comments_negative=''.join(comments_negative['comment_text'])


# In[30]:


wordcloud2=WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_negative)


# In[31]:


plt.imshow(wordcloud2)
plt.axis('off')


# In[ ]:


# emoji analysis


# In[32]:


get_ipython().system('pip install emoji==2.2.0')


# In[33]:


import emoji


# In[34]:


emoji.__version__


# In[35]:


comments['comment_text'].head(6)


# In[36]:


comment='trending 😉'


# In[37]:


[char for char in comment if char in emoji.EMOJI_DATA]


# In[38]:


all_emoji_list=[]
for comment in comments['comment_text'].dropna():
    for char in comment:
        if char in emoji.EMOJI_DATA:
            all_emoji_list.append(char)


# In[39]:


all_emoji_list[0:10]


# In[40]:


from collections import Counter


# In[41]:


Counter(all_emoji_list).most_common(10)


# In[42]:


Counter(all_emoji_list).most_common(10)[0]


# In[43]:


Counter(all_emoji_list).most_common(10)[2][1]


# In[44]:


emojis=[Counter(all_emoji_list).most_common(10)[i][0] for i in range(10)]


# In[45]:


emojis


# In[46]:


freqs=[Counter(all_emoji_list).most_common(10)[i][1] for i in range(10)]


# In[47]:


freqs


# In[48]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[49]:


trace=go.Bar(x=emojis,y=freqs)


# In[50]:


iplot([trace])


# In[51]:


import os


# In[55]:


files=os.listdir(r"C:\Users\User\Downloads\additional_data-20230927T055359Z-001\additional_data")


# In[56]:


files


# In[57]:


files_csv=[file for file in files if '.csv' in file]


# In[58]:


files_csv


# In[59]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[62]:


full_df=pd.DataFrame()
path=r"C:\Users\User\Downloads\additional_data-20230927T055359Z-001\additional_data"

for file in files_csv:
    current_df=pd.read_csv(path+'/'+ file,encoding='iso-8859-1' ,error_bad_lines=False)
    full_df=pd.concat([full_df,current_df], ignore_index=True)
    


# In[63]:


full_df.shape


# In[64]:


#6.How to export your data into  csv,json,db


# In[ ]:





# In[ ]:





# In[67]:


full_df[full_df.duplicated()].shape


# In[68]:


full_df=full_df.drop_duplicates()


# In[69]:


full_df.shape


# In[73]:


full_df[0:1000].to_csv(r'C:\Users\User\Downloads\additional_data-20230927T055359Z-001/youtube_sample.csv',index=False)


# In[74]:


full_df[0:1000].to_json(r'C:\Users\User\Downloads\additional_data-20230927T055359Z-001/youtube_sample.json')


# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


from sqlalchemy import create_engine


# In[77]:


engine=create_engine(r'sqlite:///C:\Users\User\Downloads\additional_data-20230927T055359Z-001/youtube_sample.sqlite')


# In[78]:


full_df[0:1000].to_sql('Users', con=engine, if_exists='append')


# In[79]:


##which category has highest likes


# In[80]:


full_df['category_id'].unique()


# In[81]:


json_df=pd.read_json(r"C:\Users\User\Downloads\additional_data-20230927T055359Z-001\additional_data/US_category_id.json")


# In[82]:


json_df


# In[83]:


json_df['items'][0]


# In[84]:


json_df['items'][1]


# In[87]:


cat_dict={}
for item in json_df['items'].values:
    cat_dict[int(item['id'])]=item['snippet']['title']


# In[89]:


cat_dict


# In[91]:


full_df['category_name']=full_df['category_id'].map(cat_dict)


# In[92]:


full_df.head(4)


# In[95]:


plt.figure(figsize=(12,8))
sns.boxplot(x='category_name', y='likes',data=full_df)
plt.xticks(rotation='vertical')


# In[96]:


##find out whether audience is engaged or not


# In[101]:


full_df['like_rate']=(full_df['likes']/full_df['views'])*100
full_df['dislike_rate']=(full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate']=(full_df['comment_count']/full_df['views'])*100


# In[103]:


full_df.columns


# In[105]:


plt.figure(figsize=(8,6))
sns.boxplot(x='category_name', y='like_rate',data=full_df)
plt.xticks(rotation='vertical')
plt.show()


# In[106]:


sns.regplot(x='views',y='likes',data=full_df)


# In[107]:


full_df.columns


# In[108]:


full_df[['views', 'likes', 'dislikes']].corr()


# In[109]:


sns.heatmap(full_df[['views', 'likes', 'dislikes']].corr())


# In[110]:


#which are the channels which have largest trending videos


# In[111]:


full_df.head(6)


# In[112]:


full_df['channel_title'].value_counts()


# In[114]:


cdf=full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()


# In[120]:


cdf=cdf.rename(columns={0:'total_videos'})


# In[121]:


import plotly.express as px


# In[122]:


px.bar(data_frame=cdf[0:20],x='channel_title' , y='total_videos')


# In[139]:


#does puctuations in title and tags have any relation with views, likes, dislikes


# In[124]:


full_df['title'][0]


# In[125]:


import string


# In[128]:


string.punctuation


# In[144]:


len([ char for char in full_df['title'][0] if  char in string.punctuation])


# In[145]:


def punc_count(text):
    return len([ char for char in text if  char in string.punctuation])


# In[146]:


sample=full_df[0:10000]


# In[148]:


sample['count_punc']=sample['title'].apply(punc_count)


# In[149]:


sample['count_punc']


# In[ ]:





# In[ ]:





# In[150]:


full_df['title'].apply(punc_count)


# In[151]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc', y='views',data=sample)

plt.show()


# In[152]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc', y='likes',data=sample)

plt.show()


# In[ ]:




