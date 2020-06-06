#%%
import pandas as pd
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# 讀檔
stopword = []
with open('data/stopword.txt', 'r', encoding = 'utf8') as file:
    for data in file.readlines():
        data = data.strip()
        stopword.append(data)

forum = pd.read_csv("data/forum.csv", encoding="utf-8")
bbs = pd.read_csv("data/bbs.csv", encoding="utf-8")
news = pd.read_csv("data/news.csv", encoding="utf-8")
#%%
# 將na代成'n'
news.fillna('n', inplace=True)
# 將title, content合併到all_content裡面
news['all_content'] = news[['title', 'content']].sum(1)
news = news[['id', 'post_time', 'all_content']]
news.columns = ['id', 'time', 'content']
# 把日期轉成 Datetime
news.time = [datetime.strptime(news.time[i], '%Y/%m/%d %H:%M:%S').date() for i in range(len(news))]
#%%
def get_26grams(article_df):
    """斷詞(2-6gram)"""
    tf = {}
    df = {}
    tfdf = {}
    article_df.content.replace(to_replace = r"[a-zA-Z0-9\W\d]", value = '', regex = True, inplace = True)
    for row in range(len(article_df)):
        #text = article_df.iloc[row].values.sum(axis = 0)
        text = article_df.iloc[row, 2]
        for n in range(2, 7): #斷 2-6 grams            
            tokens = [text[i:i+n] for i in range(0, len(text)-n+1)]
            for token in set(tokens):
                if token not in df.keys():
                    df[token] = 1
                else:    
                    df[token] += 1
            for token in tokens:
                if token not in tf.keys():
                    tf[token] = 1
                else:
                    tf[token] += 1
    for key, value in tf.items():
        tfdf[key] = [value, df[key]]
    dataframe = pd.DataFrame.from_dict(tfdf, orient = 'index', columns = ['tf','df'])
    return(dataframe)
#%%
def remove_sw_lowtf_0001(df, N):
    """為了減少運算需要的時間，移除：
            1. DF < 篇數(N) 0.1% 的詞
            2. stopword """
    # 移除 DF< 篇數(N) 0.1% 的詞
    low_limit = 0.001 * N
    df = df.sort_values(by = 'df') 
    for i in range(len(df)):
        if df.iloc[i, 1] > low_limit:
            df = df[i:]
            break
    # 移除 stopword
    lowtf_drop = set()
    for i in range(len(df)):
        if df.index[i] in stopword:
            lowtf_drop.add(df.index[i])
    df = df.drop(lowtf_drop)
    return df
# %%
def get_tfidf(df, docs):
    """add the tf-idf column"""
    df['tfidf'] = (1+np.log(df.tf))*np.log(docs/df.df)
    return(df)
# %%
def remove_same(dataframe):
    """移除DF相差20%內的 被較長詞包含的詞""" 
    """input dataframe w/ 3 columns: tf/df/tfidf """
    dataframe['wlen'] = dataframe.index.str.len()
    same_drop = set()
    for i in range(len(dataframe)): # 一行一行(i)檢查
        # 算出跟短字df相差1%內的 df範圍
        upper_limit = dataframe.iloc[i, 1] * 1.2
        lower_limit = dataframe.iloc[i, 1] * 0.8
        short_len = dataframe.iloc[i, 3]   
        # 只跟down中，字詞比它長 ＆ df在範圍內的字比較
        condi_df = dataframe.loc[(dataframe.wlen > short_len) & (dataframe.df <= upper_limit) & (dataframe.df >= lower_limit)].copy()
        # 檢查是否被包含
        short_word = dataframe.index[i]
        for j in range(len(condi_df)):
            if short_word in condi_df.index[j]:
                same_drop.add(short_word)
                break
    dataframe.drop('wlen', axis=1, inplace=True)
    return dataframe.drop(same_drop)
# %%
# RUN ALL(article to keyword dataframe)
def get_keyword(content):
    """content --> key words dataframe, 切2-7gram"""
    all_gram = get_26grams(content)
    all_gram_sw = remove_sw_lowtf_0001(all_gram, len(content))
    all_allgram_tfidf = get_tfidf(all_gram_sw, len(content))
    all_allgram_same = remove_same(all_allgram_tfidf)
    return(all_allgram_same)
# %%
# 判斷股價是漲還是跌
stock_data_2016 = pd.read_excel("data/stock_data.xlsx", sheet_name = "上市2016")
stock_data_2017 = pd.read_excel("data/stock_data.xlsx", sheet_name = "上市2017")
stock_data_2018 = pd.read_excel("data/stock_data.xlsx", sheet_name = "上市2018")
stock_data = pd.concat([stock_data_2016, stock_data_2017, stock_data_2018])
y9999 = stock_data.loc[stock_data['證券代碼'] == 'Y9999 加權指數']
y9999 = y9999.loc[:,['年月日', '收盤價(元)']].reset_index(drop = True)
y9999.columns = ['time','price']
#%%
def get_updown_article(stock_price, all_article = news):
    """將文章分類成上漲與下跌兩類，分別存成dataframe(基準為相較20日的加權移動平均線上漲或下跌)"""
    stock_price.sort_values(by = 'time', inplace = True) # 依時間先到後排序
    stock_price.reset_index(drop = True, inplace = True)
    delta = timedelta(days=2)
    weights = [i for i in range(1, 21)]
    for i in range(20, len(stock_price)):
        elements = [stock_price.iloc[j, 1] for j in range(i-20, i)]
        stock_price.loc[i, 'moving_average_20'] = np.average(elements, weights=weights)
        stock_price.loc[i, 'diff'] = stock_price.price[i] - stock_price.moving_average_20[i] # 計算當日價格與移動平均差
    for i in range(21, len(stock_price)):
        # 價格與移動平均差兩期正負不同時（一期較大一期較小）則代表有漲或跌的狀況（交叉）
        if stock_price.loc[i, 'diff'] * stock_price.loc[i-1, 'diff'] < 0: 
            if stock_price.loc[i, 'diff'] > 0: # 漲
                stock_price.loc[i, 'up'] = 1 
            else: # 跌 
                stock_price.loc[i, 'up'] = 0 
    # 漲跌日期獨立取出至兩個array
    up = stock_price.loc[stock_price.up == 1, 'time'].array
    down = stock_price.loc[stock_price.up == 0, 'time'].array
    # 將日期轉成datetime格式
    up = [datetime.strptime(up[i], '%Y-%m-%d').date() for i in range(len(up))]
    down = [datetime.strptime(down[i], '%Y-%m-%d').date() for i in range(len(down))]
    # 全部日期減一（因為要取出上漲或下跌前一天的文章）
    up = [up[i]-delta for i in range(len(up))]
    down = [down[i]-delta for i in range(len(down))]
    # 取出指定日期的文章
    up_article = all_article.loc[[all_article.loc[i, 'time'] in up for i in range(len(all_article))]]
    down_article = all_article.loc[[all_article.loc[i, 'time'] in down for i in range(len(all_article))]]
    up_article.reset_index(drop = True, inplace = True)
    down_article.reset_index(drop = True, inplace = True)
    return(up_article, down_article)

y9_up_article, y9_down_article = get_updown_article(y9999)
#%%
# 從已選出的上漲及下跌中，再挑出 forum & bbs的討論數多的那幾天為
# 挑選基準: > mean + std
discuss = forum.append(bbs)
discuss.post_time = discuss.post_time.astype('datetime64[ns]').apply(lambda x: x.date())
n_discuss = discuss.groupby('post_time').count().sort_values('id', ascending=False)
dates = n_discuss[n_discuss.id>(np.mean(n_discuss.id)+np.std(n_discuss.id))]
date_list = dates.index.tolist()
   
y9_up_high = y9_up_article[y9_up_article.time.isin(date_list)]
y9_down_high = y9_down_article[y9_down_article.time.isin(date_list)]
# %%
up_docs = len(y9_up_high)
down_docs = len(y9_down_high)
all_docs = len(news)

# 取得全部、上漲、下跌三個關鍵字列表
#all_allgram = get_keyword(all_article_df.content)
up_rm_same = get_keyword(y9_up_high)
down_rm_same = get_keyword(y9_down_high)
# 依照tf-idf 大小排列
up_rm_same.sort_values(by='tfidf', ascending=False, inplace=True)
down_rm_same.sort_values(by='tfidf', ascending=False, inplace=True)

# %%
# 取出up/down 各前2000個詞，並刪掉同時出現在上漲及下跌的詞
up_2000_keyword = up_rm_same.iloc[:2000,:]
down_2000_keyword = down_rm_same.iloc[:2000,:]
def del_both_word(up_df, down_df):
    """刪除up/down兩邊都有的keyword"""
    to_drop = set()
    for i in range(len(up_df)): # 以up為基礎一行一行比較
        if sum(down_df.index == up_df.index[i]) != 0: # down裡面有相同的詞
            to_drop.add(up_df.index[i])
    return (up_df.drop(to_drop), down_df.drop(to_drop))
up_keyword, down_keyword = del_both_word(up_2000_keyword, down_2000_keyword)

#  三種移除outlier的可能選項
def outlier_IQR(data):
    """用 3IQR 篩"""
    #data.sort_values(by = 'df', inplace = True)
    # 篩df
    df_Q1, df_Q3 = np.percentile(data.df.values, [25, 75])
    df_IQR = df_Q3 - df_Q1
    df_upper_limit = df_Q3 + (3 * df_IQR) 
    # 篩tf
    tf_Q1, tf_Q3 = np.percentile(data.tf.values, [25, 75])
    tf_IQR = tf_Q3 - tf_Q1
    tf_upper_limit = tf_Q3 + (3 * tf_IQR) 
    return data[(data.df < df_upper_limit) & (data.tf < tf_upper_limit)]

def outlier_sd(data):
    """用平均加減三倍標準差篩"""
    df_upper_limit = data.df.values.mean() + (data.df.values.std() * 3)
    tf_upper_limit = data.tf.values.mean() + (data.tf.values.std() * 3)
    return data[(data.df < df_upper_limit) & (data.tf < tf_upper_limit)]

def outlier_0001(df):
    """移除上限0.1%的字詞"""
    # 移除 TF 分佈前 0.1% 的詞
    upper_limit_tf = np.percentile(df.tf,99.9)
    df = df.sort_values(by = 'tf', ascending=False) 
    for i in range(len(df)):
        if df.iloc[i, 0] < upper_limit_tf:
            df = df[i:]
            break
    # 移除 DF 分佈前 0.1% 的詞
    upper_limit_df = np.percentile(df.df,99.9)
    df = df.sort_values(by = 'df', ascending=False) 
    for i in range(len(df)):
        if df.iloc[i, 1] < upper_limit_df:
            df = df[i:]
            break
    return df
# %%
# 各文章斷詞處理
def get_article_gram(articles):
    """only get tf as a two-demesion dictionary"""
    #content = articles.content
    # 移除英文及數字
    articles.content.replace(to_replace = r"[a-zA-Z0-9\W\d\uFF41-\uFF5A\uFF21-\uFF3A]", value = '', 
                            regex = True, inplace = True)
    all_tf = {} # two-dimension dictionary {id:{word:tf,...}, id:{word:tf,...}...}
    for row in range(len(articles)):
        per_tf = {} # dictionary of each article
        if pd.notnull(articles.iloc[row,2]):
            text = articles.iloc[row,2]
            for n in range(2, 7): #斷 2-6 grams
                tokens = [text[i:i+n] for i in range(0, len(text)-1)]
                for token in tokens:
                    if token not in stopword:
                        if token not in per_tf.keys():
                            per_tf[token] = 1
                        else:
                            per_tf[token] += 1
        all_tf[articles.iloc[row, 0]] = per_tf
    return(all_tf)
def count_matrix(articles, keyword):
    """from articles(df) to a count matrix"""
    data_tf_dict = get_article_gram(articles)
    # 選出的重要關鍵字放在column name
    keyword_tf_df = pd.DataFrame(columns=keyword)
    # 文章中出現關鍵字的次數放入dataframe
    for key, value in data_tf_dict.items():
        keyword_per_dict = {} # 提出每篇文章的重要關鍵字數量
        for col in range(len(keyword_tf_df.columns)):
            if keyword_tf_df.columns[col] in value.keys(): # 重要關鍵字有在此篇文章出現
                keyword_per_dict[keyword_tf_df.columns[col]] = value[keyword_tf_df.columns[col]]
            else: # 重要關鍵字未在此篇文章出現
                keyword_per_dict[keyword_tf_df.columns[col]] = 0
        keyword_tf_df = keyword_tf_df.append(keyword_per_dict, ignore_index = True) # 加入dataframe
    keyword_tf_df.index = data_tf_dict.keys() # 文章的id 作為index
    return(keyword_tf_df)

#%%
##1 IQR
up_keyword_iqr = outlier_IQR(up_keyword)
down_keyword_iqr = outlier_IQR(down_keyword)
# 漲和跌各自的前100個關鍵字(by tfidf)，並組合成一個array
up_100_word_iqr = up_keyword_iqr.index[:100].values
down_100_word_iqr = down_keyword_iqr.index[:100].values
updown_200_word_iqr = np.concatenate([up_100_word_iqr, down_100_word_iqr], axis=0)
# 製作up/down各自的count matrix，並新增一欄'up'，其中1代表上漲，0代表下跌
up_count_matrix_iqr = count_matrix(y9_up_high, updown_200_word_iqr)
up_count_matrix_iqr['up'] = 1
down_count_matrix_iqr = count_matrix(y9_down_high, updown_200_word_iqr)
down_count_matrix_iqr['up'] = 0
# 合成最後的count matrix
all_count_matrix_iqr = pd.concat([up_count_matrix_iqr, down_count_matrix_iqr])
all_count_matrix_iqr.to_csv('count_matrix_iqr_y9nobbs2.csv', encoding='utf_8_sig')

##2 SD
up_keyword_sd = outlier_sd(up_keyword)
down_keyword_sd = outlier_sd(down_keyword)
# 漲和跌各自的前100個關鍵字(by tfidf)，並組合成一個array
up_100_word_sd = up_keyword_sd.index[:100].values
down_100_word_sd = down_keyword_sd.index[:100].values
updown_200_word_sd = np.concatenate([up_100_word_sd, down_100_word_sd], axis=0)
# 製作up/down各自的count matrix，並新增一欄'up'，其中1代表上漲，0代表下跌
up_count_matrix_sd = count_matrix(y9_up_high, updown_200_word_sd)
up_count_matrix_sd['up'] = 1
down_count_matrix_sd = count_matrix(y9_down_high, updown_200_word_sd)
down_count_matrix_sd['up'] = 0
# 合成最後的count matrix
all_count_matrix_sd = pd.concat([up_count_matrix_sd, down_count_matrix_sd])
all_count_matrix_sd.to_csv('count_matrix_sd_y9nobbs2.csv', encoding='utf_8_sig')

##3 0.1%
up_keyword_0001 = outlier_0001(up_keyword)
down_keyword_0001 = outlier_0001(down_keyword)
# 漲和跌各自的前100個關鍵字(by tfidf)，並組合成一個array
up_100_word_0001 = up_keyword_0001.index[:100].values
down_100_word_0001 = down_keyword_0001.index[:100].values
updown_200_word_0001 = np.concatenate([up_100_word_0001, down_100_word_0001], axis=0)
# 製作up/down各自的count matrix，並新增一欄'up'，其中1代表上漲，0代表下跌
up_count_matrix_0001 = count_matrix(y9_up_high, updown_200_word_0001)
up_count_matrix_0001['up'] = 1
down_count_matrix_0001 = count_matrix(y9_down_high, updown_200_word_0001)
down_count_matrix_0001['up'] = 0
# 合成最後的count matrix
all_count_matrix_0001 = pd.concat([up_count_matrix_0001, down_count_matrix_0001])
all_count_matrix_0001.to_csv('count_matrix_0001_y9nobbs2.csv', encoding='utf_8_sig')