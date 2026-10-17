import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data():
    zomato_real = pd.read_csv("dataset/zomato.csv")
    
    zomato = zomato_real.drop(['url', 'dish_liked', 'phone', 'address', 'rest_type', 'type', 'menu_item'], axis=1)
    zomato.drop_duplicates(inplace=True)
    zomato.dropna(how='any', inplace=True)
    
    zomato = zomato.rename(columns={'approx_cost(for two people)':'cost', 'listed_in(type)':'type', 'listed_in(city)':'city'})
    
    zomato['cost'] = zomato['cost'].astype(str).apply(lambda x: x.replace(',','.')).astype(float)
    zomato = zomato.loc[zomato.rate !='NEW']
    zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
    remove_slash = lambda x: x.replace('/5', '') if type(x) == str else x
    zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')
    
    zomato.name = zomato.name.apply(lambda x: x.title())
    zomato.online_order.replace(('Yes','No'), (True, False), inplace=True)
    zomato.book_table.replace(('Yes','No'), (True, False), inplace=True)
    
    restaurants = list(zomato['name'].unique())
    zomato['Mean Rating'] = 0
    for i in range(len(restaurants)):
        zomato.loc[zomato['name'] == restaurants[i], 'Mean Rating'] = zomato.loc[zomato['name'] == restaurants[i], 'rate'].mean()
        
    scaler = MinMaxScaler(feature_range=(1,5))
    zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)
    
    zomato["reviews_list"] = zomato["reviews_list"].str.lower()
    PUNCT_TO_REMOVE = string.punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))
    
    df_percent = zomato.sample(frac=0.1, random_state=42)
    df_percent.set_index('name', inplace=True)
    
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return df_percent, cosine_similarities

def recommend_restaurants(name, df_percent, cosine_similarities):
    indices = pd.Series(df_percent.index)
    if name not in df_percent.index:
        return None
    
    recommend_restaurant = []
    idx = indices[indices == name].index[0]
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    top30_indexes = list(score_series.iloc[0:31].index)
    
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
        
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost', 'city', 'rate', 'votes'])
    for each in recommend_restaurant:
        match = df_percent[['cuisines', 'Mean Rating', 'cost', 'city', 'rate', 'votes']][df_percent.index == each]
        if not match.empty:
            df_new = pd.concat([df_new, match.sample()])
            
    df_new = df_new.drop_duplicates(subset=['cuisines', 'Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    return df_new