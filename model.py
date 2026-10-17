import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data():
    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "dataset" / "zomato.csv"
    zip_path = project_dir / "dataset.zip"

    if csv_path.is_file():
        zomato_real = pd.read_csv(
            csv_path, encoding='utf-8', engine='python', on_bad_lines='skip'
        )
    elif zip_path.is_file():
        with ZipFile(zip_path) as archive:
            with archive.open("dataset/zomato.csv") as dataset_file:
                zomato_real = pd.read_csv(
                    dataset_file, encoding='utf-8', engine='python', on_bad_lines='skip'
                )
    else:
        raise FileNotFoundError(
            f"Restaurant data not found. Add either '{csv_path}' or '{zip_path}' to the project."
        )

    # Data Cleaning
    zomato = zomato_real.drop(
        ['url', 'dish_liked', 'phone', 'address', 'rest_type', 'menu_item', 'votes'],
        axis=1,
        errors='ignore',
    )
    zomato = zomato.drop_duplicates().dropna(how='any').reset_index(drop=True)

    # Rename columns
    zomato = zomato.rename(columns={'approx_cost(for two people)':'cost', 'listed_in(type)':'type', 'listed_in(city)':'city'})

    # Cost & Rate Transformations
    zomato['cost'] = zomato['cost'].astype(str).str.replace(',', '.').astype(float)
    zomato = zomato.loc[zomato.rate != 'NEW']
    zomato = zomato.loc[zomato.rate != '-'].reset_index(drop=True)
    remove_slash = lambda x: x.replace('/5', '') if isinstance(x, str) else x
    zomato['rate'] = zomato['rate'].apply(remove_slash).str.strip().astype('float')

    zomato['name'] = zomato['name'].apply(lambda x: x.title())
    zomato['online_order'] = zomato['online_order'].replace({'Yes': True, 'No': False})
    zomato['book_table'] = zomato['book_table'].replace({'Yes': True, 'No': False})

    # Computing Mean Rating
    restaurants = list(zomato['name'].unique())
    zomato['Mean Rating'] = 0.0
    for restaurant in restaurants:
        zomato.loc[zomato['name'] == restaurant, 'Mean Rating'] = zomato.loc[zomato['name'] == restaurant, 'rate'].mean()

    scaler = MinMaxScaler(feature_range=(1,5))
    zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

    # Text Preprocessing on Reviews
    zomato["reviews_list"] = zomato["reviews_list"].fillna('').astype(str).str.lower()
    PUNCT_TO_REMOVE = string.punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))
    
    # Sample 10% of data to optimize performance for cloud hosting
    df_percent = zomato.sample(frac=0.1, random_state=42)
    df_percent.set_index('name', inplace=True)
    
    # TF-IDF and Cosine Similarity Matrix
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
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
        
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    for each in recommend_restaurant:
        match = df_percent[['cuisines', 'Mean Rating', 'cost']][df_percent.index == each]
        if not match.empty:
            df_new = pd.concat([df_new, match.sample()])
            
    df_new = df_new.drop_duplicates(subset=['cuisines', 'Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    return df_new