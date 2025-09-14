import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.express as px
import math

import findspark
findspark.init()

from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from gensim import corpora, models, similarities
from pyvi.ViTokenizer import tokenize
from collections import Counter

file_stopword = 'vietnamese-stopwords.txt'
file_info = 'hotel_info.csv'
file_comments = 'hotel_comments.csv'

with open(file_stopword, 'r', encoding='utf-8') as file:
    stop_words = file.read()
stop_words = stop_words.split('\n')

spark = SparkSession.builder.appName('DL07_K306_ONLINE_LyLaoViXuong_Project2_GUI').config('spark.driver.memory', '512m').getOrCreate()

df_info = pd.read_csv(file_info)

df_comments = pd.read_csv(file_comments)

df_comments = pd.read_csv('hotel_comments.csv', encoding='utf-8-sig', header=0)

def build_gensim(df):
  symbols_ = ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '(', ')', '+', '/', "'", '&']

  df['Tokenised'] = df['Content'].apply(lambda x: tokenize(x))

  # split into words
  content_gem_ = df['Tokenised'].apply(lambda x: x.split()).tolist()

  # preprocess
  content_gem_ = [[re.sub(r'[0-9]+','', e) for e in text] for text in content_gem_]
  content_gem_ = [[t.lower() for t in text if not t in symbols_] for text in content_gem_]
  content_gem_ = [[t for t in text if not t in stop_words] for text in content_gem_]
  content_gem_ = [doc for doc in content_gem_ if len(doc) > 0]

  # create dictionary, get number of features
  dict_ = corpora.Dictionary(content_gem_)
  num_features_ = len(dict_.token2id)

  # get corpus based on dictionary
  corpus_ = [dict_.doc2bow(text) for text in content_gem_]

  # use TF-IDF model to process corpus, get index
  tfidf_ = models.TfidfModel(corpus_)
  index_ = similarities.SparseMatrixSimilarity(tfidf_[corpus_], num_features=num_features_)

  df_index_ = pd.DataFrame(index_)

  return dict_, tfidf_, index_, df_index_

def search_query_gensim(query, dict, tfidf, index, df, nums=5):
  symbols_ = ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '(', ')', '+', '/', "'", '&']

  tokens_ = tokenize(query).split()
  tokens_ = [re.sub(r'[0-9]+', '', tok) for tok in tokens_]
  tokens_ = [t.lower() for t in tokens_ if t not in symbols_]
  tokens_ = [t for t in tokens_ if t not in stop_words]

  # compute Sparse Vectors
  kw_vector_ = dict.doc2bow(tokens_)

  # get similarities
  sim_ = index[tfidf[kw_vector_]]

  # DataFrame for similiarities
  df_sim_ = pd.DataFrame({
    'id': range(len(sim_)),
    'sim': sim_
  })

  # sort by descending order
  df_sorted_ = df_sim_.sort_values(by='sim', ascending=False)

  # create recommendations
  recommend_ = df_sorted_.head(nums).id.to_list()

  # returnf df list filtered with recommendations
  return df.iloc[recommend_]

def search_query_cossim(query, vectoriser, tfidf, df, nums=5):
  vec_ = vectoriser.transform([query])

  sim_ = cosine_similarity(vec_, tfidf)

  df_sim_ = pd.DataFrame({
    'id': range(len(sim_[0])),
    'sim': sim_[0]
  })

  df_sorted_ = df_sim_.sort_values(by='sim', ascending=False)

  recommend_ = df_sorted_.head(nums).id.to_list()

  res_ = df.iloc[recommend_].copy()
  res_['similarity'] = df_sorted_.head(nums)['sim'].values
  
  return res_   

def recommend_other_hotels(hotel, df, cosine_sim):
  hotel_id_ = hotel['Hotel_ID']

  df_ = df.reset_index(drop=True)
  
  if hotel_id_ not in df_['Hotel_ID'].values:
    return pd.DataFrame()

  index_ = df_[df_['Hotel_ID'] == hotel_id_].index[0]
  if index_ >= len(cosine_sim):
    return pd.DataFrame()

  sim_scores_ = list(enumerate(cosine_sim[index_]))
  sim_scores_ = sorted(sim_scores_, key=(lambda x: x[1]), reverse=True)[1:6]
  indices_ = [idx_[0] for idx_ in sim_scores_]
  
  return df.iloc[indices_]

@st.dialog(title='Thông Tin Khách Sạn', width='large')
def display_hotel_info(hotel, df_info, cosine_sim):
  hotel_comments_ = df_comments[df_comments['Hotel ID'] == hotel['Hotel_ID']]

  st.markdown(f'''
    # {hotel['Hotel_Name']}
    \b
  ''')

  # General Info
  with st.container(border=True):
    st.markdown(f'''
      ## :green[Thông Tin Chung]
      ## :blue[Địa chỉ:] {hotel['Hotel_Address']}
      ## :blue[Hạng Sao:] {hotel['Hotel_Rank']}
      ## :blue[Tổng điểm trung bình:] {hotel['Total_Score']} điểm
    ''')
  
  # Description
  with st.container(border=True, height=350):
    st.markdown(f'''
      ## :green[Mô Tả]
      {hotel['Hotel_Description']}
    ''')

  # Key Aspects
  with st.container(border=True):
    st.markdown(f'''
      ## :green[Điểm Chi Tiết]
    ''')

    scores_row = st.columns(5)
    score_cols = ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']

    for index_, col_ in enumerate(scores_row): 
      with col_.container(border=True):
        st.markdown(f'''
          ### {score_cols[index_]}

          # {hotel[score_cols[index_]]} ({str(hotel[f'{score_cols[index_]}_class']).upper()})
        ''')

  # Top comments
  with st.container(border=True):
    st.markdown(f'''
      ## :green[Đánh Giá]
    ''')

    top_comments_ = hotel_comments_.sort_values(by='Score', ascending=False).head(3)

    for index_, row_ in top_comments_.iterrows():
      with st.container(border=True):
        st.markdown(f'''
          ## {row_['Reviewer Name']} ({row_['Group Name']}) :yellow[{row_['Score']} điểm] {row_['Review Date']}
        ''')

        clean_title_ = re.sub(r'["“”]', '', row_['Title'])

        st.markdown(f"<h4>{clean_title_}</h4>", unsafe_allow_html=True)
        st.text_area(key=f'comment_textarea_{index_}', label='comment', label_visibility='collapsed', value=row_['Body'], disabled=False)

  # Other recommended hotels
  with st.container(border=True):
    st.markdown(f'''
      ## :green[Các Khách Sạn Khác Bạn Cũng Có Thể Quan Tâm]
    ''')

    other_hotels_ = recommend_other_hotels(hotel, df_info, cosine_sim)
    hotel_names_ = other_hotels_['Hotel_Name'].to_list()

    for index_, name_ in enumerate(hotel_names_):
      with st.container(border=True):
        col1_, col2_ = st.columns([10, 1])

        with col1_:
            hotel_ = other_hotels_.iloc[index_]
            st.markdown(f"""
                <div style='line-height:1.5'>
                    <span style='font-size:20px; font-weight:bold; color:#ffffff'>{hotel_['Hotel_Name']}</span><br>
                    <span style='font-size:18px; color:#27ae60;'>Tổng điểm: {hotel_['Total_Score']}</span> &nbsp;&nbsp;&nbsp;&nbsp;
                    <span style='font-size:18px; color:#2e86c1;'>⭐ {hotel_['Hotel_Rank']}</span>
                </div>
            """, unsafe_allow_html=True)

        with col2_:
            if st.button("🔍", key=f"select_{index_}", width='stretch'):
                st.session_state.selected_hotel = hotel_
                st.session_state.show_dialog = True
                st.rerun()

def display_recommend_results(df_search_result, df_info, cosine_sim):
  st.markdown('''
    ### Kết Quả Tìm Kiếm
  ''')

  if df_search_result.empty:
    return

  hotel_names_ = df_search_result['Hotel_Name'].to_list()

  for index_, name_ in enumerate(hotel_names_):
    with st.container(border=True):
      col1_, col2_ = st.columns([10, 1])

      with col1_:
          hotel_ = df_search_result.iloc[index_]
          st.markdown(f"""
              <div style='line-height:1.5'>
                  <span style='font-size:20px; font-weight:bold; color:#ffffff'>{hotel_['Hotel_Name']}</span><br>
                  <span style='font-size:18px; color:#27ae60;'>Tổng điểm: {hotel_['Total_Score']}</span> &nbsp;&nbsp;&nbsp;&nbsp;
                  <span style='font-size:18px; color:#2e86c1;'>⭐ {hotel_['Hotel_Rank']}</span>
              </div>
          """, unsafe_allow_html=True)

      with col2_:
          if st.button("🔍", key=f"search_select_{index_}", width='stretch'):
              st.session_state.selected_hotel = df_search_result.iloc[index_]
              st.session_state.show_dialog = True
              st.rerun()

def extract_review_date(text):
    match_ = re.search(r'(\d{1,2})\s+tháng\s+(\d{1,2})\s+(?:năm\s+)?(\d{4})', text)

    if match_:
        day_, month_, year_ = match_.groups()
        return pd.to_datetime(f'{year_}-{month_}-{day_}', dayfirst=True, errors='coerce')

    return pd.NaT

def preprocess_text(text):
  symbols_ = ['!', '?', '.', ',', ':', ';', '(', ')', '+', '/', '"', '&', '%', '-', '_', '[', ']', '“', '”', "'"]

  tokenised_ = text.apply(lambda x: tokenize(x.lower()))
  tokenised_ = tokenised_.apply(lambda x: re.sub(r'\d+', '', x))

  tokens_ = tokenised_.apply(lambda x: x.split())
  tokens_ = tokens_.apply(lambda ls: [w for w in ls if w not in symbols_])
  tokens_ = tokens_.apply(lambda ls: [w for w in ls if w not in stop_words])
  tokens_ = tokens_.apply(lambda ls: [w for w in ls if len(w) > 2])

  res_ = tokens_.apply(lambda ls: ' '.join(ls))
  return res_

# ======================== User Interface ======================== #

def page_hotel_search() -> st.Page:
  st.markdown('''
    # TÌM KIẾM KHÁCH SẠN
    \b              
  ''')

  if 'df_hotel_search_results' not in st.session_state:
    st.session_state.df_hotel_search_results = pd.DataFrame()

  if 'selected_hotel' not in st.session_state:
    st.session_state.selected_hotel = pd.DataFrame()

  if 'show_dialog' not in st.session_state:
    st.session_state.show_dialog = False

  desc = ''
  min_num_comments = 0
  selected_method = ''
  num_of_results = 0
  df_info_clean = df_info.copy()
  df_content = pd.DataFrame()
  vectoriser = {}
  tfidf = {}
  cosine_sim = {}

  def search() -> pd.DataFrame:
    if selected_method == 'Generate Similar':
      dict_, tfidf_, index_, df_index_ = build_gensim(df_content)
      
      results_ = search_query_gensim(desc, dict_, tfidf_, index_, df_info_clean, num_of_results)
      return results_

    elif selected_method == 'Cosine Similarity':
      results_ = search_query_cossim(desc, vectoriser, tfidf, df_info_clean, num_of_results)
      return results_

  # ------------- Search function ------------- #
  with st.container(border=True):
    score_cols_ = ['Total_Score', 'Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']

    desc_ = st.text_input(
      label='searchbar',
      label_visibility='collapsed', 
      max_chars=250,
      placeholder='Enter the hotel name or description',
      value='Khách sạn gần trung tâm, sạch sẽ, phù hợp cho gia đình'
    )
    desc = desc_

    df_info_clean = df_info_clean.drop(columns=['Comfort_and_room_quality'])

    for col_ in score_cols_:
      df_info_clean[col_] = df_info_clean[col_].astype(str).str.replace(',', '.', regex=False)
      df_info_clean[col_] = pd.to_numeric(df_info_clean[col_], errors='coerce')   

    df_info_clean = df_info_clean.dropna(subset=score_cols_, how='all')
    df_info_clean = df_info_clean.dropna(subset=['Hotel_Description'])

    for col_ in score_cols_:
      df_info_clean[col_] = df_info_clean[col_].fillna(df_info_clean[col_].median())

    def classify_score(score):
      if score < 6.0:
          return 'low'
      elif score < 8.0:
          return 'medium'
      else:
          return 'high'
      
    for col_ in score_cols_:
      df_info_clean[col_ + '_class'] = df_info_clean[col_].apply(lambda x: classify_score(x))

    min_num_comments_ = st.number_input(
       label='Minimum number of comments',
       value=5
    )
    min_num_comments = min_num_comments_

    df_info_clean = df_info_clean[df_info_clean['comments_count'] > min_num_comments_]

    def convert_row_to_content(row):
      content_ = []

      description_ = str(row.get('Hotel_Description', '')).strip()
      if description_:
          content_.append(description_)

      for col_ in score_cols_:
          score_label_ = col_.lower()
          score_class_ = str(row.get(col_ + '_class', 'unknown')).strip().lower()
          content_.append(f'{score_label_}_{score_class_}')

      return ' '.join(content_)
    
    df_content['Content'] = df_info_clean.apply(convert_row_to_content, axis=1)

    vectoriser = TfidfVectorizer(analyzer='word', stop_words=stop_words)
    tfidf = vectoriser.fit_transform(df_content['Content'])
    cosine_sim = cosine_similarity(tfidf, tfidf)

    selected_method_ = st.selectbox(
      label='Method',
      options=['Generate Similar', 'Cosine Similarity'],
      placeholder='Select a method'
    )      
    selected_method = selected_method_

    num_of_results_ = st.number_input(
      label='Recommendation size',
      value=5
    )
    num_of_results = num_of_results_

    st.markdown('\n')
    if st.button(label='Search', width='stretch'): 
      st.session_state.df_hotel_search_results = search()

  # ------------- Search result ------------- #

  with st.container(border=True):
    display_recommend_results(
      st.session_state.df_hotel_search_results,
      df_info_clean,
      cosine_sim
    )

  if st.session_state.show_dialog == True:
    display_hotel_info(st.session_state.selected_hotel, df_info_clean, cosine_sim)
    st.session_state.show_dialog = False

def page_hotel_insight() -> st.Page:
  st.markdown('''
    # HOTEL INSIGHT
    \b  
  ''')

  hotel_names_ = sorted(df_info['Hotel_Name'].unique())

  selected_hotel = st.selectbox(
    label='Chọn khách sạn', 
    options=['None'] + hotel_names_
  )

  if selected_hotel == 'None':
    return
    
  df_filtered = df_info[df_info['Hotel_Name'] == selected_hotel]
  df_filtered = df_filtered.drop(columns=['Comfort_and_room_quality'])

  score_cols_ = ['Total_Score', 'Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']

  for col_ in score_cols_:
    df_filtered[col_] = df_filtered[col_].astype(str).str.replace(',', '.', regex=False)
    df_filtered[col_] = pd.to_numeric(df_filtered[col_], errors='coerce') 

    df_info[col_] = df_info[col_].astype(str).str.replace(',', '.', regex=False)
    df_info[col_] = pd.to_numeric(df_info[col_], errors='coerce')  
    

  df_filtered_comments = df_comments[df_comments['Hotel ID'] == df_filtered['Hotel_ID'].values[0]]
  df_filtered_comments['Score'] = df_filtered_comments['Score'].astype(str).str.replace(',', '.', regex=False)
  df_filtered_comments['Score'] = pd.to_numeric(df_filtered_comments['Score'], errors='coerce')  
  df_filtered_comments['Parsed Review Date'] = df_filtered_comments['Review Date'].apply(extract_review_date)

  st.markdown('\b')
  
  with st.container(border=True):
    st.markdown(f'''
      # {df_filtered['Hotel_Name'].values[0]}
    ''')

    # General Info
    with st.container(border=True):
      st.markdown(f'''
        ## :green[Thông Tin Chung]
        ##### :blue[Địa chỉ:] {df_filtered['Hotel_Address'].values[0]}
        ##### :blue[Hạng Sao:] {df_filtered['Hotel_Rank'].values[0]}
        ##### :blue[Tổng điểm trung bình:] {df_filtered['Total_Score'].values[0]} điểm
      ''')

    # Description
    with st.container(border=True, height=350):
      st.markdown(f'''
      ## :green[Mô Tả]
      {df_filtered['Hotel_Description'].values[0]}
    ''')

    # Key Aspects
    with st.container(border=True):
      st.markdown(f'''
        ## :green[Điểm Chi Tiết]
      ''')

      scores_row_ = st.columns(5)
      score_cols_ = ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']

      for index_, col_ in enumerate(scores_row_): 
        with col_.container(border=True):
          st.markdown(f'''
            ###### {score_cols_[index_]}
            # {df_filtered[score_cols_[index_]].values[0]}
          ''')

    # Bar Graphs
    with st.container(border=True):
      hotel_scores_ = df_filtered[['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']].iloc[0]
      avg_scores_ = df_info[['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']].mean().round(2)

      categories_ = list(hotel_scores_.keys())
      
      hotel_values_ = [hotel_scores_[cat_] for cat_ in categories_]
      avg_values_ = [avg_scores_[cat_] for cat_ in categories_]

      nationality_counts_ = df_filtered_comments['Nationality'].value_counts()
      group_counts_ = df_filtered_comments['Group Name'].value_counts()
      room_type_counts_ = df_filtered_comments['Room Type'].value_counts()
      score_counts_ = df_filtered_comments['Score'].value_counts()
      score_level_counts_ = df_filtered_comments['Score Level'].value_counts()

      graph_options_ = ['vs. All Hotels', 'Nationalities', 'Groups', 'Room Types', 'Score', 'Score Level']

      selected_options_ = st.multiselect(
        label='selected_options',
        label_visibility='collapsed',
        options=graph_options_,
        default=graph_options_
      )

      cols_ = st.columns(2, gap='large')

      for index_, option_ in enumerate(selected_options_):
        col_ = cols_[index_ % 2]
        
        with col_:
          st.markdown(f'''
            ## :green[{option_}]
          ''')
          
          if option_ == 'vs. All Hotels':
            chart_data_ = pd.DataFrame({
              'Category': categories_ * 2,
              'Score': hotel_values_ + avg_values_,
              'Type': [df_filtered['Hotel_Name'].values[0]] * len(categories_) + ['All Hotels'] * len(categories_)
            })
            
            fig_ = px.bar(
              data_frame=chart_data_,
              x='Score', y='Category',
              color='Type', barmode='group',
              orientation='h'
            )
            st.plotly_chart(fig_)

          elif option_ == 'Nationalities':
            fig_ = px.bar(
              data_frame=nationality_counts_.reset_index(),
              x='count', y='Nationality',
              color='Nationality',
              orientation='h'
            )
            st.plotly_chart(fig_)
          
          elif option_ == 'Groups':
            fig_ = px.bar(
              data_frame=group_counts_.reset_index(),
              x='count', y='Group Name',
              color='Group Name',
              orientation='h'
            )
            st.plotly_chart(fig_)

          elif option_ == 'Room Types':
            fig_ = px.bar(
              data_frame=room_type_counts_.reset_index(),
              x='count', y='Room Type',
              color='Room Type',
              orientation='h'
            )
            st.plotly_chart(fig_)
          
          elif option_ == 'Score':
            fig_ = px.bar(
              data_frame=score_counts_.reset_index(),
              x='count', y='Score',
              color='Score',
              orientation='h'
            )
            st.plotly_chart(fig_)

          elif option_ == 'Score Level':
            fig_ = px.bar(
              data_frame=score_level_counts_.reset_index(),
              x='count', y='Score Level',
              color='Score Level',
              orientation='h'
            )
            st.plotly_chart(fig_)

    # Line Graphs
    with st.container(border=True):
      col1_, col2_ = st.columns(2, gap='large')

      with col1_.container(border=False):
        st.markdown('## :green[Score Trend]')
                
        score_by_month_ = df_filtered_comments.groupby(pd.Grouper(key='Parsed Review Date', freq='ME'))['Score'].mean()

        df_score_trend = score_by_month_.reset_index()
        df_score_trend.columns = ['Month', 'Average Score']

        fig_ = px.line(
          data_frame=df_score_trend,
          x='Month',
          y='Average Score',
          markers=True
        )
        st.plotly_chart(fig_)

      with col2_.container(border=False):
        st.markdown('## :green[Review Volume Trend]')

        review_by_month_ = df_filtered_comments.groupby(pd.Grouper(key='Parsed Review Date', freq='ME')).size()

        df_review_trend_ = review_by_month_.reset_index()
        df_review_trend_.columns = ['Month', 'Average Review']

        fig_ = px.line(
          data_frame=df_review_trend_,
          x='Month',
          y='Average Review',
          markers=True
        )
        st.plotly_chart(fig_)

    # High / Low Ratings
    with st.container(border=True):
      def count_frequency(text, num=10):
        tokens_ = text.split()
        word_counts_ = Counter(tokens_)
        top_words_ = word_counts_.most_common(num)
        
        return pd.DataFrame(top_words_, columns=['Word', 'Count'])
      
      high_rating_ = df_filtered_comments[df_filtered_comments['Score Level'] == 'Trên cả tuyệt vời']
      low_rating_ = df_filtered_comments[df_filtered_comments['Score Level'] == 'Hài Lòng']

      high_rating_text_ = ' '.join(preprocess_text(high_rating_['Body'].dropna().astype(str)))
      low_rating_text_ = ' '.join(preprocess_text(low_rating_['Body'].dropna().astype(str)))

      df_high_rating = count_frequency(high_rating_text_)
      df_low_rating = count_frequency(low_rating_text_)

      col1_, col2_ = st.columns(2, gap='large')

      with col1_.container(border=False):
        st.markdown(f'## :green[High Rating]')

        fig_ = px.bar(df_high_rating,
          x='Count', y='Word',
          orientation='h'
        )
        st.plotly_chart(fig_)

      with col2_.container(border=False):
        st.markdown(f'## :green[Low Rating]')

        fig_ = px.bar(df_low_rating,
          x='Count', y='Word',
          orientation='h'
        )
        st.plotly_chart(fig_)

def page_user_review() -> st.Page:
  st.markdown('''
    # USER REVIEW
    \b  
  ''')

  score_cols_ = ['Total_Score', 'Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']

  data_info_ = (
    spark.read.csv(
        file_info,
        inferSchema=True,
        header=True,
        multiLine=True,
        escape='"',
        quote='"'
    )
    .drop('Comfort_and_room_quality')
  )

  for c in score_cols_:
    data_info_ = data_info_.withColumn(
      c,
      regexp_replace(col(c), ',', '.').cast('float')
    )

  clean_cond_ = None
  for c in score_cols_:
    cond_ = (~col(c).isNull()) & (~isnan(col(c)))
    clean_cond_ = cond_ if clean_cond_ is None else (clean_cond_ & cond_)

  data_info_ = (
    data_info_
    .filter(clean_cond_)
    .filter(col('Hotel_Description').isNotNull())
  )

  # Add class categories
  for c in score_cols_:
    data_info_ = data_info_.withColumn(
      f'{c}_class',
      when(col(c).isNull(), 'unknown')
      .when(col(c) < 6.0, 'low')
      .when(col(c) < 8.0, 'medium')
      .otherwise('high')
    )

  # ------------------------------
  # Load & clean user comments
  # ------------------------------
  data_comments_ = (
    spark.read.csv(
      file_comments,
      inferSchema=True,
      header=True,
      multiLine=True,
      escape='"',
      quote='"'
    )
    .dropna(subset=['Body', 'Reviewer Name'])
  )

  # Clean Score BEFORE filtering
  data_comments_ = data_comments_.withColumn(
    'Score',
    regexp_replace(col('Score'), ',', '.').cast('float')
  )

  # Generate pseudo user id
  data_comments_ = data_comments_.withColumn(
    'pseudo_user_id',
    concat_ws('_',
      trim(lower(col('Reviewer Name'))),
      trim(lower(col('Nationality'))),
      trim(lower(col('Group Name')))
    )
  )

  users_ = data_comments_.select('pseudo_user_id').distinct().limit(1000).toPandas()['pseudo_user_id'].tolist()

  selected_user_id_ = st.selectbox(
    label='User',
    options=users_
  )

  min_num_results_ = st.number_input(
    label='Minimum results',
    min_value=1,
    value=5
  )

  valid_hotels_ = data_info_.select(col('Hotel_ID').alias('Hotel ID')).distinct()
  
  features_ = data_comments_.select('pseudo_user_id', 'Hotel ID', 'Score')
  features_ = features_.withColumn('Score', expr("try_cast(regexp_replace(Score, ',', '.') as double)"))
  features_ = features_.join(valid_hotels_, on='Hotel ID', how='inner')


  user_id_map_ = features_.select('pseudo_user_id').distinct().withColumn('userId', monotonically_increasing_id())
  hotel_id_map_ = features_.select('Hotel ID').distinct().withColumn('hotelId', monotonically_increasing_id())

  features_mapped_ = features_.join(user_id_map_, 'pseudo_user_id', 'left').join(hotel_id_map_, 'Hotel ID', 'left').select('userId', 'hotelId', 'Score')
  features_mapped_ = features_mapped_.withColumnRenamed('Score', 'rating')
  features_mapped_ = features_mapped_.select(
    col('userId').cast('int'),
    col('hotelId').cast('int'),
    col('rating').cast('float')
  )
  features_mapped_ = features_mapped_.na.drop()

  (train_, test_) = features_mapped_.randomSplit([.8, .2], seed=42)

  model_ = ALS(
    maxIter=10, regParam=0.5, rank=10,
    userCol='userId', itemCol='hotelId', ratingCol='rating',
    coldStartStrategy='drop'
  ).fit(train_)

  pred_ = model_.transform(test_)
  pred_ = pred_.filter(pred_.prediction.isNotNull())
  pred_ = pred_.withColumn('prediction', when(col('prediction') > 10, 10.0).when(col('prediction') < 0, 0.0).otherwise(col('prediction')))

  selected_user_ = user_id_map_.filter(user_id_map_.pseudo_user_id == selected_user_id_)

  recommends_ = model_.recommendForUserSubset(selected_user_, min_num_results_)

  final_rec_ = recommends_.withColumn('recommendations', explode('recommendations'))
  final_rec_ = final_rec_.select('userId', col('recommendations.hotelId'), col('recommendations.rating'))
  final_rec_ = final_rec_.withColumn('rating', round(col('rating'), 1))
  final_rec_ = final_rec_.join(user_id_map_, 'userId')
  final_rec_ = final_rec_.join(hotel_id_map_, 'hotelId')
  final_rec_ = final_rec_.select('userId', 'hotelId', 'rating')

  final_with_ids_ = final_rec_.join(hotel_id_map_, on='hotelId', how='left')

  final_with_info_ = final_with_ids_.join(data_info_, final_with_ids_['Hotel ID'] == data_info_['Hotel_ID'], how='left')
  final_with_info_ = final_with_info_.filter(col('Hotel_Name').isNotNull())

  df_results_ = final_with_info_.toPandas() 

  with st.container(border=True):
    st.markdown(f'''
      ## Kết Quả
    ''')

    hotel_names_ = df_results_['Hotel_Name'].to_list()

    for index_, name_ in enumerate(hotel_names_):
      with st.container(border=True):
        col1_, col2_ = st.columns([10, 1])

        with col1_:
          hotel_ = df_results_.iloc[index_]
          score_ = hotel_['Total_Score']
          score_str_ = f'{score_:.1f}' if pd.notnull(score_) else 'N/A'
          
          st.markdown(f"""
            <div style='line-height:1.5'>
              <span style='font-size:20px; font-weight:bold; color:#ffffff'>{hotel_['Hotel_Name']}</span><br>
              <span style='font-size:18px; color:#27ae60;'>Tổng điểm: {score_str_}</span> &nbsp;&nbsp;&nbsp;&nbsp;
              <span style='font-size:18px; color:#2e86c1;'>⭐ {hotel_['Hotel_Rank']}</span>
            </div>
          """, unsafe_allow_html=True)
        
        with col2_:
          if st.button("🔍", key=f"select_{index_}", width='stretch'):
            st.session_state.selected_hotel = hotel_
            st.session_state.show_dialog = True
            st.rerun()

st.set_page_config(layout='wide')

pg = st.navigation([
    st.Page(page_hotel_search, title='Tìm kiếm Khách sạn'),
    st.Page(page_hotel_insight, title='Hotel Insight'),
    st.Page(page_user_review, title='User Review')
  ],
  position='top'
)

pg.run()
