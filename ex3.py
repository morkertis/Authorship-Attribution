# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 08:40:24 2019

@author: mor
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import re
import os
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
#import json #######################hide
#import matplotlib.pyplot as plt#########################hide
#import seaborn as sns#########################hide
#from gensim.models import word2vec #########################hide
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_validate
from warnings import filterwarnings
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.externals import joblib
filterwarnings('ignore')


def read_text(path,cols_names,date_col_str):
    '''
    returns pandas DataFrame after reading the tsv file.
    Args:
        path(str) the path of the file
        cols_names(list) list contain string for each column name
        date_col_str(str) string of the column that parse as date
    Return:
        Pandas DataFrame with columns names as pass and one column in date format
    '''
    time_format = '%Y-%m-%d %H:%M:%S'
    dateparse = lambda x: pd.datetime.strptime(x, time_format)
    return pd.read_csv(path,delimiter='\t|\n',names=cols_names, engine='python',
                       parse_dates=[date_col_str],date_parser= dateparse)


def create_label(df):
    '''
    Returns pandas DataFrame after adding column name 'label' with value 0 or 1 for each row.
        if the device is android and the user_handle  'realDonaldTrump' the 0 else 1
    Args:
        df(DataFrame) that contain column device and column user_handle for labeling
    Return: 
        The same df(DataFrame) with column 'label' 
    '''
    device_android = df.device == 'android'
    user_trump = df.user_handle == 'realDonaldTrump'
    df['label'] = np.where(device_android & user_trump, 0, 1)
    return df


def drop_cols(df,cols):
    '''
    Return pandas DataFrame after deleting some columns.
    Args:
        df(DataFrame)
        cols(list) list contain string for each column that we want to delete
    Return:
        The same df(DataFrame) without the deleted columns
    '''
    return df.drop(cols,axis=1)


def time_features(df,hour_sep,time_stamp_col):
    '''
    Return pandas DataFrame after adding time features. 
        1. feature 'hour_cat' that represent part of day by hour of the tweet - category feature
        2. feature 'day_of_week' that represent the day in the week the tweet post [0-6] - category feature
        3. feature 'weekend' that represent if american weekend saturday and sunday - binary feature
    Args:
        df (DataFrame)
        hour_sep(int): set the hour for each part of day need to divide from 24 hours 
                    - for exmple if set to 6 than will be 4 category [0-6,6-12,12-18,18-24]
        time_stamp_col(str): the column name that is a datetime format
    Return: 
        The same df(DataFrame) with adding columns 'hour_cat', 'day_of_week', 'weekend'    
    '''
    time_dist= list(range(0,24 + hour_sep,hour_sep))
    df['hour_cat'] = pd.cut(pd.to_datetime(df[time_stamp_col]).dt.hour,time_dist,labels = False,right=False)
    df['day_of_week'] = df[time_stamp_col].dt.dayofweek
    df['weekend'] = np.where(df.day_of_week.isin([5,6]) , 1, 0)
    return df
    

def text_features(df,text_col):
    """
    Returns df pandas DataFrame with columns for counting multiple text features. need to include the text column. 
        features:    1. uppercase charracters 2.uppercase word 3. quotation mark 4. 'At' sign 5. hashtags 6. numbers
                     7. question mark 9. exclamation mark  9. retweet ('RT')
    Args:
        df (pandas DataFrame): the dataframe with text column. need to contain the text column
        text_col(str): name of the text columns 
    Returns:
         pandas DataFrame df after adding the columns
    """
    # counting quotation mark
    df['quotation'] = df[text_col].apply(lambda text: len(re.findall('\"',text)))
    
    # counting upper case of characters
    df['upper_char'] = df[text_col].apply(lambda text: len(re.findall('[A-Z]',text)))
    
    # counting upper case of words
    df['upper'] = df[text_col].apply(lambda text: len(re.findall('((?:^|\s)[A-Z]+)\W',text)))
    
    # counting at sign
    df['at'] = df[text_col].apply(lambda text: len(re.findall('\@',text)))
    
    # counting hashtags
    df['hashtags'] = df[text_col].apply(lambda text: len(re.findall('\#',text)))
    
    # counting exclamation mark
    df['exclamation'] = df[text_col].apply(lambda text: len(re.findall('\!',text)))
    
    # counting question mark
    df['question'] = df[text_col].apply(lambda text: len(re.findall('\?',text)))
    
    # counting numbers
    df['numerics'] = df[text_col].apply(lambda text: len(re.findall('\s(\d+)\s',text)))
    
    # counting numbers of retweets
    df['rt'] = df[text_col].apply(lambda text: len(re.findall('((?:^|\s)RT)',text)))
    
    return df


def normalize_text(text,pad_punc=string.punctuation,remove_punc=string.punctuation):
    """Returns a normalized string based on the specify string.
        Explain: 
            - Change website url string to str 'website'
            - Multi dots in a row switch to 'dots'.
            - Change hashtags(#) sign to str 'hash' and separate text by upper case in hashtags words. 
                    example '#ObamacareFail'--> '#Obamacare Fail' --> 'hash Obamacare Fail'
            - Lower case of all characters.
            - Change at sign(@) sign to str 'atsign'
            - Remove all string punctuation characters.
            - Remove extra spaces in a row and leave only one space between tokens
            - Remove single characters if chars is False except 'a' and 'i'
            - Change all numbers that separated by space to 'num'
       Args:
           text (str): the text to normalize
           pad_punc(str): characters for creating a space before and after the characters
                           default: string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
           remove_punc(str): characters to remove from the text
                           default: string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
       Returns:
           string. the normalized text.
    """
    punc_spaces = re.compile('([%s])' % re.escape(pad_punc))
    punc = re.compile('[%s]' % re.escape(remove_punc))

    text = re.sub(r'(http(?:.*?))(?:\s|$)',' website ',text) #catch website url
    text = re.sub('\.{2,}',' dots',text) #catch dots
    text = re.sub(r'([A-Z][a-z]{2,})', r' \1 ', text)# separate hash tags by upper case 
    text = text.lower()
    text = re.sub(r'\#',' hash ',text)
    text = re.sub(r'\@',' atsign ',text)                  
    text = re.sub(punc_spaces, r'', text)
    text = re.sub(punc,'',text)
    text = re.sub(r'\b((?![ai])[a-z])\b','',text)
    text = re.sub('\s(\d+)\s',' num ',text)
    text = re.sub('\s{2,}', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text=text.strip()
        
    return text


def words(text,str_format=False): 
    '''
    Returns list or string of all the words in the text 
    Args:
        text (str): string of text
        str_format(boolean): check if return string or list of all words. 
            default=False
    Returns: 
        List or string of all the words in the text
    '''
    if str_format:
        return ' '.join(re.findall(r'\w+', text))
    return re.findall(r'\w+', text)


def clean_text(df,text_col):
    '''
    Returns pandas DataFrame df with preprocess text with column name 'norm_text' .
    :Args:
        df(pandas DataFrame) dataframe with the original columns
        text_col(str): name of the text column
    Returns: df(DataFrame) with 'norm_text' that contain normalize text (normalize_text function) of all the text column data
    '''
    df['norm_text'] = df[text_col].apply(lambda text: words(normalize_text(text),True))
    df['norm_text_token'] = df[text_col].apply(lambda text: words(normalize_text(text),False))
    
    return df
    

def normilize_text_features(df,text_col,stop_words):
    '''
    Returns pandas DataFrame df with text features of normilize text 'norm_text'.DataFrame need to include the text column. 
        The use of 'norm_text' column is not mandatory but recommended for better features.
        text features: 
            - tweet_length_char - count length of tweet by number of characters
            - tweet_length_word - count length of tweet by number of words
            - char_word_ratio - calculating char word ratio - length of average word
            - stopwords - counting the use of stopwords
    Args:
        df (pandas DataFrame)
        text_col(str): the column that contain the text
        stop_words(list): list of string contain stop words
    Returns:
        pandas DataFrame df with extra columns 'tweet_length_char','tweet_length_word','char_word_ratio','stopwords'
    '''
    # count length of tweet by number of characters
    df['tweet_length_char'] = df[text_col].apply(len)
    
    # count length of tweet by number of words
    df['tweet_length_word'] = df[text_col].apply(lambda text: len(re.findall('\w+',text)))
    
    # calculating char word ratio - length of average word
    df['char_word_ratio'] = df['tweet_length_char'] / (df['tweet_length_word'] + 1)
    
    # counting the use of stopwords
    df['stopwords'] = df[text_col].apply(lambda text: len([w for w in re.findall('\w+',text) if w in stop_words]))
    
    return df


def plot_hist(data,title,filename= None,**kwargs):
    '''
    This function need matplotlib package.
    Returns histogram plot and allowing to save the plot as a picture.
    Args:
        data: can be as list or numpy array with flatten dimension. contain the data for the histogram
        title(str): the name given to the plot
        filname(str): the filename given to the saving file. default=None -> dont save the file and only display
        **kwargs(dictionary): enable to change parameters of the histogram dynamically. dictionary can be in any size
            but need to contain only keys that exists in matplotlib histogram api
    Returns:
        display histogram plot and save the plot if required
    '''    
    plt.hist(data,**kwargs)
    plt.title(title)
    if filename:
        plt.savefig(filename + '.png')
    plt.show()


def plot_face_grid(df,col,sep_col,title,filename=None):
    '''
    This function need seaborn package.
    Returns multiple histograms that  separate by column value. For example: histogram of length of text  separate by label.
    Args:
        df (pandas DataFrame): need to contain the column for the histogram and the column for separating by.
        col(str): main column name for the histogram
        sep_col(str): column name for separating the histograms
        title(str): the name given to the plot
        filname(str): the filename given to the saving file. default=None -> dont save the file and only display
    Returns:
        display multiple histograms plot and save the plot if required
    '''
    g = sns.FacetGrid(data=df, col=sep_col)
    g.fig.suptitle(title)
    g.map(plt.hist, col, bins=50)
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        plt.savefig(filename+'.png')
    plt.show()


def train_word2vec(data,out_model = "word2vec.model"):
    '''
    This function need gensim package.
    
    Train word2vec SkipGram model and save the model on disk.
    Args:
        data(list): list contain string of sentences
        out_model(str): name of the model when save
    '''
    model = word2vec.Word2Vec(data, sg = 1, # 0=CBOW , 1= SkipGram
                     size=100, window=5, min_count=5)

    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)
    model.save(out_model)

  
def getModel(model = "word2vec.model"):
    '''
    This function need gensim package.
    
    Load train word2vec model.
    Arg:
        model(str): model name to load
    Return:
        The loaded model
    '''
    model = word2vec.Word2Vec.load(model)

    return model


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=100):
    '''
    Returns numpy array of average embedding for a list of tokens.
    Args:
        tokens_list: list of string.
        vector(dictionary): contain trained vectors of the word embedding model
        generate_missing(boolean): check if missing word than generate random embedding if True.
            defualt = False than if missing puts zeros
        k(int): size of the embedding dimension
    Return numpy array of average embedding for a list of tokens.
    '''
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, tokenDF, generate_missing=False):
    '''
    Returns numpy array with average embedding for each text row in the pandas Series
    Args:
        vectors(dictionary): contain trained vectors of the word embedding model
        tokenDF(pandas Series): the text column tokenize by word
        generate_missing(boolean): check if missing word than generate random embedding if True.
            defualt = False than if missing puts zeros
    Returns numpy array with average embedding for each text row in the pandas Series
    '''
    embeddings = tokenDF.apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return np.array(list(embeddings))


def train_classifier_sklearn(clf,X,y,fold=5):
    '''
    Returns f1 score and accuracy score of classifier after k-fold train validation.
        the score return in dictionary object. The X data are insert to pipline that standardize features.
    Args:
        clf(sklearn classifier)
        X: data in foramt of numpy array as explanatory variables
        y: data in foramt of numpy array as explained variable
        fold(int): number of fold for the cross validation train. defualt = 5
    Returns:
        dictionary with the results of the metrics with the mean standard deviation.
    '''
    clf_pipe = make_pipeline(StandardScaler(), clf)
    scoring = ['f1', 'accuracy']
    scores = cross_validate(clf_pipe, X, y, cv=fold,scoring=scoring)
    return {'f1' :(scores['test_f1'].mean(), scores['test_f1'].std() * 2),
                          'accuracy' :(scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2)}


def train_models_sklearn(X,y):
    '''
    Returns dictionary object {'model':{'f1':(mean,std*2)}} where model is the name of the classifier. 
        'f1'/'accuracy' are the metrices. mean and std are mean and standard deviation of the metric.
        we test 4 different classifiers: logistic regression, svm with rbf kernel, svm with linear kernel and random forest
    Args:
        X: data in foramt of numpy array as explanatory variables
        y: data in foramt of numpy array as explained variable
    Returns:
        dictonary with the models and there scores in the metrics
    '''
    models={}
    models['logistic_regression'] = LogisticRegression()
    models['svm_rbf'] = SVC(kernel='rbf',max_iter=100000)
    models['svm_linear'] = SVC(kernel='linear' ,max_iter=1000000)
    models['random_forset'] = RandomForestClassifier()
    for key,model in models.items():
       models[key]=train_classifier_sklearn(model,X,y) 
    return models


def save_json(data,filename):
    '''
    This function need json package.
    
    Save data in the disk in json format.
    data(dictionary): the data for daving in the disk
    filname(str): the file name for saving
    '''
    with open(filename+'.txt', 'w') as outfile:
        json.dump(data, outfile,indent = 4)


# =============================================================================
# torch network and train evaluate pipeline
# =============================================================================
class Net(nn.Module):
    """
    The class implements pytorch neural network by inherited torch module
    """
    def __init__(self, x_shape):
        '''
        initialize network layers of MLP network. Contains relu, sigmoid, dropout with 0.2 probability
            and 4 fully connected layers.
        Arg:
            x_shape(int): the inout shape for the network
        '''
        super(Net, self).__init__()
		
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(x_shape, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)
        
    def forward(self, x):
        '''
        Return the output of the network after going through the entire network.
            the netwrok flow: fully connected(fc)-> relu -> dropout -> fc -> relu -> fc -> relu -> fc -> sigmoid
        Arg:
            x(torch tensor): contain the input data
        Return:
            the output of the network.
        '''
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        output = self.sigmoid(self.fc4(x))

        return output


def train_torch_model(model,X_train,y_train,epochs=5):
    '''
    Return torch model after training.
    Args:
        model(torch model): initialize torch network
        X_train(numpy array): X data for train the model
        y_train(numpy array): y data for train the model
        epochs(int): number of cycle on the data. defualt = 5
    Return torch train model
    '''
    x_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(dataset=dataset, batch_size=16)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    epochs=5
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:   
            optimizer.zero_grad()
            
            # Complete a forward pass
            output = model(x_batch)
            
            # Compute the loss, gradients and change the weights
            y_batch=y_batch.long()
            loss = criterion(output,y_batch)
            loss.backward()
            optimizer.step()
    return model


def predict_torch_model(model,X_val,y_val):
    '''
    Returns numpy array of the probability prediction of 1 by the model.
    Args:
        model(torch model): trained torch network
        X_val(numpy array): X data for evaluate the model
        y_val(numpy array): y data for evaluate the model
    Returns numpy array of the probability prediction of 1
    '''
    x_tensor = torch.from_numpy(X_val).float()
    y_tensor = torch.from_numpy(y_val).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    test_loader = DataLoader(dataset=dataset, batch_size=16)
    
    y_pred_list=[]
    y_val_list=[]
    model.eval()
    for x_batch, y_batch in test_loader:   
    
        y_pred = model(x_batch)
        y_pred = y_pred.data.tolist()
        y_batch = y_batch.long().data.tolist()
        y_pred_list.extend(y_pred)
        y_val_list.extend(y_batch)
    return np.array(y_pred_list)[:,-1]


def train_torch_net_kfold(X,y,epochs=5,fold=5):
    '''
    Returns dictionary {'f1':(mean,std*2)}. 
        where 'f1'/'accuracy' are the metrices. mean and std are mean and standard deviation of the metric.
        the torch model train as cross validation. The data are split to k-fold. than standardize afterward train and evaluate
    Args:
        X: data in format of numpy array as explanatory variables
        y: data in format of numpy array as explained variable
        epochs(int): number of cycle on the data. defualt = 5
        fold(int): number of fold for the cross validation train. defualt = 5
    Return dictionary with the results of the metrics with the mean and standard deviation.
    '''
    accuracy_li=[]
    f1_list=[]
    kf = KFold(n_splits=fold,shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        
        scaler=StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        model = Net(x_shape=X_train.shape[-1])
        model = train_torch_model(model,X_train,y_train,epochs=epochs)
        y_pred = predict_torch_model(model,X_val,y_val)
        
        accuracy_li.append(accuracy_score(y_val,(y_pred>0.5).astype(int)))
        f1_list.append(f1_score(y_val,(y_pred>0.5).astype(int)))
        
    acc_np = np.array(accuracy_li)
    f1_np = np.array(f1_list)
    return {'f1':(f1_np.mean(),f1_np.std()*2),'accuracy':(acc_np.mean(),acc_np.std()*2)}


def create_random_forest_parametr_grid():
    '''
    Returns dictionary for parameter tuning of random forest.
    '''
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]
    criterion =['gini','entropy']
    max_features = ['auto', 'log2',None]
    max_depth = [int(x) for x in np.linspace(1, 20, num = 11)]+[None]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    
    # Create the random grid
    random_grid = {'criterion':criterion,
                   'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    print(random_grid)  
    return random_grid


def write_to_txt(data_list,filename):
    '''
    Save data on txt file on the disk.
    Args:
       data_list(iterable object): iterable object that contain str or number
       filename(str): the name of the file for save
    '''
    with open(filename+'.txt','w') as f:
        f.write(" ".join([str(p) for p in data_list]))      


def save_pickle(obj,obj_name):
    '''
    This function save object in format .pkl on the disk
    Args:
        obj(object): python object, can be list, model, sklearn classes etc.
        obj_name(str): the name of the pickle file without the ending .pkl
    '''
    joblib.dump(obj,obj_name+'.pkl')


def load_pickle(obj_name):
    '''
    This function load pickle object by name 'obj_name' and return the object.
    Args
        obj_name(str): the name of the pickle file without the ending .pkl
    Return the pickled object after loading it
    '''
    return joblib.load(obj_name+'.pkl')


def tfidf_svd_pipeline(X,nltk_stopwords,pipe_name=None):
    '''
    This function create a pipeline of tf-idf -> svd -> 
        -> transform/fit_transform on data -> save pipeline in a pickle format -> returns data after passing the pipeline
    The pipeline contain: 
        tfidf with stopwords , lowercasing, ngram in size unigram to trigram, and remove rare words 
        svd for dimension reduction to 100 dim
        and finally transform the data
    Args:
        X(list): list of sentences(str)
        nltk_stopwords(list): list of string contain stop words
        pipe_name(str): pipeline name for loading tfidf->svd if exists in the directory. default=None
    Returns
        numpy array of X after passing in the pipeline and save pipeline in a pickle format
    '''
    if pipe_name:
        pl = load_pickle(pipe_name)
        return pl.transform(X)
    else:
        pl = Pipeline([
                ('tfidf', TfidfVectorizer(lowercase=True, stop_words=nltk_stopwords,ngram_range=(1,3),min_df=4)),
                ('svd', TruncatedSVD(n_components=100, n_iter=7, random_state=42))
                ])
        X_tf_svd = pl.fit_transform(X)
        save_pickle(pl,'tf_svd')
        return X_tf_svd
        
    
def preprocess_hand_crafted(path,cols):
    '''
    This function read .tsv file preporcess the data by function like: time_features,text_features,
    clean_text, normilize_text_features (see an explanation of these functions) and return pandas DataFrame df
    Args:
        path(str): string of the file .tsv path
        cols(list): list of the names of columns for .tsv file
    Returns
        pandas DataFrame df after adding columns for creating features and removing unnecessary columns
    '''
    df = read_text(path,cols,'time_stamp')
    if 'device' in cols:
        df = create_label(df)
    cols_to_drop = set(['tweet_id','user_handle','device'] ).intersection(set(cols))
    df = drop_cols(df,cols_to_drop)
    
    hour_sep = 4
    nltk_stopwords = stopwords.words('english')
    nltk_stopwords = list(filter(None,map(normalize_text,nltk_stopwords)))#clean stopword from special characters
      
    df = time_features(df,hour_sep,'time_stamp')
    df = text_features(df,'tweet_text')
    df = clean_text(df,'tweet_text')
    df = normilize_text_features(df,'norm_text',nltk_stopwords)
    
    return df
  
    
def preprocess_data(path,cols,emb_pkl_path,tf_svd_pipe=None):
    '''
    This function preprocessing handcrafted features by preprocess_hand_crafted function and creating svd tfidf features
    and embedding features.
    Returns three numpy arrays: the first array contain concat data of handcrafted features with svd on tfidf data,
        the second array contain concat data of handcrafted features with embedding data
        and the last contain the labeled data. three numpy returns are X_data_svd, X_data_emb, y_data.
    Args
        path(str): string of the file .tsv path
        cols(list): list of the names of columns for .tsv file
        emb_pkl_path(str): string of the path for the trained embedding file pickle .pkl 
        tf_svd_pipe(str): pipeline name for loading tfidf->svd if exists in the directory (.pkl file). default=None
    Returns:
        Three numpy arrays that in a format (features,features,labels) -> X_data_svd, X_data_emb, y_data
    '''
    nltk_stopwords = stopwords.words('english')
    nltk_stopwords = list(filter(None,map(normalize_text,nltk_stopwords)))#clean stopword from special characters
    
    df = preprocess_hand_crafted(path,cols)
    
    cols_to_drop = set(['tweet_text','time_stamp']).intersection(set(cols))
    df = drop_cols(df,cols_to_drop)
    
    cols = df.columns.tolist()
    
    X_data_svd = tfidf_svd_pipeline(df.norm_text.tolist(),nltk_stopwords,tf_svd_pipe)
    
    word_vec_dict_v2 = load_pickle(emb_pkl_path)
    embeddingAVG = get_word2vec_embeddings(word_vec_dict_v2, df["norm_text_token"])
    
    cols_to_drop = set(['label','norm_text', 'norm_text_token']).intersection(set(cols))
    
    X_data_svd = np.concatenate([df.drop(cols_to_drop,axis=1).values,X_data_svd],axis=-1)
    
    y_data = None
    if 'label' in df.columns.tolist():
        y_data = df.label.values
    
#    embeddingAVG = pd.read_csv(emb_path,compression='gzip').values
    X_data_emb = np.concatenate([df.drop(cols_to_drop,axis=1).values,embeddingAVG],axis=-1)
    
    return X_data_svd, X_data_emb, y_data


# =============================================================================
# function that are mandatory
# =============================================================================
def train_best_model():
    '''
    This function train the best model of this assignment - random forest with his best parameters tune
    Return trained model after training on the data.
    Function read: train .tsv file, embedding .csv.gz. preprocess data and train the best model  
    '''
    train_path = 'trump_train.tsv'
    emb_pkl_path = 'word_vec_dict' #path to saved embedding #'embedding_train.csv.gz'
    cols_train = ['tweet_id','user_handle','tweet_text','time_stamp','device']
    
    best_param={'n_estimators': 150, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': 12,
     'criterion': 'entropy', 'bootstrap': True}
    X_train_svd, X_train_emb, y_train = preprocess_data(train_path,cols_train,emb_pkl_path)
    
    rf_best = RandomForestClassifier(**best_param)
    rf_best.fit(np.concatenate([X_train_svd,X_train_emb],axis=-1), y_train)
    return rf_best


def load_best_model():
    '''
    Return trained model after loading it from pickle file
    '''
    return load_pickle('best_model')


def predict(m, fn):
    '''
    Returns m(model) prediction on data. This function get model and path(str) for data and return the prediction
    on the data after preprocessing the data and creating the necessary features.
    Args
        m(model): model that have a predict method (recommended trained model)
        fn(str): file path
    Return list of predictions for each row
    '''
    test_path = fn
    cols_test = ['user_handle','tweet_text','time_stamp']
#    emb_test_path = 'embedding_test.csv.gz'
    emb_pkl_path = 'word_vec_dict'
    
    tf_svd_pipe='tf_svd_best'
    if os.path.exists(os.getcwd()+'\\tf_svd.pkl'):
        tf_svd_pipe='tf_svd'

    X_test_svd, X_test_emb, _ = preprocess_data(test_path,cols_test,emb_pkl_path,tf_svd_pipe=tf_svd_pipe)
    preds = m.predict(np.concatenate([X_test_svd,X_test_emb],axis=-1))
    return list(preds)


# =============================================================================
# main function for running the program
# =============================================================================
def main():
    '''
    This function contain all the assignment requirements that include 3 main steps: preprocess, train and prediction
    preprocess include: creating cleannig data, handcarfted features, tf-idf, svd(dimension reduction), 
        and training embedding (package gensim therefore under comment).
    train include: train validation on 5 different models: logistic_regression, svm_rbf, svm_linear, random_forset and torch mlp. 
    and finally predict on the test data: after preprocces test data, tranform the fitted tf-idf and svd by pipeline and load the
    trained embedding for predition on the test data.
    ***training the word2vec model by gensim package and saving the vectors in csv.gz file. 
    ***gensim and all related it under comment and only using the load of the vectors.
    Some of the code are under comments because: 
        - using extra packages for pre-training and for visualization. 
        - time consuming process of parameters tuning for searching the best model - random search on 3 models 
        - save results and plots to disk
    '''

    
    pd.set_option('display.max_columns', 8)
    train_path = 'trump_train.tsv'
#    emb_path = 'embedding_train.csv.gz'#path to saved embedding
    emb_pkl_path = 'word_vec_dict'
    cols_train = ['tweet_id','user_handle','tweet_text','time_stamp','device']
    
    
    # =============================================================================
    # =============================================================================
    # pre-training section - #need other packages
    # =============================================================================
    # =============================================================================
    
    # =============================================================================
    # data visualization - using matplotlib and seaborn packages
    # =============================================================================

    #df = preprocess_hand_crafted(train_path,cols_train)
    #plot_hist(df.time_stamp.dt.hour.values,'Hour Distribution')
    #plot_hist(df.time_stamp.dt.dayofweek.values,'Day of Week Distribution',**{'bins':7})
    #plot_face_grid(df,'tweet_length_word','label','tweet length by words','tweet_length_word')
    #plot_face_grid(df,'tweet_length_char','label','tweet length by characters','tweet_length_char')
    
    # =============================================================================
    # train word2vec and load embedding using gensim package
    # =============================================================================
    #train_word2vec(df.norm_text_token.tolist())
    #load word2vec model vectors

    # =============================================================================
    # need to load the model and gensim package
    # =============================================================================
    #model=getModel()
    #model_en=model.wv
    #word_vec_dict={word:model_en[word] for word in model_en.vocab}
    #save_pickle(word_vec_dict,'word_vec_dict')

 

    ## calculations average embedding for tweet
    #embeddingAVG = get_word2vec_embeddings(model_en, df["norm_text_token"])
    #word_vec_dict_v2=load_pickle('word_vec_dict')
    #embeddingAVG = get_word2vec_embeddings(word_vec_dict_v2, df["norm_text_token"]) 
    
    ##save embedding on csv file
    #pd.DataFrame(embeddingAVG).to_csv('embedding_train.csv.gz',index=False,compression='gzip') #save compress csv file
    # =============================================================================
    # =============================================================================
    
    
        
    # =============================================================================
    # =============================================================================
    # train section
    # =============================================================================
    # =============================================================================
    
    
    # =============================================================================
    # preprocess train data
    # =============================================================================   
    
    X_train_svd, X_train_emb, y_train = preprocess_data(train_path,cols_train,emb_pkl_path,tf_svd_pipe='tf_svd')
    
    
    # =============================================================================
    # train all the models 
    # =============================================================================
    
    # train validation sklearn models
    # ===============================
    # handcrafted features concat with: 1 - tfidf_svd 2 - embedding 3 -  tfidf_svd and embedding
    # ======================================================================================
    result_svd = train_models_sklearn(X_train_svd,y_train)
    result_emb = train_models_sklearn(X_train_emb,y_train)
    result_svd_emb = train_models_sklearn(np.concatenate([X_train_svd,X_train_emb],axis=-1),y_train)
    # ======================================================================================
    
    
    # train validation torch model
    # ===============================
    # handcrafted features concat 1 - tfidf_svd 2 - embedding 3 -  tfidf_svd and embedding
    # ======================================================================================
    result_svd['torch'] = train_torch_net_kfold(X_train_svd,y_train)
    result_emb['torch'] = train_torch_net_kfold(X_train_emb,y_train)
    result_svd_emb['torch'] = train_torch_net_kfold(np.concatenate([X_train_svd,X_train_emb],axis=-1),y_train)
    # ======================================================================================
    # ======================================================================================
    
    
    # =============================================================================
    # save result in json - need json packages
    # =============================================================================
    #save_json(result_svd,'result_svd')
    #save_json(result_emb,'result_emb')
    #save_json(result_svd_emb,'result_svd_emb')
    #
    #print(json.dumps(result_svd, indent = 4))
    # =============================================================================
    
    
    # =============================================================================
    #  parameter tuning on best model
    # =============================================================================
    # =============================================================================
    # train on all data after selecting the best classifier - random forset
    # and preform parameter tuning. Also checking three types of data with the handcarfted data: 
    #    handcrafted concat + 1. embedding concat with svd_tfidf. 2. svd_tfidf. 3.embedding 
    # =============================================================================
    
    
    # =============================================================================
    #   run tuning of 3 models take a long time therefore under comment 
    # =============================================================================
    #rf_all_data = RandomForestClassifier()
    #rf_tfidf = RandomForestClassifier()
    #rf_emb = RandomForestClassifier()
     
    #from sklearn.metrics import make_scorer
    #scoring = {'f1': 'f1','accuracy': make_scorer(accuracy_score)}
     
    
    #parameters = create_random_forest_parametr_grid()
    #gs_rf_all = RandomizedSearchCV(estimator=rf_all_data, param_distributions=parameters,scoring=scoring,
    #                               refit='f1',n_iter=50,n_jobs=-1,verbose=2, random_state=0)
    #gs_rf_all.fit(np.concatenate([X_train_svd,X_train_emb],axis=-1), y_train)
    #
    #gs_rf_tf = RandomizedSearchCV(estimator=rf_tfidf, param_distributions=parameters,scoring=scoring,
    #                               refit='f1',n_iter=50,n_jobs=-1,verbose=2, random_state=0)
    #gs_rf_tf.fit(X_train_svd, y_train)
    # 
    #gs_rf_emb = RandomizedSearchCV(estimator=rf_emb, param_distributions=parameters,scoring=scoring,
    #                               refit='f1',n_iter=50,n_jobs=-1,verbose=2, random_state=0)
    #gs_rf_emb.fit(X_train_emb, y_train)
    
    
    # =============================================================================
    # compare three cross validation with diffrent types of data
    # =============================================================================
    #scoredf_all = pd.DataFrame(gs_rf_all.cv_results_)[['mean_test_accuracy', 'std_test_accuracy','rank_test_accuracy','mean_test_f1', 'std_test_f1','rank_test_f1']]
    #print(scoredf_all[scoredf_all.rank_test_f1==1])
    #
    #scoredf_tf = pd.DataFrame(gs_rf_tf.cv_results_)[['mean_test_accuracy', 'std_test_accuracy','rank_test_accuracy','mean_test_f1', 'std_test_f1','rank_test_f1']]
    #print(scoredf_tf[scoredf_tf.rank_test_f1==1])
    #
    #scoredf_emb = pd.DataFrame(gs_rf_emb.cv_results_)[['mean_test_accuracy', 'std_test_accuracy','rank_test_accuracy','mean_test_f1', 'std_test_f1','rank_test_f1']]
    #print(scoredf_emb[scoredf_emb.rank_test_f1==1])
    # =============================================================================
    # =============================================================================
    # =============================================================================
    
    
    # =============================================================================
    #  the selected model and best parameters and data   
    # =============================================================================
    best_param={'n_estimators': 150, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': 12,
     'criterion': 'entropy', 'bootstrap': True}
    
    rf_best = RandomForestClassifier(**best_param)
    rf_best.fit(np.concatenate([X_train_svd,X_train_emb],axis=-1), y_train)
    #save_pickle(rf_best,'rf_best')
    
    # =============================================================================
    # =============================================================================
    # test section
    # =============================================================================
    # =============================================================================
    
    test_path = 'trump_test.tsv'
    cols_test = ['user_handle','tweet_text','time_stamp']
#    emb_test_path = 'embedding_test.csv.gz'
    emb_pkl_path = 'word_vec_dict'
    # =============================================================================
    # pre - preprocessing data
    # =============================================================================
    # =============================================================================
    #   need to load the model and gensim package
    # =============================================================================
    ##calculations average embedding for tweet
    #embeddingAVG_test = get_word2vec_embeddings(model_en, df_test["norm_text_token"])
    
    ##save embedding on csv file
    #pd.DataFrame(embeddingAVG_test).to_csv('embedding_test.csv',index=False)
    #pd.DataFrame(embeddingAVG_test).to_csv('embedding_test.csv.gz',index=False,compression='gzip') #compress csv file
    # =============================================================================
    # =============================================================================
    
    
    # =============================================================================
    # preprocess test data
    # =============================================================================
    X_test_svd, X_test_emb, _ = preprocess_data(test_path,cols_test,emb_pkl_path,tf_svd_pipe='tf_svd')
    
    # =============================================================================
    # predict test file and write to txt file the results
    # =============================================================================
    preds = rf_best.predict(np.concatenate([X_test_svd,X_test_emb],axis=-1))
    
    #save file of prediction
    #write_to_txt(preds,'300830692')


#run the relevent function
if __name__ == "__main__":
#    main()
    
#    model = train_best_model()
    model = load_best_model()
    print(predict(model, 'trump_test.tsv'))




