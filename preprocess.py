'''clean up and preliminary feature engineering
'''

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import string, re
from collections import Counter
import itertools
import enchant


#correct spelling for search_term, and create stemmed sets for search_term and product_description
def clean_up(df_all,df_description):
    #change encoding for all relevant columns
    df_all['product_title'] = df_all['product_title'].str.decode('ISO-8859-1').str.encode('ascii', 'ignore')
    df_all['search_term'] = df_all['search_term'].str.decode('ISO-8859-1').str.encode('ascii', 'ignore')
    df_description['product_description'] = df_description['product_description'].str.decode('ISO-8859-1').str.encode('ascii', 'ignore')

    #correct spelling, create vocabulary sets in search_term and product_description and focus on those not present in description
    search_set,search_token = vocab_set(df_all,'search_term')
    description_set,description_token = vocab_set(df_description,'product_description')
    bad_term = set.difference(search_set,description_set)
    #use pyenchant to correct spelling, with weights from product_description
    d = enchant.Dict("en_US")
    NWORDS = Counter(description_token)
    bad_dic={}
    for word in bad_term:
        candidates = d.suggest(word)
        if candidates:
            bad_dic[word]=max(d.suggest(word),key=NWORDS.get)
        else:
            bad_dic[word]=word
    #after correct_spelling, bad_term decrease from 3619 to 338
    df_all['search_term_correct'] = df_all['search_term'].apply(correct_spelling)
    df_all['search_term_correct'] = df_all['search_term_correct'].str.decode('ISO-8859-1').str.encode('ascii', 'ignore')
    #make stemmed sets for search_term,title, and product_description
    df_all['search_set'] = df_all['search_term_correct'].apply(lambda s: tokenize(s,True,False,True,False))
    df_all['title_set'] = df_all['product_title'].apply(lambda s: tokenize(s,True,False,True,False))
    df_description['description_set'] = df_description['product_description'].apply(lambda s: tokenize(s,True,False,True,True))
    return df_all, df_description


#create new features based on token ratios between search_term, and brand/product_description/title
def feature_engineering(df):
    #use count_word() to find overlap between sets
    df['word_in_title'] = df[['title_set','search_set']].apply(count_word,axis=1)
    df['word_in_description'] = df[['description_set','search_set']].apply(count_word,axis=1)
    df['brand'] = df_new['brand'].str.lower()
    df['word_in_brand'] = df[['brand','search_set']].apply(count_word,axis=1)

    df['len_search'] = df['search_set'].apply(lambda x:len(x))
    df['len_brand'] = df['brand'].map(lambda x:len(x.split()))

    df['ratio_title'] = df['word_in_title']/df['len_search']
    df['ratio_description'] = df['word_in_description']/df['len_search']
    #for brand, ratio by len_brand instead of len_search
    df['ratio_brand'] = df['word_in_brand']/df['len_brand']
    #find whole query in product_description
    df['query_in_description'] = df[['product_description','search_term_correct']].apply(lambda x: x[0].count(x[1]),axis=1)

    df['search_term_digit'] = df["search_term"].str.split().apply(lambda x: find_digit(x))
    df = df.fillna(0) #some null in ratio columns
    return df


#count words overlap between two columns in dataframe
def count_word(df):
    return sum(word in df[0] for word in df[1])

#find if there is digit in a list of strings
def find_digit(alist):
    for item in alist:
        if item.isdigit():
            return True
    return False


#this uniform units, no stemming
def unify_units(s):
    """
    :type s: str
    :rtype: str
    """

    if isinstance(s, str):
        s = s.replace(".",". ")
        s = s.replace(". 0",".0")
        s = s.replace(". 1",".1")
        s = s.replace(". 2",".2")
        s = s.replace(". 3",".3")
        s = s.replace(". 4",".4")
        s = s.replace(". 5",".5")
        s = s.replace(". 6",".6")
        s = s.replace(". 7",".7")
        s = s.replace(". 8",".8")
        s = s.replace(". 9",".9")
        s = s.replace("  "," ")

        s = s.replace("'","in.")
        s = s.replace("inches","in.")
        s = s.replace("inch","in.")
        s = s.replace(" in ","in. ")
        s = s.replace(" in.","in.")

        s = s.replace("''","ft.")
        s = s.replace(" feet ","ft. ")
        s = s.replace("feet","ft.")
        s = s.replace("foot","ft.")
        s = s.replace(" ft ","ft. ")
        s = s.replace(" ft.","ft.")

        s = s.replace(" pounds ","lb. ")
        s = s.replace(" pound ","lb. ")
        s = s.replace("pound","lb.")
        s = s.replace(" lb ","lb. ")
        s = s.replace(" lb.","lb.")
        s = s.replace(" lbs ","lb. ")
        s = s.replace("lbs.","lb.")

        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")

        s = s.replace(" sq ft","sq.ft. ")
        s = s.replace("sq ft","sq.ft. ")
        s = s.replace("sqft","sq.ft. ")
        s = s.replace(" sqft ","sq.ft. ")
        s = s.replace("sq. ft","sq.ft. ")
        s = s.replace("sq ft.","sq.ft. ")
        s = s.replace("sq feet","sq.ft. ")
        s = s.replace("square feet","sq.ft. ")

        s = s.replace(" gallons ","gal. ")
        s = s.replace(" gallon ","gal. ")
        s = s.replace("gallons","gal.")
        s = s.replace("gallon","gal.")
        s = s.replace(" gal ","gal. ")
        s = s.replace(" gal","gal.")

        s = s.replace("ounces","oz.")
        s = s.replace("ounce","oz.")
        s = s.replace(" oz.","oz. ")
        s = s.replace(" oz ","oz. ")

        s = s.replace("centimeters","cm.")
        s = s.replace(" cm.","cm.")
        s = s.replace(" cm ","cm. ")

        s = s.replace("milimeters","mm.")
        s = s.replace(" mm.","mm.")
        s = s.replace(" mm ","mm. ")

        s = s.replace("°","deg. ")
        s = s.replace("degrees","deg. ")
        s = s.replace("degree","deg. ")

        s = s.replace("volts","volt. ")
        s = s.replace("volt","volt. ")

        s = s.replace("watts","watt. ")
        s = s.replace("watt","watt. ")

        s = s.replace("ampere","amp. ")
        s = s.replace("amps","amp. ")
        s = s.replace(" amp ","amp. ")

        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = s.replace("  "," ")
        s = s.replace("..",".")

        return s.lower()

    else:

        return "null"


#tokenize and stem long string; call unify_units first; with options to remove digits, remove stopwords, stemming, and use regular expression
def tokenize(s, stem=True, digit=False, stop=True, use_re=False):
    """
    :type s: str
    :type stem: bool
    :type use_re: bool
    :rtype: set(str)
    """
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    table = string.maketrans("","")

    if use_re:
        s = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', s)

    if digit:
        tokens = set(word_tokenize(unify_units(s).translate(table, string.punctuation + string.digits)))
    else:
        tokens = set(word_tokenize(unify_units(s).translate(table, string.punctuation)))

    if stop:
        tokens = set(word for word in tokens if word not in stop_words)

    if stem:
        tokens = set(stemmer.stem(word) for word in tokens)

    return tokens


#make vocabulary set,remove punctuation, digits and stopwrods
def vocab_set(df,col):
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    table = string.maketrans("","")

    tokens =[word_tokenize(s.lower().translate(table, string.punctuation+ string.digits)) for s in df[col]]
    flats = list(itertools.chain(*tokens))
    vocab = set(flats)
    return set.difference(vocab,set(stop_words)),flats


#function to correct spelling for search_term
def correct_spelling(s):
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()
    table = string.maketrans("","")

    tokens =word_tokenize(s.lower().translate(table, string.punctuation))
    res=[]
    for word in tokens:
        if word in bad_dic:
            res.append(bad_dic[word])
        else:
            res.append(word)
    return ' '.join(x for x in res)



if __name__ == '__main__':
    #reading data and merge train/test into df_all
    df_train = pd.read_csv('train-2.csv')
    df_attributes = pd.read_csv('attributes.csv')
    df_description = pd.read_csv('product_descriptions.csv')
    df_test = pd.read_csv('test-2.csv')
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

    #correct spelling, and make stemmed sets of words
    df_all, df_description = clean_up(df_all,df_description)

    #rename 'brand' for df_attributes
    df_brand = df_attributes[df_attributes.name == "MFG Brand Name"][["product_uid", 'value']].rename(columns={"value": "brand"})

    #join df_description, df_brand with df_all to create df_new
    df_new = pd.merge(df_all, df_brand, how='left', on='product_uid')
    df_new = pd.merge(df_new, df_description, how='left', on='product_uid')
    df_new.fillna('NaN',inplace=True) #need to fillna for brand

    #feature engineering for df_new
    df_new = feature_engineering(df_new)
    df_new.to_pickle('df_new.pkl')
