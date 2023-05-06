import numpy as np
import pandas as pd

import nltk
from nltk.corpus import wordnet
import gensim
from tqdm import tqdm

import unicodedata
import re
from textblob import Word
import spacy
from bs4 import BeautifulSoup

from contractions import CONTRACTION_MAP



nlp = spacy.load('en_core_web_sm')
stopword_list = nltk.corpus.stopwords.words('english')



def set_default_pandas_options(max_columns=10, max_rows=2000, width=1000, max_colwidth=50):
    
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.width', width)
    pd.set_option('max_colwidth', max_colwidth)



def write_html_to_file(filename, html):
    
    f = open(filename,'w')
    f.write(html)
    f.close()



def data_quality_report(df):
    
    if isinstance(df, pd.core.frame.DataFrame):
        
        descriptive_statistics = df.describe(include = 'all')
        data_types = pd.DataFrame(df.dtypes, columns=['Data Type']).transpose()
        missing_value_counts = pd.DataFrame(df.isnull().sum(), columns=['Missing Values']).transpose()
        present_value_counts = pd.DataFrame(df.count(), columns=['Present Values']).transpose()
        data_report = pd.concat([descriptive_statistics, data_types, missing_value_counts, present_value_counts], axis=0)
        
        return data_report
    
    else:
    
        return None



# text normalizer

def tokenize_text_to_sentences(text):
    
    sentences = nltk.sent_tokenize(text)
    
    return sentences



def tokenize_sentence_to_words(sentence):
    
    words = nltk.word_tokenize(sentence)
    
    return words



def strip_html_tags(text):
    
    soup = BeautifulSoup(text, "html.parser")
    
    if bool(soup.find()):
        
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
        
    else:
        
        stripped_text = text
    
    return stripped_text



def remove_accented_chars(text):

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    return text



def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    
    return expanded_text



def remove_special_characters(text, remove_digits=False):
    
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    text = text.replace('[', '').replace(']', '')
    
    return text



def remove_repeated_characters(tokens):
    
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    
    def replace(old_word):
        
        if wordnet.synsets(old_word):
            
            return old_word
            
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
    
    correct_tokens = [replace(word) for word in tokens]
    
    return correct_tokens



def correct_spelling(word_tokens):
    
    for i in range(len(word_tokens)):
    
        w = Word(word_tokens[i])
        word_tokens[i] = str(w.correct())
    
    return word_tokens



def lemmatize_tokens(tokens):
    
    for index in range(len(tokens)):
        
        tokens[index] = nlp(tokens[index])                

        if tokens[index][0].lemma_ != '-PRON-':

            tokens[index] = tokens[index][0].lemma_
        
        else:
        
            tokens[index] = tokens[index][0].text
    
    return tokens



def remove_stopword(tokens, is_lower_case=False):
    
    for index in range(len(tokens)):    
        
        if is_lower_case:

            if tokens[index] in stopword_list:

                tokens[index] = ''

        else:

            if tokens[index].lower() in stopword_list:

                tokens[index] = ''
    
    return tokens



def exclude_stopwords(stopword_exclusion_list):
    
    for exclude in stopword_exclusion_list:
        
        stopword_list.remove(exclude)



def normalize_corpus(dataframe, raw_column, clean_column,
                        html_stripping=False,
                        accented_char_removal=True, contraction_expansion=True,
                        text_lower_case=True, extra_newlines_removal=True, extra_whitespace_removal=True,
                        special_char_removal=True, remove_digits=True, repeating_char_removal=True,
                        spelling_correction=True, lemmatize=True, stop_word_removal=True):
    
    dataframe[clean_column] = ''
    
    for i in range(len(dataframe)):
        
        text = dataframe.loc[i, raw_column]
        
        if html_stripping:
            
            text = strip_html_tags(text)
            
        if accented_char_removal:
            
            text = remove_accented_chars(text)
        
        if contraction_expansion:
            
            text = expand_contractions(text)
        
        if text_lower_case:
            
            text = text.lower()
        
        if extra_newlines_removal:
            
            text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        
        if extra_whitespace_removal:
            
            text = re.sub(' +', ' ', text)
        
        if special_char_removal:
            
            text = remove_special_characters(text, remove_digits)
            
        # tokenize into words
        word_tokens = tokenize_sentence_to_words(text)
        
        if repeating_char_removal:
            
            word_tokens = remove_repeated_characters(word_tokens)
            
        if spelling_correction:
            
            word_tokens = correct_spelling(word_tokens)
        
        if lemmatize:
            
            word_tokens = lemmatize_tokens(word_tokens)
        
        if stop_word_removal:
            
            word_tokens = remove_stopword(word_tokens, text_lower_case)
        
        word_tokens = [word_token for word_token in word_tokens if word_token != '']
        text = ' '.join(word_tokens)
        
        dataframe.loc[i, clean_column] = text
    
    return dataframe



# word2vec embeddings

def average_word_vectors(words, model, vocabulary, num_features):

    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    
    for word in words:
        
        if word in vocabulary:
            
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])
    
    if nwords:
        
        feature_vector = np.divide(feature_vector, nwords)
    
    return feature_vector



def averaged_word_vectorizer(corpus, model, num_features):

    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) 
        for tokenized_sentence in corpus]
    
    return np.array(features)



# topic modelling

def topic_model_coherence_generator(model_name,
                                    corpus, texts, dictionary, 
                                    start_topic_count=2, end_topic_count=3, step=1, cpus=1,
                                    print_topics=False):
    
    models = []
    coherence_scores = []

    for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
        
        print('\nMODEL: {} - NUMBER OF TOPICS: {}'.format(model_name, topic_nums))
        
        if model_name == 'LSI':

            model = gensim.models.LsiModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=topic_nums,
                                           chunksize=1740,
                                           power_iters=1000)
        
        elif model_name == 'LDA':

            model = gensim.models.LdaModel(corpus=corpus, 
                                           id2word=dictionary,
                                           num_topics=topic_nums,
                                           chunksize=1740,
                                           alpha='auto',
                                           eta='auto',
                                           iterations=500)
        
        cv_coherence_model = gensim.models.CoherenceModel(model=model,
                                                          corpus=corpus,
                                                          texts=texts,
                                                          dictionary=dictionary,
                                                          coherence='c_v',
                                                          processes=cpus)
        
        
        
        coherence_score = cv_coherence_model.get_coherence()
        
        coherence_scores.append(coherence_score)
        models.append(model)
        
        
        
        if print_topics:

            for topic_id, topic in model.print_topics(num_topics=10, num_words=20):

                print('Topic #'+str(topic_id+1)+':')
                print('='*10)
                print(topic)
                print()

            print('-'*50)


    
    return models, coherence_scores
