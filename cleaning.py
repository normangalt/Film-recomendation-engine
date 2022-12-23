'''
Cleaning functions.
'''

from nltk.corpus import stopwords

def to_lower_case(sentence):
    '''
    Returns a string in lower case.

    >>> to_lower_case('AAAA')
    'aaaa'
    >>> to_lower_case('AaAA')
    'aaaa'
    '''
    return sentence.lower()

def remove_non_lettersdigits(sentence):
    '''
    Retuns a string with only digits and alphabet letters.

    >>> remove_non_lettersdigits('AaAA%$$ &&gtYY45')
    ' a        gt  45'
    '''
    filtered_sentence = ''
    for letter in sentence:
        letter_ord = ord(letter)
        if letter_ord in range(97, 123) \
                      or letter_ord == 32 \
                      or letter_ord in range(48, 59):
            filtered_sentence += letter
        else:
            filtered_sentence += ' '
    return filtered_sentence

def remove_stop_words(sentence):
    '''
    Retuns a string without stop words.

    >>> from nltk.corpus import stopwords
    >>> example = 'I love python and math, it will have any effect that\
                    could be achieved technically with schizophrenia and\
                    boring doctests'.split()
    >>> stop_words = set(stopwords.words('english'))
    >>> remove_stop_words(example)
    ['I', 'love', 'python', 'math,', \
'effect', 'could', 'achieved', \
'technically', 'schizophrenia', \
'boring', 'doctests']
    '''
    stopwords_words = stopwords.words('english')
    filtered_sentence = [word for word in sentence
                         if not word in stopwords_words]
    return filtered_sentence

def clean_up(dataframe, columns):
    '''
    Retuns a dataframe with only digits and alphabet letters.

    >>> import pandas as pd
    >>> frame = pd.DataFrame({'col':['SHARK caG%$£mel shark \
and camel will eat shark kamel and that will \
disapoint Steve so it won\\'t have any \
propper effect on economics in Belgia.']})
    >>> clean_up(frame,['col'])['col'][0]
    ['shark', 'cag', 'mel', 'shark', 'camel', 'eat', \
'shark', 'kamel', 'disapoint', 'steve', 'propper', \
'effect', 'economics', 'belgia']
    '''
    def if_empty(string: str):
        '''
        Returns a splitted string if it is not empty
and a list with 'None' otherwise.

        >>> if_empty([1,2,3])
        [1,2,3]
        >>> if_empty([])
        []
        '''
        if len(string) == 0:
            return ['None']
        array = string.split()
        return array

    for column in columns:
        dataframe[column] = dataframe[column].apply(func = to_lower_case)
        dataframe[column] = dataframe[column].apply(func = remove_non_lettersdigits)
        dataframe[column] = dataframe[column].apply(func = remove_non_lettersdigits)
        dataframe[column] = dataframe[column].apply(if_empty)
        if column != 'title':
            dataframe[column] = dataframe[column].apply(func = remove_stop_words)

    return dataframe

def clean_empty(charlie_frame):
    '''
    Returns a given dataframe with rows with null values
replaced by 'None'.

    >>> import pandas as pd
    >>> import numpy as np
    >>> frame = pd.DataFrame({'authors':[np.nan,'gt','Shark','Nan'],\
'desc':[np.nan,'gt','Shark','Nan'],\
'genre':[np.nan,'gt','Shark','Nan'],\
'criteria':[np.nan,'gt','Shark','Nan'],\
'title':[np.nan,'gt','Shark','Nan']})
    >>> clean_empty(frame).to_numpy()
    array([['None', 'None', 'None', 'None', 'None'],
           ['gt', 'gt', 'gt', 'gt', 'gt'],
           ['Shark', 'Shark', 'Shark', 'Shark', 'Shark'],
           ['Nan', 'Nan', 'Nan', 'Nan', 'Nan']], dtype=object)
    '''
    not_genres = list(charlie_frame['genre'].isnull())
    charlie_frame[not_genres] = 'None'
    not_genres = charlie_frame['authors'].isnull()
    charlie_frame['authors'].loc[not_genres] = 'None'
    not_genres = charlie_frame['desc'].isnull()
    charlie_frame['desc'].loc[not_genres] = 'None'
    not_genres = charlie_frame['desc'] == ''
    charlie_frame['desc'].loc[not_genres] = 'None'
    not_genres = charlie_frame['criteria'].isnull()
    charlie_frame['criteria'].loc[not_genres] = 'None'
    not_genres = charlie_frame['title'].isnull()
    charlie_frame['title'].loc[not_genres] = 'None'
    return charlie_frame
