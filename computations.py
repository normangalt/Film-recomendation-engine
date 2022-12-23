'''
Contains functions for computing Word Mover's Distance.
'''
import numpy as np
import pandas as pd

def term_frequence(word, documents):
    '''
    Returns an array of term frequences for a given word
in given documents.

    >>> term_frequence('shark', [['shark','shark'],['shark','shark', 'camel',\
                                'camel','camel'],['camel','camel','camel',\
                                'camel','camel']])
    array([2., 2., 0.])
    >>> term_frequence('camel', [['shark','shark'],['shark','shark', 'camel',\
                                'camel','camel'],['camel','camel','camel',\
                                'camel','camel']])
    array([0., 3., 5.])
    '''

    frequences = np.array([])
    for document in documents:
        frequence = document.count(word)
        frequences = np.append(frequences, frequence)
    return frequences

def inverse_document_frequence(frequences):
    '''
    Return an array of inverse document frequences for a given word
in given documents.

    >>> import numpy as np
    >>> inverse_document_frequence(np.array([2, 5, 7, 8, 21, 43, 0, 0 ,0 ]))
    0.4054651081081644
    >>> inverse_document_frequence(np.array([0.5, 0.3, 0.7, 0.8999, 23]))
    0.0
    '''

    number_of_documents = len(frequences)
    number_of_documents_with_word = len(frequences[frequences > 0])
    inverse_document_frequence_value = np.log(number_of_documents
                                        / (number_of_documents_with_word))
    return inverse_document_frequence_value

def word_to_tf_idf(word, frequence, frequences):
    '''
    Returns a word as a TF-IDF value.

    >>> frequence = term_frequence('shark', [['shark','shark'],['shark','shark', 'camel',\
                                             'camel','camel'],['camel','camel','camel',\
                                             'camel','camel']])
    >>> word_to_tf_idf('shark', 0.7, frequence)
    0.28382557567571504
    '''

    if word == 'None':
        return 0

    inverse_document_frequence_value = inverse_document_frequence(frequences)
    tf_idf_value = inverse_document_frequence_value*(frequence)
    return tf_idf_value

def document_to_tf_idf(document, documents):
    '''
    Returns a document as a TF-IDF vector.

    >>> import pandas as pd
    >>> frame = pd.DataFrame({'col':['shark','camel','shark',\
                                 'camel','shark','kamel']})
    >>> document_to_tf_idf('kamel',frame['col'])
    array([[0.40546511],
           [0.        ],
           [0.69314718],
           [0.69314718],
           [0.69314718]])
    '''

    document_vector = np.array([])
    setdocuments = documents.apply(set)
    index = setdocuments == set(document)
    index = documents[index]
    index = index.index[0]
    for word in document:
        frequences = term_frequence(word, documents)
        frequence = frequences[index]
        tf_idf_value = word_to_tf_idf(word, frequence, frequences)
        document_vector = np.append(document_vector, tf_idf_value)
    document_vector = document_vector.reshape((document_vector.shape[0],1))
    return document_vector

def cosine_similarity(sentence1, sentence2) -> float:
    '''
    Returns a number that shows how similar words are.

    >>> import pandas as pd
    >>> frame = pd.DataFrame({'col':['shark','camel',\
                                     'shark','camel',\
                                     'shark','kamel',\
                                     'james']})
    >>> sen1 = document_to_tf_idf('kamel', frame['col'])
    >>> sen2 = document_to_tf_idf('james', frame['col'])
    >>> round(cosine_similarity(sen1, sen2),2)
    0.78
    '''

    lenght1 = sentence1.shape[0]
    lenght2 = sentence2.shape[0]
    if lenght1 > lenght2:
        sentence2 = np.pad(sentence2, [(0, lenght1 - lenght2)], mode = 'constant')
    elif lenght1 < lenght2:
        sentence1 = np.pad(sentence1, [(0, lenght2 - lenght1)], mode = 'constant')
    pairwise_product_sum = np.sum(np.dot(sentence1.T, sentence2))

    module1 = np.sqrt(np.sum(np.power(sentence1,2)))
    module2 = np.sqrt(np.sum(np.power(sentence2,2)))
    cosine = pairwise_product_sum / (module1 * module2)
    return cosine


def dataframe_to_tfidf(charlie_frame):
    '''
    Returns dataframe with data trandsformed into tf_idf vectors.

    >>> import pandas as pd
    >>> frame = pd.DataFrame({'title':['shark','camel',\
                                     'shark','camel',\
                                     'shark','kamel',\
                                     'james'],\
                    'desc':['shark','camel',\
                                     'shark','camel',\
                                     'shark','kamel',\
                                     'james'],\
                    'text':['shark','camel',\
                                     'shark','camel',\
                                     'shark','kamel',\
                                     'james']})
    >>> frame = dataframe_to_tfidf(frame)
    >>> frame['title'][0]
    array([[0.55961579],
           [0.84729786],
           [0.        ],
           [0.84729786],
           [0.55961579]])
    >>> frame['desc'][1]
    array([[1.25276297],
           [0.        ],
           [0.55961579],
           [0.55961579],
           [0.84729786]])
    >>> frame['text'][5]
    array([[0.55961579],
           [0.        ],
           [0.55961579],
           [0.55961579],
           [0.84729786]])
    '''

    charlie_frame['title'] = charlie_frame['title'].apply(lambda cell:document_to_tf_idf
                                                (cell, charlie_frame['title']))
    charlie_frame['desc'] = charlie_frame['desc'].apply(lambda cell:document_to_tf_idf
                                                (cell, charlie_frame['desc']))
    charlie_frame['text'] = charlie_frame['text'].apply(lambda cell:document_to_tf_idf
                                                (cell, charlie_frame['text']))
    return charlie_frame

def similarity_between_frames(frame1, frame2, key1, key2):
    '''
    Returns how similar content in two frames is.

    >>> import pandas as pd
    >>> frame = pd.DataFrame({'title':['shark','camel',\
                                     'shark','camel',\
                                     'shark','kamel',\
                                     'james'],\
                    'desc':['shark','camel',\
                                     'shark','camel',\
                                     'shark','kamel',\
                                     'james'],\
                    'text':['shark','camel',\
                                     'shark','camel',\
                                     'shark','kamel',\
                                     'james']})
    >>> frame3 = frame['title'].copy()
    >>> frame1 = dataframe_to_tfidf(frame)
    >>> frame2 = frame1.copy()
    >>> frame1['title_pure'] = frame3
    >>> frame2['title_pure'] = frame3
    >>> similarity_between_frames(frame1, frame2, 'desc', 'text')
          shark     camel     kamel     james
    0  1.000000  0.672885  0.682348  0.601019
    1  0.672885  1.000000  0.930731  0.953498
    2  1.000000  0.672885  0.682348  0.601019
    3  0.672885  1.000000  0.930731  0.953498
    4  1.000000  0.672885  0.682348  0.601019
    5  0.682348  0.930731  1.000000  0.782295
    6  0.601019  0.953498  0.782295  1.000000
    '''

    films = pd.DataFrame()
    for index in range(len(frame2)):
        films[frame2.iloc[index]['title_pure']] = frame1[key1].apply(
                                                    func = lambda cell:cosine_similarity
                                                    (cell, frame2.iloc[index][key2]))
    return films

def find_best_matches(similarities, title):
    '''
    Returns a list of books that best match
the films from the user's history.

    >>> import pandas as pd
    >>> data = pd.DataFrame(data = {'title':['The prince','C for dummies',\
                            'Python as a way up in the social hierarchy',\
                            'Big bad computer science'],\
                            'No way around the corner':[0.5, 0.9, 0.567, 0.3456]})
    >>> find_best_matches(data, 'No way around the corner')[0]
    'C for dummies'
    '''

    results = similarities[title]
    results = [similarities.iloc[name]['title']
               for name in results.nlargest(5).index.to_list()]

    return results
