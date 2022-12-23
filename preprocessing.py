'''Data preprocessing.'''

import sys
import os
import pandas as pd
from recommandation_engine_2023 import cleaning
from recommandation_engine_2023 import computations

def main():
    '''
    Preprocess the data.

    >>> main()

    '''
    #Getting data about the Roahl Doah's literature.
    path = sys.argv[0]
    path = os.path.dirname(__file__)
    charlie_frame = pd.read_csv(path + '/data/topics.csv')
    charlie_frame = charlie_frame[['Title','All names',
                                        'Topic', 'Genre',
                                        'Date of creation/publication']][:1000]

    #Renaming the frame's columns and cleaning collected data.
    charlie_frame = charlie_frame.rename(columns = {'Title':'title','All names':'authors',
                                'Topic':'desc','Genre':'genre',
                                'Date of creation/publication':'criteria'})
    charlie_frame = charlie_frame.drop_duplicates(subset = ['title'])
    charlie_frame = charlie_frame.dropna()
    charlie_frame = cleaning.clean_empty(charlie_frame)

    #Getting and renaming datasets with books and films.
    books_frame = pd.read_csv(path + '/data/book_data.csv')
    books_frame = books_frame[['book_title','book_authors', 'book_desc',
                               'genres','book_rating',]][:500]

    films_frame = pd.read_csv('films_imdb.csv')
    films_frame = films_frame.rename(columns = {'Title':'title','People':'authors',
                                'Desc':'desc','Genre':'genre',
                                'Date':'criteria'})
    books_frame = books_frame.rename(columns = {'book_title':'title','book_authors':'authors',
                                'book_desc':'desc', 'genres':'genre',
                                'book_rating':'criteria',})

    #Dropping rows with Nan values and cleaning the datasets.
    books_frame = books_frame.dropna(how='all')
    films_frame = films_frame.dropna(how='all')
    books_frame = books_frame.drop(260)
    books_frame = cleaning.clean_empty(books_frame)
    films_frame = cleaning.clean_empty(films_frame)
    films_frame = films_frame.drop(columns = ['Unnamed: 0'])

    #Concatting datasets with books together and resetting indexes.
    all_books_frame = pd.concat([books_frame,charlie_frame])
    all_books_frame = all_books_frame.reset_index()
    films_frame = films_frame.reset_index()

    #Creating a new column with concatted info from all other columns.
    films_frame['text'] = films_frame['title'] + ' ' + films_frame['authors'] + ' ' \
                                + films_frame['desc'] + ' ' + films_frame['genre']
    all_books_frame['text'] = all_books_frame['title'] + ' ' +  all_books_frame['authors'] + ' ' \
                                    + all_books_frame['desc'] + ' ' + all_books_frame['genre']

    #Dropping unneeded columns.
    films_frame = films_frame.drop(columns = ['genre','authors'])
    all_books_frame = all_books_frame.drop(columns = ['genre','authors'])
    all_books_frame.at[9, 'desc']  = 'None'

    #Creating a dataframe with unprocessed titles.
    title = pd.DataFrame({'pure_title_films':films_frame['title'].copy(),
                            'pure_books_titles':all_books_frame['title'].copy()})

    #Filing up empty columns.
    films_frame = cleaning.clean_up(films_frame, columns = ['title','desc','text'])
    all_books_frame = cleaning.clean_up(all_books_frame, columns = ['title','desc','text'])
    all_books_frame.to_csv('all_books_frame.csv')
    films_frame.to_csv('films_frame.csv')

    #Converting data to TF-IDF vectors
    films_frame_tfidf = computations.dataframe_to_tfidf(films_frame.copy())
    books_frame_tfidf = computations.dataframe_to_tfidf(all_books_frame.copy())

    #Saving data into files.
    all_books_frame.to_csv('all_books_frame.csv')
    films_frame.to_csv('films_frame.csv')
    books_frame_tfidf.to_csv('books_frame_tfidf.csv')
    films_frame_tfidf.to_csv('films_frame_tfidf.csv')

    #Adding tittle column.
    books_frame_tfidf['title_pure'] = title['pure_books_titles']
    films_frame_tfidf['title_pure'] = title['pure_title_films']

    #Computing similarity of films and books on desc and title.
    books_films_similarity_td = computations.similarity_between_frames(books_frame_tfidf,
                                                            films_frame_tfidf,
                                                            'desc', 'title')
    #Computing similarity of films and books on text and text.
    books_films_similarity_txtx = computations.similarity_between_frames(books_frame_tfidf,
                                                            films_frame_tfidf,
                                                            'text', 'text')
    #Computing similarity of films and books on title and title.
    books_films_similarity_ttl = computations.similarity_between_frames(books_frame_tfidf,
                                                            films_frame_tfidf,
                                                            'title', 'title')

    #Setting up needed indexes.
    books_films_similarity_td['title'] = books_frame_tfidf['title_pure']
    books_films_similarity_td = books_films_similarity_td.set_index('title')

    books_films_similarity_txtx['title'] = books_frame_tfidf['title_pure']
    books_films_similarity_txtx = books_films_similarity_txtx.set_index('title')

    books_films_similarity_ttl['title'] = books_frame_tfidf['title_pure']
    books_films_similarity_ttl = books_films_similarity_ttl.set_index('title')

    #Saving similarities into a file.
    books_films_similarity_td.to_csv('books_films_similarity_td.csv')
    books_films_similarity_txtx.to_csv('books_films_similarity_txtx.csv')
    books_films_similarity_ttl.to_csv('books_films_similarity_ttl.csv')

if __name__ == '__main__':
    main()
