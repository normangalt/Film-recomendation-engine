'''
The module scrapes top 250 films on imdb.
'''

import requests
from bs4 import BeautifulSoup
import pandas as pd

def main():
    '''
    Gets data about films.

    >>> main()

    '''
    sites = ['https://www.imdb.com/chart/top/?ref_=nv_mv_250']
    films_dataframe = pd.DataFrame({'Title':[], 'People':[],
                                        'Desc':[],'Genre':[],'Date':[]})
    for site in sites:
        #Requesting site with top 250 films and aprsing it. Getting needed content.
        imdb_site = requests.get(site)
        soup = BeautifulSoup(imdb_site.content, 'html.parser')
        table = soup.find_all('td', class_ = 'titleColumn')
        table = [list(film.children)[1] for film in table]
        table = [film.get('href') for film in table]
        table = ['https://www.imdb.com/'+film for film in table]

        titles = []
        humanity = []
        descriptions = []
        genre = []
        dates = []
        #Looping over links for the films and collecting
    #needed data
        for link in table:
            page = requests.get(link)
            soup = BeautifulSoup(page.content, 'html.parser')
            book_title = soup.find_all('div', class_ ='originalTitle')
            if book_title == []:
                book_title = str(soup.title)
                book_title = book_title[book_title.find('<title>')+7:book_title.find(' (')]
            else:
                book_title = book_title[0].text
            book_title = book_title[:book_title.find(' (original title)')].strip()

            #Getting description of the film.
            summary = soup.find_all('div', class_ ='summary_text')
            summary = [child.text for child in summary]
            summary = summary[0].strip()

            #Getting info about people involved in film creation.
            people = soup.find_all('div', class_ ='credit_summary_item')
            people = [human.text for person in people for human in person.find_all('a')]
            subtext = soup.find_all('div', class_ ='subtext')
            subtext = subtext[0].find_all('a')
            subtext = [line.text for line in subtext]

            #Getting data from the subtext array.
            date = subtext.pop()
            try:
                date = date.split(' ')[2]
            except IndexError:
                date = date.split(' ')[0]

            titles.append(book_title)
            dates.append(date)
            genre.append('|'.join(subtext))
            descriptions.append(summary)
            humanity.append('|'.join(people))

        #Creating a dataframe with data collected from a site and concating it
    #with the a dataframe for all films.
        films_dataframe_part = pd.DataFrame({'Title':titles, 'People':humanity,
                                        'Desc':descriptions,'Genre':genre,'Date':dates})
        films_dataframe = pd.concat([films_dataframe,films_dataframe_part])

    films_dataframe.to_csv('films_imdb.csv')

if __name__ == '__main__':
    main()
