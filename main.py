'''
Recomandation.
'''
import sys
from tabulate import tabulate
import pandas as pd
from recommandation_engine_2023 import computations

def main():
    '''
    Recommands books.

    >>> main()

    '''
    file_path = input('Введіть шлях до файлу історії: ')

    books_films_similarity_td = pd.read_csv('books_films_similarity_td.csv')
    books_films_similarity_txtx = pd.read_csv('books_films_similarity_txtx.csv')
    books_films_similarity_ttl = pd.read_csv('books_films_similarity_ttl.csv')

    try:
        history = open(file_path, 'r', encoding='utf-8').readlines()
        history = [film[:-1] for film in history]
    except FileNotFoundError:
        print('Wrong path.')
        sys.exit()

    results_td = pd.DataFrame()
    results_txtx = pd.DataFrame()
    results_ttl = pd.DataFrame()
    general = pd.DataFrame()

    general['sum'] = books_films_similarity_td[history].sum(axis = 1) \
                + books_films_similarity_ttl[history].sum(axis = 1) \
                + books_films_similarity_txtx[history].sum(axis = 1)

    general_results = general['sum'].nlargest(5)
    general_results = pd.DataFrame(data = {'Top 5 books for you: ':
                            [books_films_similarity_ttl.iloc[index]['title']
                            for index in general_results.index]})

    for film in history:
        results_td[film] = computations.find_best_matches(books_films_similarity_td ,film)
        results_txtx[film] = computations.find_best_matches(books_films_similarity_txtx ,film)
        results_ttl[film] = computations.find_best_matches(books_films_similarity_ttl ,film)

    results = (results_td, results_txtx, results_ttl)
    analysises = ('Title - Description', 'All info - All info', 'Title - Title')
    user_request = []
    while True:
        analyse = input(f'Введіть номер критерію для фільмів - книг (число від 1 до 3),\
за яким хочете здіснити порівння, доступні критерії {analysises} або EXIT, щоб \
завершити введеня: ')

        if analyse == 'EXIT':
            break
        try:
            analyse = int(analyse)
            if analyse > 3 or analyse == 0:
                raise ValueError
            user_request.append(analyse)
        except ValueError:
            print('Значеннями повинні бути числа від 1 до 3.')

    request = input('Якщо бажаєте пропустити виведення рекомендацій для кожного з фільмів \
введіть SKIP:')
    if request != 'SKIP':
        for result in range(len(user_request)):
            for film in results[result]:
                print(f'Найкращі книги за порівнянням: \
[{analysises[user_request[result]-1]}] для фільму {film}.' )
                print(tabulate(results[user_request[result]-1][film].to_frame(),
                               headers='keys', tablefmt='grid'))
                print()
                print()
                inp = input('Введіть NEXT, щоб отримати рекомендації \
для наступноги фільму або будь-що інше, щоб завершити процес: ')
                if inp != 'NEXT':
                    break
            request = input('Якщо бажаєте пропустити виведення \
наступних рекомендацій введіть SKIP:')
            if request == 'SKIP':
                break
            if result < 2 and len(user_request) > result + 1:
                inp = input(f'Введіть NEXT, щоб отримати рекомендації \
за наступим критерієм: [{analysises[result + 1]}] - або будь-що інше, щоб завершити процес:')
                if inp != 'NEXT':
                    break

    print(f'Історія перегляду: {history}')
    print()
    print()
    print('Найкращі книги для історії загалом:' )
    print(tabulate(general_results, headers='keys', tablefmt='grid'))

    with open('results.txt', 'w', encoding='utf-8') as file:
        file.write(str(general_results) + '\n')

    results_td.to_csv('results_td.csv')
    results_txtx.to_csv('results_txtx.csv')
    results_ttl.to_csv('results_ttl.csv')

if __name__ == '__main__':
    main()
