import pandas as pd
import numpy as np
from module.hash import *
from collections import defaultdict

def find_two_most_similar_users(query_user, candidates, user_ids, user_movies_dict):
    '''
    Function that finds the two users most similar to a given query user
    Inputs:
    - query_user (int): user ID of the query user
    - candidates (defaultdict): defaultdict containing the candidates of each user
    - user_ids (ndarray of integers): user IDs
    - user_movies_dict (defaultdict): defaultdict containing the virtual rows of the movies each user has watched
    Outputs:
    - two_most_similar_users (list): list of the two most similar users to the query user, with similarity scores
    '''
    user_candidates = candidates[query_user] # get the candidates of the query user
    similarity = lambda user: jaccard_similarity(set(user_movies_dict[query_user]), set(user_movies_dict[user])) # lambda function that computes the jaccard similarity between two users

    # Sort candidates by similarity, and always pick the top two
    sorted_candidates = sorted(user_candidates, key=similarity, reverse=True) # get the user with the highest jaccard similarity
    most_similar = sorted_candidates[0]
    second_most_similar = sorted_candidates[1]

    # List of the two most similar users
    two_most_similar_users = [
        (most_similar, jaccard_similarity(set(user_movies_dict[query_user]), set(user_movies_dict[most_similar]))),
        (second_most_similar, jaccard_similarity(set(user_movies_dict[query_user]), set(user_movies_dict[second_most_similar])))
    ]

    # create dataframe of the two most similar users with their similarities to the input user
    two_most_similar_df = pd.DataFrame(two_most_similar_users, columns = ['User', 'Similarity'])
    return two_most_similar_df

def get_rated_movies(user, rating):
    '''
    Function that retrieves all the rated movies of a user and the rating they gave it
    Inputs:
    - user (int): user ID
    - rating (DataFrame): DataFrame containing information about userId, movieId, rating and timestamps
    Outputs:
    - rated_movies (list): list of tuples containing the movieId and the rating the user gave it
    '''
    # Restrict the dataframe to the rows where the user ID appears
    rated_movies_df = rating[rating['userId']==user][['movieId', 'rating']]
    # Turn the DataFrame into a list of tuples (movieId, rating)
    rated_movies = list(zip(rated_movies_df['movieId'], rated_movies_df['rating']))
    return rated_movies

def rated_movies_intersection(user1, user2, rating):
    '''
    Function that finds the rated movies that two users have rated in common
    Inputs:
    - user1 (int): user ID of the first user
    - user2 (int): user ID of the second user
    - rating (DataFrame): DataFrame containing information about userId, movieId, rating and timestamps
    Outputs:
    - movie_recs (list): list of tuples containing the movieId and the rating the user gave it
    '''
    # Get movies rated by the two users and the rating
    rated_movies1 = get_rated_movies(user1, rating)
    #print(sorted(rated_movies1, key=lambda x: x[0]))
    rated_movies2 = get_rated_movies(user2, rating)
    #print(sorted(rated_movies2, key=lambda x: x[0]))
    # Just the movies rated by the users
    movies1 = set([movie[0] for movie in rated_movies1])
    movies2 = set([movie[0] for movie in rated_movies2])
    # Get intersection of movies
    movies_intersection = list(movies1.intersection(movies2))
    # Initialize movie recommendations list
    movie_recs = []
    for movie in movies_intersection:
        # Get the rating of the movie (perhaps some users rated the same movie twice)
        rating1 = [mov[1] for mov in rated_movies1 if mov[0]==movie]
        rating2 = [mov[1] for mov in rated_movies2 if mov[0]==movie]
        # Get the average rating
        avg_rating = (np.mean(rating1) + np.mean(rating2))/2
        movie_recs.append((movie, avg_rating))
    return movie_recs

def recommend_movies(query_user, candidates, user_ids, user_movies_dict, rating, movie_df):
    '''
    Function that recommends movies based on a user's two most similar users and their rated movies
    Inputs:
    - query_user (int): user ID of the query user
    - candidates (defaultdict): defaultdict containing the candidates of each user
    - user_ids (ndarray of integers): user IDs
    - user_movies_dict (defaultdict): defaultdict containing the virtual rows of the movies each user has watched
    Outputs:
    - recommended_movies (DataFrame): movie ID(S) and rating(s) of the recommended movie(s)
    '''
    if query_user not in user_ids:
        print('User not found')
        return None

    # Get the two most similar users to the query user
    two_most_similar = find_two_most_similar_users(query_user, candidates, user_ids, user_movies_dict)
    userId1, userId2 = two_most_similar['User'][0], two_most_similar['User'][1]

    # Find the intersection of movies and average ratings
    movies_intersection = rated_movies_intersection(userId1, userId2, rating)

    # Remove from these movies the ones seen by the query user
    movies_intersection = [(movie[0], movie[1]) for movie in movies_intersection if movie[0] not in user_movies_dict[query_user]]

    # If the intersection is not empty, return a DataFrame with the recommended movies
    if movies_intersection:
        # Replace movie Id with the movie name using the movie_df DataFrame
        movies_intersection = [(movie_df[movie_df['movieId']==movie[0]]['title'].values[0], movie[1]) for movie in movies_intersection]
        movies_intersection_df = pd.DataFrame(movies_intersection, columns = ['movie', 'rating'])
        movies_intersection_df.sort_values(by='rating', ascending=False, inplace=True) # Sort the DataFrame by average rating

        # If there are at least five movies in the intersection, return these five
        if len(movies_intersection_df) >= 5:
            return movies_intersection_df.iloc[:5]

        # If the two most similar users have less than five movies in common
        else: # Fill the remaining recommendations with top rated movies from the most similar user
            top_rated = get_rated_movies(userId1, rating)
            top_rated = [(movie[0], movie[1]) for movie in top_rated if movie[0] not in user_movies_dict[query_user]] # Eliminate the movies that the user has seen
            top_rated.sort(key=lambda x: x[1], reverse=True) # Sort the movies by rating
            num_movies_from_most_similar = 5 - len(movies_intersection_df) # Return five movie recommendations, the ones movie_intersection_df plus movies from the most similar user until we get to five movies
            top_rated = top_rated[:num_movies_from_most_similar] # Restrict number of top rated movies from most similar user to num_movies_from_most_similar
            top_rated = [(movie_df[movie_df['movieId']==movie[0]]['title'].values[0], movie[1]) for movie in top_rated] # Replace movie Id with the movie name using the movie_df DataFrame
            movies_intersection_df = pd.concat([movies_intersection_df, pd.DataFrame(top_rated[:num_movies_from_most_similar], columns = ['movie', 'rating'])]) # Concatenate movies in the intersection with movies rated by the most similar user
            movies_intersection_df.sort_values(by='rating', ascending=False, inplace=True)  # Sort movies again because we concatenated two dfs
            return movies_intersection_df

    else:
        top_rated = get_rated_movies(userId1, rating)
        top_rated.sort(key=lambda x: x[1], reverse=True)
        return top_rated[:5]