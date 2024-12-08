import numpy as np
import random
from collections import defaultdict
import pandas as pd

def linear_hash(x, a=238, b=943, p=27281):
    '''
    Use linear congruential generator to hash a positive integer (or array of integers)
    and use modulo to fit the result into the range [1, p]
    Inputs:
    - x (int or ndarray): integer or array of integers to hash
    - a (int): LCG scaling parameter
    - b (int): LCG shift parameter
    - p (int): prime
    Output:
    - int or ndarray: the hashed value(s) constrained to the range [1, p]
    '''
    return (a*x + b) % p

# -------------------------------------------------------------------------------------------------------------------- #

def bitwise_hash(x, a=238, b=4435, p=27281):
    '''
    Use bitwise hashing with XOR operator to hash a positive integer
    and use modulo to fit the result into the range [1, p]
    Inputs:
    - x (int or ndarray): integer or array of integers to hash
    - a (int): parameter
    - b (int): additive parameter
    - p (int): prime
    Output:
    - int or ndarray: the hashed value(s) constrained to the range [1, p]
    '''
    x = np.asarray(x)  # ensure input is a NumPy array
    return (np.bitwise_xor(x,a) + b) % p

# -------------------------------------------------------------------------------------------------------------------- #

def random_minhash(user_ids, user_movies_dict, k = 10, p=27281, seed=198324):
    '''
    Creates signature vector of length k for each user ID by picking random hash functions
    to hash movie IDs
    Inputs:
    - user_ids (ndarray of integers): categories we want to create the signature vectors for
    - user_movies_dict (dict): dictionary made of items of the structure userId: [movieId1, movieId2, ...]
    - p (int): large prime number (larger than max movieId)
    - k (int): length of the signature vectors
    - seed (int): seed for reproducibility in randomness
    Outputs:
    - signature: matrix of signature vectors
    - hash_functions: list of tuples (i,a,b) where 'i' identifies the chosen hash function, and 'a', 'b' are its parameters
    '''
    # Pick k random hash functions like this:
    # 1. Generate k random binary numbers (sample from {0,1}^k)
    # 2. If the i-th number is 0, use linear_hash as the i-th function -> randomly generate parameters 'a' and 'b'
    # 3. If the i-th number is 1, use bitwise_hash as the i-th function -> randomly generate parameters 'a' and 'b'

    rng = np.random.RandomState(seed) # create random number generator
    k_rand_binary = rng.randint(0,2,k) # generate k random numbers between 0 and 1
    signature = np.zeros((k, user_ids.shape[0]), dtype=int)  # Initialize matrix of signature vectors
    #print(k_rand_binary)

    # Initialize list to store information about the hash functions that will be randomly chosen
    # To compare results and find the best combinations, we can use this
    hash_functions = []

    # Create signature vectors
    for i in range(k):

        # Case: use linear_hash function
        if k_rand_binary[i] == 0:
            # Generate random parameters 'a' and 'b'
            a, b = rng.randint(1,p), rng.randint(0,p)
            hash_functions.append((0,a,b))

            # create signature values for elements i of the signature vectors
            for idx, user in enumerate(user_ids):
                hashed_movies = linear_hash(user_movies_dict[user],a,b,p)
                signature[i,idx] = np.min(hashed_movies)

        # Case: use bitwise_hash function
        if k_rand_binary[i] == 1:
            # Generate random parameters 'a' and 'b'
            a, b = rng.randint(1,p), rng.randint(0,p)
            hash_functions.append((1,a,b))

            # create signature values for elements i of the signature vectors
            for idx, user in enumerate(user_ids):
                hashed_movies = bitwise_hash(user_movies_dict[user],a,b,p)
                signature[i,idx] = np.min(hashed_movies)

    return signature, hash_functions

# -------------------------------------------------------------------------------- #


def minhash(user_ids, user_movies_dict, k_hash_function_choices, linear_parameters=None, bitwise_parameters=None, k = 10, p=27281):
    '''
    Creates signature vector of length k for each user ID given k pre-selected hash functions with
    established parameters
    Inputs:
    - user_ids (ndarray of integers): categories we want to create the signature vectors for
    - user_movies_dict (dict): dictionary made of items of the structure userId: [movieId1, movieId2, ...]
    - k_hash_function_choices (list): list of 0s and 1s representing the hash function chosen (0 for linear_hash, 1 for bitwise_hash)
    - linear_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for linear_hash
    - bitwise_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for bitwise_hash
    - p (int): large prime number (larger than max movieId)
    - k (int): length of the signature vectors
    Outputs:
    - signature: matrix of signature vectors
    '''
    signature = np.zeros((k, user_ids.shape[0]), dtype=int)  # Initialize matrix of signature vectors

    # Create signature vectors
    for i in range(k):

        # Case: use linear_hash function
        if k_hash_function_choices[i] == 0:
            # Get parameters 'a' and 'b' for linear_hash
            a, b = linear_parameters.get(i)

            # create signature values for elements i of the signature vectors
            for idx, user in enumerate(user_ids):
                hashed_movies = linear_hash(user_movies_dict[user],a,b,p)
                signature[i,idx] = np.min(hashed_movies)

        # Case: use bitwise_hash function
        if k_hash_function_choices[i] == 1:
            # Get parameters 'a' and 'b' for bitwise_hash
            a, b = bitwise_parameters.get(i)

            # create signature values for elements i of the signature vectors
            for idx, user in enumerate(user_ids):
                hashed_movies = bitwise_hash(user_movies_dict[user],a,b,p)
                signature[i,idx] = np.min(hashed_movies)

    return signature

# -------------------------------------------------------------------------------- #

def jaccard_similarity(A, B):
    '''
    Function that calculates the jaccard similarity between two sets A and B
    Inputs:
    - A (set): set 1
    - B (set): set 2
    Output:
    - int: jaccard similarity between A and B
    '''
    return len(A.intersection(B))/len(A.union(B))

# -------------------------------------------------------------------------------- #

def random_minhash_results(user_ids, user_movies_dict, k = 10, p=27281, seed=198324):
    '''
    Function that computes the signature matrix of user IDs and compares the signature vectors with
    the similarity between the original watched movie sets associated with the user IDs
    Inputs:
    - user_ids (ndarray of integers): categories we want to create the signature vectors for
    - user_movies_dict (dict): dictionary made of items of the structure userId: [movieId1, movieId2, ...]
    - k_hash_function_choices (list): list of 0s and 1s representing the hash function chosen (0 for linear_hash, 1 for bitwise_hash)
    - linear_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for linear_hash
    - bitwise_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for bitwise_hash
    - p (int): large prime number (larger than max movieId)
    - k (int): length of the signature vectors
    Outputs:
    - error (int): error rate of this configuration
    - hash_functions (list): list of tuples (i,a,b) where 'i' identifies the chosen hash function, and 'a', 'b' are its parameters
    '''
    signature_matrix, hash_functions = random_minhash(user_ids, user_movies_dict, k, seed=seed)
    N = len(user_ids) # number of users to compare
    # List of all user columns
    all_user_columns = list(range(signature_matrix.shape[1]))
    # Sampled user columns
    sampled_user_columns = random.sample(all_user_columns, N)
    # Split the sampled users into two groups
    first_group = sampled_user_columns[:(N//2)]
    second_group = sampled_user_columns[(N//2):]

    errors = [] # Initialize list to store errors, the abs differences between P(s1[i]=s2[i]) and Jaccard(user1, user2)
    prob_values = [] # Initialize list to store probability values P(s1[i]=s2[i]) as defined above
    jaccard_values = [] # Initialize list to store Jaccard similarity values between pairs of users

    # Pair-wise compare jaccard similarity of the users with their
    # probability of having corresponding signature vector elements
    for i in range(N//2):

        # Signature vectors of the current pair of users
        signature1 = signature_matrix[:,first_group[i]]
        signature2 = signature_matrix[:,second_group[i]]
        # Compare signature vectors
        prob_same_el = sum(signature1==signature2)/k
        prob_values.append(prob_same_el) # add probability to prob_values list

        # User ID of the current user
        user1 = user_ids[first_group[i]]
        user2 = user_ids[second_group[i]]
        # Sets of watched movies of the users (actually the virtual movie rows, but the result is the same)
        watched_movies1 = set(user_movies_dict[user1])
        watched_movies2 = set(user_movies_dict[user2])
        # Jaccard similarity of the users
        jaccard_sim = jaccard_similarity(watched_movies1, watched_movies2)
        jaccard_values.append(jaccard_sim) # add jaccard similarity to the list

        # Calculate error
        errors.append(abs(prob_same_el - jaccard_sim))

    return np.mean(errors), hash_functions

# -------------------------------------------------------------------------------------------------- #

def minhash_results(user_ids, user_movies_dict, k_hash_function_choices, linear_parameters=None, bitwise_parameters=None, k = 10, p=27281):
    '''
    Function that computes the signature matrix of user IDs and compares the signature vectors with
    the similarity between the original watched movie sets associated with the user IDs
    Inputs:
    - user_ids (ndarray of integers): categories we want to create the signature vectors for
    - user_movies_dict (dict): dictionary made of items of the structure userId: [movieId1, movieId2, ...]
    - k_hash_function_choices (list): list of 0s and 1s representing the hash function chosen (0 for linear_hash, 1 for bitwise_hash)
    - linear_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for linear_hash
    - bitwise_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for bitwise_hash
    - p (int): large prime number (larger than max movieId)
    - k (int): length of the signature vectors
    Outputs:
    - error (int): error rate of this configuration
    - hash_functions (list): list of tuples (i,a,b) where 'i' identifies the chosen hash function, and 'a', 'b' are its parameters
    '''
    signature_matrix = minhash(user_ids, user_movies_dict, k_hash_function_choices, linear_parameters, bitwise_parameters, k = 10)

    N = len(user_ids) # number of users to compare
    # List of all user columns
    all_user_columns = list(range(signature_matrix.shape[1]))
    # Sampled user columns
    sampled_user_columns = random.sample(all_user_columns, N)
    # Split the sampled users into two groups
    first_group = sampled_user_columns[:(N//2)]
    second_group = sampled_user_columns[(N//2):]

    errors = [] # Initialize list to store errors, the abs differences between P(s1[i]=s2[i]) and Jaccard(user1, user2)
    prob_values = [] # Initialize list to store probability values P(s1[i]=s2[i]) as defined above
    jaccard_values = [] # Initialize list to store Jaccard similarity values between pairs of users

    # Pair-wise compare jaccard similarity of the users with their
    # probability of having corresponding signature vector elements
    for i in range(N//2):

        # Signature vectors of the current pair of users
        signature1 = signature_matrix[:,first_group[i]]
        signature2 = signature_matrix[:,second_group[i]]
        # Compare signature vectors
        prob_same_el = sum(signature1==signature2)/k
        prob_values.append(prob_same_el) # add probability to prob_values list

        # User ID of the current user
        user1 = user_ids[first_group[i]]
        user2 = user_ids[second_group[i]]
        # Sets of watched movies of the users (actually the virtual movie rows, but the result is the same)
        watched_movies1 = set(user_movies_dict[user1])
        watched_movies2 = set(user_movies_dict[user2])
        # Jaccard similarity of the users
        jaccard_sim = jaccard_similarity(watched_movies1, watched_movies2)
        jaccard_values.append(jaccard_sim) # add jaccard similarity to the list

        # Calculate error
        errors.append(abs(prob_same_el - jaccard_sim))

    return np.mean(errors)

# -------------------------------------------------------------------------------------------------- #

def linear_dot_prod_hash(x,a,b,p):
    '''
    Function that hashes an array of positive integers to the range [1, p] using linear dot product hashing
    Inputs:
    - x (int or ndarray): integer or array of integers to hash
    - a (int): LCG scaling parameter
    - b (int): LCG shift parameter
    - p (int): prime
    Output:
    - int or ndarray: the hashed value(s) constrained to the range [1, p]
    '''
    a = np.array(a, dtype=np.int64)  # Ensure `a` is 64-bit
    x = np.array(x, dtype=np.int64)  # Ensure `x` is 64-bit
    b = np.int64(b)  # Ensure `b` is 64-bit
    p = np.int64(p)  # Ensure `p` is 64-bit
    return (np.dot(a,x) + b) % p

# -------------------------------------------------------------------------------------------------- #

def LSH(signature_matrix, user_ids, r=4, p=200003, seed=4294967295):
    '''
    Function that performs LSH on a signature matrix
    Inputs:
    - signature_matrix (ndarray): matrix of signature vectors
    - user_ids (ndarray of integers): user IDs
    - r (int): number of hash functions
    - p (int): prime indicating the max possible number of buckets
    - seed (int): seed for reproducibility in randomness
    Outputs:
    - buckets (defaultdict): dictionary that maps buckets to the users hashed to it
    - candidates (defaultdict): dictionary that maps users to all the users that share buckets with them
    '''
    rng = np.random.RandomState(seed) # create random number generator

    num_bands = signature_matrix.shape[0] // r # number of bands

    buckets = defaultdict(list) # Initialize defaultdict that maps buckets to the users hashed to it
    candidates = defaultdict(set) # Initialize defaultdict that maps users to all the users that share buckets with them
    user_buckets = defaultdict(list) # Initialize defaultdict that maps users to their buckets

    # Hash each signature vector band to a bucket
    for j in range(num_bands):
        # Generate random parameters for the hash function linear_dot_prod_hash
        a = rng.randint(1, int(0.2*p), r)
        b = rng.randint(0, int(0.2*p))
        for i in range(signature_matrix.shape[1]):
            band_range = list(range(j*r, (j+1)*r)) # row indeces range of the band
            hashed_value = linear_dot_prod_hash(signature_matrix[band_range,i],a,b,p) # hash signature vector band to a bucket
            buckets[hashed_value].append(user_ids[i]) # append user ID to the bucket
            user_buckets[user_ids[i]].append(hashed_value) # append bucket to the user

    # Find all pairs of users that share buckets
    # Iterate over the users
    for user in user_buckets:
        # Iterate over the user's buckets
        for bucket in user_buckets[user]:
            # Add the users from that bucket to the user's candidates
            candidates[user].update(buckets[bucket])

    # Remove self-candidates
    for user in candidates:
        candidates[user].remove(user)

    return buckets, candidates

# -------------------------------------------------------------------------------------------------- #

def LSH_performance(signature_matrix, user_ids, similar_pairs, r=4, p=200003, seed=4294967295):
    '''
    Function that performs LSH on a signature matrix and returns the average size of the candidate sets and
    the percentage of similar users that are candidates
    Inputs:
    - signature_matrix (ndarray): matrix of signature
    - user_ids (ndarray of integers): user IDs
    - similar_pairs (list of tuples): pairs of similar users with their jaccard similarities
    - r (int): number of hash functions
    - p (int): prime indicating the max possible number of buckets
    - seed (int): seed for reproducibility in randomness
    Outputs:
    - avg_candidate_set_size (float): average size of the candidate sets
    - percentage_similar_candidates (float): percentage of similar users that are candidates
    '''
    buckets, candidates = LSH(signature_matrix, user_ids, r, p, seed)
    print("Computed LSH, now evaluating performance")
    # Create dataframe to store the users and candidates with potential similarity to them
    candidates_df = pd.DataFrame(candidates.items(), columns = ['User', 'Candidates'])
    candidates_df['NumCandidates'] = candidates_df['Candidates'].apply(len)

    # Average size of the candidate sets
    avg_candidate_set_size = np.mean(candidates_df['NumCandidates'])

    similar_candidates = [] # Initialize list of similar user pairs that are mutual candidates

    # Iterate over user pairs with jaccard similarity of at least 0.5
    for i in range(len(similar_pairs)):
        # If user1 is a candidate of user2 and user2 is a candidate of user1
        if similar_pairs[i][0] in candidates[similar_pairs[i][1]] and similar_pairs[i][1] in candidates[similar_pairs[i][0]]:
            similar_candidates.append(similar_pairs[i]) # add the pair to the list

    # Percentage of similar pairs that are also mutual candidates
    percentage_similar_candidates = len(similar_candidates)/len(similar_pairs)

    return avg_candidate_set_size, percentage_similar_candidates