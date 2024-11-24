import numpy as np

def LCG_hash(x, a=238, b=943, p=196613):
    '''
    Use linear congruential generator to hash a positive integer (or array of integers)
    and use modulo to fit the result into the range [1, p]
    Inputs:
    - x: integer or array of integers to hash
    - a: LCG scaling parameter
    - b: LCG shift parameter
    - p (int): prime
    Output:
    - int: the hashed value constrained to the range [1, p]
    '''
    return (a*x + b) % p


def bitwise_hash(x, shift=44, p=196613):
    '''
    Use bitwise hashing to hash a positive integer with XOR and shifting bits, 
    and use modulo to fit the result into the range [1, p]
    Inputs:
    - x: integer or array of integers to hash
    - shift (int): number to shift the bits by
    - p (int): prime number larger than the number of movie Ids
    Output:
    - int: the hashed value constrained to the range [1, p]
    '''
    x = np.asarray(x)  # ensure input is a NumPy array
    x = np.bitwise_xor(x, x >> shift)  # XOR each element with its shifted version
    x = (x % p) +1 # Apply modulo to fit in the range [1, max_val]
    return x

def minhash(userIds, movieIds, user_movies, k = 10, p=196613, seed=198324):
    '''
    Creates signature vector of length k for each userId given the movieIds and
    the list of movieIds associated with each user
    Inputs:
    - userIds (ndarray of integers): categories we want to create the signature vectors for
    - movieIds (ndarray of integers): shingles
    - user_movies (dict): dictionary made of items of the structure userId: [movieId1, movieId2, ...]
    - p (int): large prime number (larger than max movieId)
    - k (int): length of the signature vectors
    - seed (int): seed for reproducibility in randomness
    Outputs:
    - tensor of signature vectors
    '''
    # Pick k random hash functions like this:
    # 1. Generate k random binary numbers (sample from {0,1}^k)
    # 2. If the i-th number is 0, use LCG hash as the i-th function -> randomly generate parameters 'a' and 'b'
    # 3. If the i-th number is 1, use bitwise hash as the i-the function -> randomly generate the 'shift' parameter
    rng = np.random.RandomState(seed) # create random number generator
    k_rand_binary = rng.randint(0,2,k) # generate k random numbers between 0 and 1
    signature = np.zeros((k, userIds.shape[0]), dtype=int)  # Initialize matrix of signature vectors
    
    # Create signature vectors
    for i in range(k):

        # Case: use LCG_hash function
        if k_rand_binary[i] == 0:
            # Generate random parameters 'a' and 'b'
            a, b = rng.randint(1,p), rng.randint(0,p)

            # create signature values for elements i of the signature vectors
            for idx, user in enumerate(userIds):
                hashed_movies = LCG_hash(user_movies[user],a,b,p)
                signature[i,idx] = np.min(hashed_movies)

        # Case: use bitwise_hash function
        if k_rand_binary[i] == 1:
            # Generate random 'shift' parameter
            shift = rng.randint(1,64)

            # create signature values for elements i of the signature vectors
            for idx, user in enumerate(userIds):
                hashed_movies = bitwise_hash(user_movies[user],shift,p)
                signature[i,idx] = np.min(hashed_movies)

    return signature