# ADM Homework 4 - Movie Recommendation System, Group #15

This GitHub repository contains the implementation of the fourth homework for the course **Algorithmic Methods of Data Mining** for the Data Science master's degree at Sapienza (2024-2025). The details about the homework are specified here: https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_4

### Group #15 Team Members
* Sandro Khizanishvili, 2175979, khizanishvilisandro@gmail.com
* Domenico Azzarito, 2209000, domazza6902@outlook.it
* Alessandro Querqui, 2031384, querqui.2031384@studenti.uniroma1.it
* Géraldine Valérie Maurer, 1996887, gmaurer08@gmail.com

### Repository Files:
* [main.ipynb](main.ipynb) - Jupyter notebook where you can see solutions and results of Homework 4 in ADM
* [hash.py](module/hash.py) - Python file containing functions that implement the Minhash and LSH algorithms, along with the hash functions used
* [recommendations.py](module/recommendations.py) - Python file containing functions that provide movie recommendations for users
* [k_means.py](module/k_means.py) - Python file that contains custom functions used to implement the K-means algorithm from scratch with the help of the MapReduce technique.
* [k_means_plus_plus.py](module/k_means_plus_plus.py) - Python file that contains custom functions used to implement the K-means++ algorithm from scratch with the help of the MapReduce technique.
* [exercise1.md](exercise1.md) - Markdown file with descriptions and experiment results for exercise 1

### Dataset:
The MovieLens 20M dataset used in this notebook is sourced from kaggle, at https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset. It is composed of the following files:
* ```genome_scores.csv```: assigns relevance scores to movie tags
* ```genome_tags.csv```: assigns unique identifiers to tags
* ```link.csv```: contains movie identifiers on the imdb and tmdb platforms
* ```movie.csv```: contains movie title and genre information
* ```rating.csv```: contains movie ratings information and related timestamps
* ```tag.csv```: store information about timestamp creation, the user that created them an the movie they were created for
 
### Libraries Used:
* *zipfile*
* *pandas*
* *numpy*
* *matplotlib.pyplot*
* *re*
* *nltk*
* *collections*
* *random*
* *tqdm*
* *sklearn*
* *ipywidgets*
* *IPython.display*
