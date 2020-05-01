We used Google Cloab recommendation system code to generate the matrices of users
and movies: https://github.com/google/eng-edu/blob/master/ml/recommendation-systems/recommendation-systems.ipynb which uses Tensorflow library.

`ml-100k` folder contains the public MovieLens 100 000 ratings data set.
It can be downloaded using `dl_movie_lens.py` after uncommenting the necessary lines.
`extract_from_data.py` defines the functions and objects used to parse the data
set.
`factor.py` generates the movies and users files. Run factor with
`pyhton3 factor.py X` or `./factor.py X`.
As an example, `MoviesX.py` contains 1682 movies represented by vectors of
dimension *X*. They are used in the experiments: when an algorithm is run with
parameter *d = X*, a user is chosen in `UsersX.txt` and *K* movies are chosen
in `MoviesX.py`.

If you want to generate other Movies or Users files, you will need to run 
`install-additional-libraries.sh` before running `factor.py` with the dimension of
your choice.
