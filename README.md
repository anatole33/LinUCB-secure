This is the code related to our paper <https://link.springer.com/chapter/10.1007%2F978-3-030-62576-4_13>

    @inproceedings{CDLS20,
      author    = {Ciucanu, R. and Delabrouille, A. and Lafourcade, P. and Soare, M.},
      title     = {{Secure Cumulative Reward Maximization in Linear Stochastic Bandits}},
      booktitle = {International Conference on Provable and Practical Security (ProvSec)},
      year      = {2020},
      pages     = {257--277}
    }

`linucb.py` is the unencrypted and undistributed version whereas `linucb_ds.py`
has both. `linucb_ds_parall.py` allows parallelization as described in Section 5
**Experiments**.
The same pattern applies for spectralucb files.
`tools.py` and `spectralucb_tools.py` contain useful functions for the algorithms
and plot functions.

`script_stacked.py` generates Figure **6(a)** and **6(b)** in folders
`experiment_n_cores/linucb_50_5` and `experiment_n_cores/linucb_6_18`, respectively.

`script_n_cores.py` generates Figure **6(c)** in folder `experiment_n_cores/linucb_K_varies`.

`script_compare_spectral.py` generates Figure **7** in folder `experiment_spectral`.

All scripts generate the figures using the results of previous runs, the
`*.txt` files in experimental folders. If you wish to run again the algorithms,
uncomment in the script of your interest the line starting with
`#os.system("python3 " + algo + ".py "...`

The script `install-python-and-libraries.sh` installs Python 3 and the necessary libraries.
