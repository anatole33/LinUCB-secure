This is the code related to the paper **Secure Cumulative Reward Maximization
in Linear Stochastic Bandits**.

`linucb.py` is the unencrypted and undistributed version whereas `linucb_ds.py`
has both. `linucb_ds_parall` allows parallelization as described in Section 5
**Experiments**.
The same pattern applies for spectralucb files.
`tools.py` and `spectralucb_tools.py` contain useful functions for the algorithms
and plot functions.

`script_stacked.py` generates Figure **6(a)** and **6(b)** in folder
`experiment_n_cores/50_5` and `experiment_n_cores/6_18` respectively.

`script_n_cores.py` generates Figure **6(c)** in folder `experiment_n_cores/2048`.

`script_compare_spectral` generates Figure **7** in folder `experiment_spectral`.

All scripts generates the figures using the results of previous runs, the
`*.txt` files in experimental folders. If you wish to run again the algorithms,
uncomment in the script of your interest the line starting with
`#os.system("python3 " + algo + ".py "...`

The script `install-python-and-libraries.sh` installs Python 3 and the necessary libraries.
