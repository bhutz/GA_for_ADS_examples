This code provides a flexible genetic algorithm to search extreme examples in the arithmetic of dynamical systems. There are four built in scoring functions provided
- points with small canonical heights
- points with large rational preperiodic tails
- points with large rational periodic cycles
- maps with math rational preperiodic points
There are a significant number of parameters provided to control the functioning of the algorithm as well as two examples of parameter settings in the comments at the top of the file. They are defined as follows
- 'random_seed' - a positive integer to control the random generation of data for reproducibility
- 'map_type' - either  'polynomial' or 'rational'
- 'degree' - a positive integer at least 2
- 'population' - a positive integer (number of orbits to generate)
- 'generations' - a positive integer. How many iterations of the genetic algorithm to perform 
- 'survival' - a real number (0,1). Percentage of best examples to keep for the next generateion
- 'reset_survival'- a real number (0,1). Percentage of best examples to keep in a reset
- 'reset_interval'- a poisitive integer. number of generations to compute prior to a reset
- 'normalize_orbit' - boolean. Whether to remove the gcd from an orbit
- 'bound'- a positive integer. Max absolute value of elements of the orbit
- 'mixing_method' - either 'crossover' or 'permutation'. Permutation is slightly more expensive from a computation perspective.
- 'mutation_rate' - a real number (0,1). Percentage of time to randomly mutate an orbit
- 'mutation_method' - 'all' or 'single'. Whether to mutate the entire orbit or a single entry within the orbit
- 'target' - which scoring function to use: 'small_height', 'preperiodic', 'Morton-Silverman'
- 'error_bound' - a positive real number. Error bound for computation the canonical heights
- 'prec' - positive integer. Precision to use for the height computations
- 'orbit_target'- a positive integer. Which entry in the orbit to use for the scoring functions.
- 'orbit_weights' - for the 'preperiodic' scoring function, how the tail versus the periodic portion are weighted
- 'compare_to_random' - Boolean - whether to generate the same number of random maps to compare effectivity of the algorithm
- 'save_graph' - Boolean - save the plot of the best score from each generation
- 'graph_file'- string - name of the file to save the graph to
- 'log_file'- string - name of the file for logging output

Output contains
- a list of the best orbits from each generation
- the list of orbits in the final generation
- a list of the best orbits from the random comparison

For a small number of generations and population sizes (< 200) this will easily run on a standrd 2025 laptop with 16Gb of RAM with Sage version 10.5. For larger population sizes,
and especially for the Morton-Silverman problem in higher degrees, a signficant amount of memory is needed (on the order of 150Gb for a 2000 population size).
The first example in the code file runs in 1-2s on standard a laptop. The second examples take 10-15 minutes on standard a laptop.
