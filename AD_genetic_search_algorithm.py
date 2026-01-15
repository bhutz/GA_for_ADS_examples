"""
A genetic algorithm to search for extreme examples in
the arithmetic of dynamical systems.

AUTHORS:

- Ben Hutz (Jan 2026): initial version
"""

#*****************************************************************************
#       Copyright (C) 2026 Ben Hutz <bn4941@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#*****************************************************************************

#########################################
"""
Example:

params = {}
params['map_type'] = 'polynomial'
params['degree'] = 2
params['population'] = 50
params['generations'] = 20
params['prec'] = 100
params['survival'] = 0.2
params['normalize_orbit'] = True
params['bound'] = 20
params['mixing_method'] = 'permutation'
params['mutation_rate'] = 0.1
params['mutation_method'] = 'all'
params['target'] = 'preperiodic'
params['orbit_target'] = 6
params['orbit_weights'] = (1,3)
params['compare_to_random'] = False
params['save_graph'] = False
params['graph_file'] = 'graph1.pdf'
params['reset_survival'] = 0.05
params['reset_interval'] = 1000
best_pts, final_batch, rand_best = run_algorithm(params)
"""

"""
Example:

params = {}
#params['random_seed'] = 2345
params['map_type'] = 'polynomial'
params['degree'] = 8
params['population'] = 200
params['generations'] = 100
params['survival'] = 0.15
params['reset_survival'] = 0.02
params['reset_interval'] = 50
params['normalize_orbit'] = True
params['bound'] = 20
params['mixing_method'] = 'crossover'
params['mutation_rate'] = 0.05
params['mutation_method'] = 'all'
params['target'] = 'small_height'
params['error_bound'] = 0.00000000001
params['prec'] = 1000
params['orbit_target'] = 10
params['orbit_weights'] = (5,1)
params['compare_to_random'] = True
params['save_graph'] = True
params['graph_file'] = 'graph.pdf'
params['log_file'] = 'log.txt'
params['prec'] = 100
params['reset_survival'] = 0.05
params['reset_interval'] = 1000
best_pts, final_batch, rand_best = run_algorithm(params)
"""
##################################

def orbit_to_map_polynomial(orbit, deg):
    """
        This function converts an orbit to a polynomial of given degree via
        Lagrange interpolation. It assumes that the orbit starts with 0
        and that 0 is not part of the input parameter.
    """
    R = PolynomialRing(QQ,'x')
    g = gcd(orbit)
    orbit = [t/g for t in orbit]
    orbit = [0] + orbit
    f = R.lagrange_polynomial([(orbit[i], orbit[i+1]) for i in range(len(orbit)-1)])
    return DynamicalSystem_affine([f]).homogenize(1)

def orbit_to_map_rational(orbit, deg):
    """
        This function converts an orbit to a rational function of given degree
        via interpolation.
    """
    R = PolynomialRing(QQ, 2*deg+2, 'a')
    P = ProjectiveSpace(R,1,'x,y')
    orb = [P(o) for o in orbit]
    x,y = P.gens()
    L = []
    for j in range(2):
        poly =0 
        for i in range(deg+1):
            poly += R.gen(i+(deg+1)*j)*x**(deg-i)*y**i
        L.append(poly)
    f=DynamicalSystem(L)
    eqns = []
    for i in range(len(orb)-1):
        Q1 = orb[i]
        Q2 = orb[i+1]
        T = f(Q1)
        g = T[0]*Q2[1] - T[1]*Q2[0]
        eqns.append([g.coefficient(u) for u in R.gens()])
    M = matrix(len(orb)-1, 2*deg+2, eqns)
    C = M.right_kernel().gen()
    polys = []
    for j in range(2):
        poly = 0
        for i in range(deg+1):
            poly += C[i + j*(deg+1)]*x**(deg-i)*y**i
        polys.append(poly)
    return DynamicalSystem(polys).change_ring(QQ)

def score_orbit_small_height(orbit, deg, map_type, error_bound, prec=100):
    """
        scores the orbit for looking for points with small positive canonical height ratio
        (canonical height of point)/(height of fixed point sigma invariants)
        Smaller score is better

        orbit - the orbit of 0 (does not include 0)
        deg - the degree of the map to create
        map_type - either rational or polynomial
        error_bound - the error_bound with which to compute the canonical height
    """
    R = PolynomialRing(QQ,'x')
    g = gcd(orbit)
    orbit = [t/g for t in orbit]
    orbit = [0] + orbit
    if map_type == 'polynomial':
        try:
            f = R.lagrange_polynomial([(orbit[i], orbit[i+1]) for i in range(len(orbit)-1)])
            F = DynamicalSystem_affine([f]).homogenize(1)
        except ZeroDivisionError: # periodic
            return RR(prec),0
    else: #rational
        try:
            F = orbit_to_map_rational(orbit, deg)
        except ValueError:
            return RR(prec),0
    F.normalize_coordinates()
    if F.degree() != deg:
        return RR(prec),0
    P = F.domain()
    Q = P(0)
    if Q.is_preperiodic(F):
        return RR(prec),0
    h_F = max([sig.global_height(prec=prec) for sig in F.sigma_invariants(1)])
    h_pt = F.canonical_height(Q,error_bound=error_bound, prec=prec)
    if h_pt == 0 or h_F == 0:
        return RR(prec),0
    return (h_pt/h_F).n(),0

def score_orbit_preperiodic(orbit, deg, map_type, orbit_target, orbit_weights):
    """
        scores the orbit for looking for perperiodic points based on orbit_weights
        Smaller score is better.

        It first reduces the height of a forward orbit point until it finds a preperiodic
        point, then uses the (tail,period) to score the orbit

        orbit - the orbit of 0 (does not include 0)
        deg - the degree of the map to create
        map_type - either rational or polynomial
        orbit_target - which iterate to check for height
        orbit_weights - (a,b) score the orbit as -a*(tail) - b*(period) (a,b should be positive)
    """
    R = PolynomialRing(QQ,'x')
    if len(set(orbit)) != len(orbit) or 0 in orbit:
        # too short
        return [10000, (0,0)]
    orbit = [0] + orbit
    if map_type == 'polynomial':
        try:
            f = R.lagrange_polynomial([(orbit[i], orbit[i+1]) for i in range(len(orbit)-1)])
            F = DynamicalSystem_affine([f]).homogenize(1)
        except ZeroDivisionError: # periodic
            return [10000, (0,0)]
    else: #rational
        try:
            F = orbit_to_map_rational(orbit, deg)
        except ValueError:
            return [10000, (0,0)]
    F.normalize_coordinates()
    if F.degree() != deg:
        return [10000, (0,0)]
    P = F.domain()
    h = F.nth_iterate(P(0), orbit_target).global_height()
    if h > 10: # pick a 'big' cut-off since is_preperiodic is slow
        return (h, [-1,-1])
    m,n = P(0).is_preperiodic(F,return_period=True)
    if (m,n) == (0,0):
        return (h, (m,n))
    return (-orbit_weights[0]*m - orbit_weights[1]*n, (m,n))


def score_orbit_MS(orbit, deg, map_type, orbit_target):
    """
        scores the orbit for looking for many perperiodic points.
        Smaller score is better.

        It first reduces the height of a forward orbit point until it finds a preperiodic
        point, then it takes -#(preperiodic)

        orbit - the orbit of 0 (does not include 0)
        deg - the degree of the map to create
        map_type - either rational or polynomial
        orbit_target - which iterate to check for height
    """
    R = PolynomialRing(QQ,'x')
    if len(set(orbit)) != len(orbit) or 0 in orbit:
        # too short
        return [10000, (0,0)]
    orbit = [0] + orbit
    if map_type == 'polynomial':
        try:
            f = R.lagrange_polynomial([(orbit[i], orbit[i+1]) for i in range(len(orbit)-1)])
            F = DynamicalSystem_affine([f]).homogenize(1)
        except ZeroDivisionError: # periodic
            return [10000, (0,0)]
    else: #rational
        try:
            F = orbit_to_map_rational(orbit, deg)
        except ValueError:
            return [10000, (0,0)]
    F.normalize_coordinates()
    if F.degree() != deg:
        return [10000, (0,0)]
    P = F.domain()
    h = F.nth_iterate(P(0), orbit_target).global_height()
    if h > 10: # pick a 'big' cut-off since is_preperiodic is slow
        return (h, [-1,-1])
    m,n = P(0).is_preperiodic(F,return_period=True)
    if (m,n) == (0,0):
        return (h, (m,n))
    try:
        pre = F.all_preperiodic_points(prime_bound=[1,50], lifting_prime=47)
        return(-len(pre), (0,0))
    except:
        try:
            pre = F.all_preperiodic_points(prime_bound=[1,80], lifting_prime=47)
            return(-len(pre), (0,0))
        except:
            #print("fail : " + str(orbit))
            return(10000, (0,0))

def insert_item_smaller(pts, item, index):
    """
        Binary search and insert into the list.
        smallest item is the first in the list
    """
    N = len(pts)
    if N == 0:
      return [item]
    elif N == 1:
        if item[index] < pts[0][index]:
            pts.insert(0,item)
        else:
            pts.append(item)
        return pts
    else: #binary insertion
        left = 1
        right = N
        mid = (left + right)//2
        if item[index]  < pts[mid][index]:
        # item goes into first half
            return insert_item_smaller(pts[:mid], item, index) + pts[mid:N]
        else:
        # item goes into second half
            return pts[:mid] + insert_item_smaller(pts[mid:N], item, index)

def orbit_in_batch(orbit,batch):
    """
        Check if the orbit is already in the batch (list of (orbit,score)
    """
    for i in range(len(batch)):
        if orbit == batch[i][0]:
            return true
    return false
        
def pop_random(lst):
    """
        Pop a random element from the list (removed from the list)
    """
    idx = randrange(len(lst))
    return lst.pop(idx)

def get_random(lst):
    """
        Get a random element from the list (does not remove the element)
    """
    idx = randrange(len(lst))
    return lst[idx]

def mix_crossover(survivors, normalize_orbit, mutation_rate, mutation_method, population, deg, map_type, target, orbit_target, orbit_weights, orbit_len, bound, error_bound, prec=100):
    """
        Take two orbits and create two new orbits in the next generation. Does not remove the used orbits from the list
        of survivors. This is done by splitting the orbits in half and taking first+last and last+first. This is
        continued until the next generation reaches the desired population size.
        The other parameters control scoring and mutations.

        survivors - the set of orbits to use to create the next generation
        normalize_orbit - bool - whether or not to remove the gcd of the orbit
        mutation_rate - RR [0,1] - what percentage of time to cause a mutation
        mutation_method - single or all. whether to check mutation on a single entry in the orbit
            or for every entry in the orbit
        population - how many orbits in each generation
        deg - degree of the map
        map_type - polynomial or rational
        target - type of scoring to use: preperiodic, small_height, or Morton-Silverman
        orbit_target - which iterate to use in scoring
        orbit_weights - how to weight the (tail, period) when scoring preperiodic points (see scoring function)
        orbit_len - how many enteries are in the orbit
        bound - entries in the orbit are in [-bound,bound]
        error_bound - error used in canonical height computations
    """
    new_batch = copy(survivors)
    mid = orbit_len //2
    done = false
    while not done:
        # get 2 random ones
        rand1 = copy(
        get_random(survivors)[0])
        rand2 = copy(get_random(survivors)[0])

        if mutation_method == 'single':
            prob = random()
            if prob < mutation_rate:
                ind = randrange(orbit_len)
                new_val = randint(-bound, bound)
                rand1[ind] = new_val
            if normalize_orbit:
                g = max(gcd(rand1), 1)
                rand1 = [t/g for t in rand1]
            prob = random()
            if prob < mutation_rate:
                ind = randrange(orbit_len)
                new_val = randint(-bound, bound)
                rand2[ind] = new_val
            if normalize_orbit:
                g = max(gcd(rand2), 1)
                rand2 = [t/g for t in rand2]
        else: #mutate all
            probs = [random() for _ in range(2*orbit_len)]
            for i in range(orbit_len):
                if probs[i] < mutation_rate:
                    new_val = randint(-bound, bound)
                    rand1[i] = new_val
            for i in range(orbit_len):
                if probs[i+orbit_len] < mutation_rate:
                    new_val = randint(-bound, bound)
                    rand2[i] = new_val

        #mix
        new_orbit1 = rand1[mid:] + rand2[:mid]
        new_orbit2 = rand1[:mid] + rand2[mid:]
        if normalize_orbit:
            g = max(gcd(new_orbit1), 1)
            new_orbit1 = [t/g for t in new_orbit1]
            g = max(gcd(new_orbit2), 1)
            new_orbit2 = [t/g for t in new_orbit2]
        for O in [new_orbit1, new_orbit2]:
            if (len(set(O)) == len(O)) and (0 not in O) and\
                (not orbit_in_batch(O, new_batch)) and (len(new_batch) < population):
                if target == 'preperiodic':
                    sc = score_orbit_preperiodic(O, deg, map_type, orbit_target, orbit_weights)
                elif target == 'small_height':
                    sc = score_orbit_small_height(O, deg, map_type, error_bound, prec=prec)
                else:
                    sc = score_orbit_MS(O, deg, map_type, orbit_target)
                new_batch = insert_item_smaller(new_batch, [O, sc[0], sc[1]], 1)
        if len(new_batch) == population:
            done = True
    return new_batch


def mix_permutation(survivors, perm, normalize_orbit, mutation_rate, mutation_method, population, deg, map_type, target, orbit_target, orbit_weights, orbit_len, bound, error_bound, prec=100):
    """
        Take two orbits and create two new orbits in the next generation. Does not remove the used orbits from the list
        of survivors. This is done by taking a random element of the symmetric group S_N where N = 2*(orbit len). This is
        continued until the next generation reaches the desired population size.
        The other parameters control scoring and mutations.

        survivors - the set of orbits to use to create the next generation
        normalize_orbit - bool - whether or not to remove the gcd of the orbit
        mutation_rate - RR [0,1] - what percentage of time to cause a mutation
        mutation_method - single or all. whether to check mutation on a single entry in the orbit
            or for every entry in the orbit
        population - how many orbits in each generation
        deg - degree of the map
        map_type - polynomial or rational
        target - type of scoring to use: preperiodic, small_height, or Morton-Silverman
        orbit_target - which iterate to use in scoring
        orbit_weights - how to weight the (tail, period) when scoring preperiodic points (see scoring function)
        orbit_len - how many enteries are in the orbit
        bound - entries in the orbit are in [-bound,bound]
        error_bound - error used in canonical height computations
    """
    new_batch = copy(survivors)
    mid = orbit_len //2
    done = false
    while not done:
        # get 2 random ones
        rand1 = copy(get_random(survivors)[0])
        rand2 = copy(get_random(survivors)[0])

        if mutation_method == 'single':
            prob = random()
            if prob < mutation_rate:
                ind = randrange(orbit_len)
                new_val = randint(-bound, bound)
                rand1[ind] = new_val
            if normalize_orbit:
                g = max(gcd(rand1), 1)
                rand1 = [t/g for t in rand1]
            prob = random()
            if prob < mutation_rate:
                ind = randrange(orbit_len)
                new_val = randint(-bound, bound)
                rand2[ind] = new_val
            if normalize_orbit:
                g = max(gcd(rand2), 1)
                rand2 = [t/g for t in rand2]
        else: #mutate all
            probs = [random() for _ in range(2*orbit_len)]
            for i in range(orbit_len):
                if probs[i] < mutation_rate:
                    new_val = randint(-bound, bound)
                    rand1[i] = new_val
            for i in range(orbit_len):
                if probs[i+orbit_len] < mutation_rate:
                    new_val = randint(-bound, bound)
                    rand2[i] = new_val

        C = rand1+rand2
        new_orbit1 = [0 for _ in range(orbit_len)]
        new_orbit2 = [0 for _ in range(orbit_len)]
        for j in range(orbit_len):
            new_orbit1[j] = C[perm(j+1)-1]
            new_orbit2[j] = C[perm(j+orbit_len)-1]
            if normalize_orbit:
                g = max(gcd(new_orbit1), 1)
                new_orbit1 = [t/g for t in new_orbit1]
                g = max(gcd(new_orbit2), 1)
                new_orbit2 = [t/g for t in new_orbit2]
        for O in [new_orbit1, new_orbit2]:
            #exclude short periodic or preperiodic orbits
            if (len(set(O)) == len(O)) and (0 not in O) and\
                (not orbit_in_batch(O, new_batch)) and (len(new_batch) < population):
                if target == 'preperiodic':
                    sc = score_orbit_preperiodic(O, deg, map_type, orbit_target, orbit_weights)
                elif target == 'small_height':
                    sc = score_orbit_small_height(O, deg, map_type, error_bound, prec=prec)
                else:
                    sc = score_orbit_MS(O, deg, map_type, orbit_target)
                new_batch = insert_item_smaller(new_batch, [O, sc[0], sc[1]], 1)
        if len(new_batch) == population:
            done = True
    return new_batch



#######################################################

def run_algorithm(kwds):
    """
    The algorithm is generating maps by interpolating on orbit data. All orbits start at the point 0.

    The next generation in created by joining two orbits (without 0) either by mixing or by permuting.

    The dictionary 'kwds' can contain the following keywords
      - map_type: rational or polynomial (required). Type of map to search over
      - degree: integer (required). What degree map to search for
      - population: integer (required). number of maps to generate in each generation
      - generations: integer (required). number of generations before stopping
      - survival: real in [0,1] (required). What percentage of population survives to next generation
      - normalize_orbit: boolean (required). Whether to remove gcd from orbits
      - bound: integer (required). Generate orbit values from [-bound, bound]
      - mixing_method: string (required). 'crossover' or 'permutation'
      - mutation_method: 'single' or 'all' (required). Whether to check to mutate every element in the new orbit or just a single element
      - mutation_rate: real in [0,1] (required). The percentage of the time a mutation occurs
      - target: 'small_height' or 'preperiodic' or 'Morton-Silverman' (required). Whether we are looking for maps of small canonical height or maps with long orbits or many preperiodic points
      - orbit_target: integer (required for 'orbit' target). Which iterate of 0 to examine when scoring the orbit
      - orbit_weights : (integer, integer) (required for 'orbit target'). This is a pair (m,n) that weight preperiodic (m) versus periodic (n) when
            scoring an orbit. For example, (1,5) would score a point with structure (tail,period)=(3,4) as -(1*3 + 5*4) = -23.

      - 'initial_population' : (optional) set of orbits to start with (can be up to size of population)

      - 'reset_survival' - when reseting the population, what percentage to keep in [0,1]
      - 'reset_interval' - after how many generations to do a reset

      - 'log_file': string (optional. defaults to stdout).
      - compare_to_random: boolean (optional: default=False) Whether to randomly generate populations to compare performance
      - 'save_graph' : boolean (optional. default False) save the graph of best in each generation to local home directory
      - 'graph_file' : string (optional). file to save the graph to (should end in .pdf)
    """
    ## initialize random
    if 'random_seed' in kwds.keys():
        rnd_seed = kwds['random_seed']
        set_random_seed(rnd_seed)
    else:
        #comes from os, may not really be random
        set_random_seed()

    ## Set-up environment
    deg = kwds['degree']
    bound = kwds['bound']
    prec = kwds['prec']
    map_type = kwds['map_type']
    population = kwds['population']
    generations = kwds['generations']
    survival = kwds['survival']
    reset_survival = kwds['reset_survival']
    reset_interval = kwds['reset_interval']
    normalize_orbit = kwds['normalize_orbit']
    mixing_method = kwds['mixing_method']
    mutation_method = kwds['mutation_method']
    mutation_rate = kwds['mutation_rate']
    target = kwds['target']
    if 'error_bound' in kwds.keys():
        error_bound = kwds['error_bound']
    else:
        error_bound = 0.000001
    if target == 'preperiodic':
        orbit_target = kwds['orbit_target']
        orbit_weights = kwds['orbit_weights']
    elif target == 'Morton-Silverman':
        orbit_target = kwds['orbit_target']
        orbit_weights = (0,0)
    else:
        orbit_target = 0
        orbit_weights = (0,0)
    if 'log_file' in kwds.keys():
        log_file = kwds['log_file']
        logfile = open(log_file, 'w')
    else:
        logfile = sys.stdout
    if 'compare_to_random' in kwds.keys():
        compare_to_random = kwds['compare_to_random']
        rand_pts = []
        rand_best = []
    else:
        compare_to_random = False
    if 'save_graph' in kwds.keys():
        save_graph = kwds['save_graph']
        graph_file = kwds['graph_file']
    else:
        save_graph = False

    if map_type == 'polynomial':
        orbit_len = deg + 1
    elif map_type == 'rational':
        orbit_len = 2*deg + 1
    else:
        raise(ValueError, "must be polynomial or rational")

    #needed for mixing orbits
    if mixing_method == 'permutation':
        S = SymmetricGroup(2*(orbit_len))

    #get starting set
    batch = []
    best_pts = []
    if 'initial_population' in params.keys():
        for orbit in params['initial_population']:
            if normalize_orbit:
                g = max(gcd(orbit),1)
                orbit = [t/g for t in orbit]
            #exclude short periodic or preperiodic orbits
            if (len(set(orbit)) == len(orbit)) and (0 not in orbit) and (not orbit_in_batch(orbit, batch)):
                if target == 'preperiodic':
                    sc = score_orbit_preperiodic(orbit, deg, map_type, orbit_target, orbit_weights)
                elif target == 'small_height':
                    sc = score_orbit_small_height(orbit, deg, map_type, error_bound, prec=prec)
                else:
                    sc = score_orbit_MS(orbit, deg, map_type, orbit_target)
                batch = insert_item_smaller(batch, [orbit, sc[0],sc[1]],1)
    while len(batch) != population:
        orbit = [randint(-bound, bound)for t in range(orbit_len)]
        if normalize_orbit:
            g = max(gcd(orbit),1)
            orbit = [t/g for t in orbit]
        #exclude short periodic or preperiodic orbits
        if (len(set(orbit)) == len(orbit)) and (0 not in orbit) and (not orbit_in_batch(orbit, batch)):
            if target == 'preperiodic':
                sc = score_orbit_preperiodic(orbit, deg, map_type, orbit_target, orbit_weights)
            elif target == 'small_height':
                sc = score_orbit_small_height(orbit, deg, map_type, error_bound, prec=prec)
            else:
                sc = score_orbit_MS(orbit, deg, map_type, orbit_target)
            batch = insert_item_smaller(batch, [orbit, sc[0],sc[1]],1)
    best_pts.append((0, batch[0][1]))

    # how many survive to next generation
    cut_off = (population*survival).trunc()
    reset_cut_off = (population*reset_survival).trunc()

    for i in range(1,generations):
        if i % reset_interval == 0:
            print(i,"reset")
            new_batch = batch[:reset_cut_off]
            # repopulate with random
            while len(batch) != population:
                orbit = [randint(-bound, bound)for t in range(orbit_len)]
                if normalize_orbit:
                    g = max(gcd(orbit),1)
                    orbit = [t/g for t in orbit]
                #exclude short periodic or preperiodic orbits
                if (len(set(orbit)) == len(orbit)) and (0 not in orbit) and (not orbit_in_batch(orbit, new_batch)):
                    if target == 'preperiodic':
                        sc = score_orbit_preperiodic(orbit, deg, map_type, orbit_target, orbit_weights)
                    elif target == 'small_height':
                        sc = score_orbit_small_height(orbit, deg, map_type, prec=prec)
                    else:
                        sc = score_orbit_MS(orbit, deg, map_type, orbit_target)
                    new_batch = insert_item_smaller(new_batch, [orbit, sc[0],sc[1]],1)
        else:
            survivors = batch[:cut_off]
            if mixing_method == 'permutation':
                perm = S.random_element()
                new_batch = mix_permutation(survivors, perm, normalize_orbit, mutation_rate, mutation_method, population, deg, map_type, target, orbit_target, orbit_weights, orbit_len, bound, error_bound, prec=prec)
            else:
                new_batch = mix_crossover(survivors, normalize_orbit, mutation_rate, mutation_method, population, deg, map_type, target, orbit_target, orbit_weights, orbit_len, bound, error_bound, prec=prec)

        batch = new_batch
        #print(str(batch[0]) + ' : ' + str(batch[1]))
        logfile.write(str(batch[0]) + ' : ' + str(batch[1]) + '\n')
        best_pts.append((i+1, batch[0][1]))

        if compare_to_random:
            #comparison to random
            rand_pts = rand_pts[:5]
            while len(rand_pts) != population:
                C = [randint(-bound, bound) for t in range(orbit_len)]
                if normalize_orbit:
                    g = max(gcd(C),1) #avoid 0
                    C = [t/g for t in C]
                if not orbit_in_batch(C, rand_pts):
                    if target == 'preperiodic':
                        sc = score_orbit_preperiodic(C, deg, map_type, orbit_target, orbit_weights)
                    elif target == 'small_height':
                        sc = score_orbit_small_height(C, deg, map_type, error_bound, prec=prec)
                    else:
                        sc = score_orbit_MS(orbit, deg, map_type, orbit_target)
                    rand_pts = insert_item_smaller(rand_pts, [C, sc[0], sc[1]], 1)
            rand_best.append((i, rand_pts[0][1]))

    if compare_to_random:
        result_graph = points(best_pts, color='blue') + points(rand_best, color='green')
    else:
        result_graph = points(best_pts[2:], color='blue')

    try:
        if save_graph:
            result_graph.save(graph_file)
    except:
        pass
    result_graph.show()
    if logfile != sys.stdout:
        logfile.close()

    return (best_pts, batch, rand_best)

