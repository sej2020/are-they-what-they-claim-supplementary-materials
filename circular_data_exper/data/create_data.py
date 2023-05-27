import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy as sp
from itertools import chain, combinations
import random
from pathlib import Path


def make_data_ellipse(axes, resolution, ran_seed=100):
    print('workin')
    a,b = axes
    num = 1/resolution
    angles = 2 * np.pi * np.arange(num) / num
    e2 = (1.0 - a ** 2.0 / b ** 2.0)
    tot_size = sp.special.ellipeinc(2.0 * np.pi, e2)
    arc_size = tot_size / num
    arcs = np.arange(num) * arc_size
    res = sp.optimize.root(lambda x: (sp.special.ellipeinc(x, e2) - arcs), angles)
    angles = res.x 
    pairs = []
    for angle in angles:
        x = a * np.cos(angle)
        y = b * np.sin(angle)
        pairs.append([x,y])
    return np.array(pairs)


def rotate2d(pairs, degrees):
    rot_mat = np.identity(pairs.shape[1])
    theta = np.deg2rad(degrees)
    rot_mat[0][0], rot_mat[0][1], rot_mat[1][0], rot_mat[1][1] = math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)
    new_pairs = np.array(rot_mat) @ np.array(pairs).T
    return new_pairs.T


def make_line(data, n_subset, combo, rotation):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'yellow', 'indigo', 'chartreuse', 'lightseagreen', 'mediumslateblue', 'orangered', 'violet', 'navy', 'rebeccapurple', 'orchid', 'lime', 'gold', 'firebrick']
    fig = plt.figure()
    X = data[:,0]
    Y = data[:,1]
    ax = fig.add_subplot()
    ax.plot(X,Y,'.', color = random.choice(colors), markersize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('{}-subsets_{}-combo_{}-rot.csv'.format(n_subset, combo, rotation))
    ax.grid()
    ax.set_aspect('equal')
    plt.grid(False)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.savefig('BetaDataExper/HyperEllipsoid/rem_points_test_v2/pretty_pictures/_{}-subsets_{}-combo_{}-rot.png'.format(n_subset, combo, rotation))


def relevant_powerset(splits):
    s = list(range(splits))
    return list(chain.from_iterable(combinations(s, r) for r in range(int(math.ceil(splits/2)), splits)))


def main(axes, rotation_set, n_subset_set, resolution):
    data = pd.DataFrame(make_data_ellipse(axes, resolution))
    p = Path('BetaDataExper/HyperEllipsoid/rem_points_test_v2/data/')
    p.mkdir(exist_ok=True, parents=True)
    data.to_csv(p / "_0-subsets_0-combo_0-rot.csv", index=False, header=False)
    for rotation in rotation_set:
        data_rot = rotate2d(data, rotation)
        for n_subset in n_subset_set:
            r_cnt_in_part = data_rot.shape[0]//n_subset
            data_part = [data_rot[(i-1)*r_cnt_in_part : i*r_cnt_in_part] for i in range(1, n_subset+1)]
            powerset = relevant_powerset(n_subset)
            print(powerset)
            for combo in powerset:
                df = pd.DataFrame(np.vstack(tuple([data_part[c] for c in combo])))
                df.to_csv('BetaDataExper/HyperEllipsoid/rem_points_test_v2/data/_{}-subsets_{}-combo_{}-rot.csv'.format(n_subset, combo, rotation), index=False, header=False)
            
    

######## Removing Points from Ellipse ###########
axes = [10,10]
rotation_set = [0,5,15,30,60,90] #5, 15, 30, 60, 90
n_subset_set = [3,4,5]
resolution = 0.001
main(axes,rotation_set, n_subset_set, resolution)