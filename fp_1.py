# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:48:15 2023

@author: Charl
"""

from os import path
import numpy as np

import fp_1_power_solve

# import txt file
dir = "C:/Users/Charl/Documents/College/3 - Comp LA/Work/project/"
filename = "top250movies.txt"
f = path.join(dir, filename)
if path.isfile(f):
    print(f)   
    with open(f, encoding='utf-8') as movie_file:
        movie_list = movie_file.readlines()
        movie_file.close()

# create list of actors in all 250 movies
actor_list = []
for line in movie_list:
    line_ents = line.split("/")
    for ent in line_ents[1:len(line_ents)]:
        actor_list.append(ent)

# remove duplicates
uniq_actors = list(set(actor_list))
uniq_actors.sort()

# create matrix 
size = (len(uniq_actors), len(uniq_actors))
edge_mat = np.zeros(size)

# iterate through each actor and each movie adding a 1 to the intersection of the 
# actor and each person billed above them in each movie
for actor in uniq_actors:
    for movie in movie_list:
        line_ents = movie.split("/")
        if actor in line_ents:
            stop_ind = line_ents.index(actor)
            col_num = uniq_actors.index(actor)
            for i in range(1, stop_ind):
                if line_ents[i] in uniq_actors:
                    edge_mat[uniq_actors.index(line_ents[i]), col_num] += 1.0

# # check connections between individual actors
uniq_actors.index("Michael Caine")
uniq_actors.index("Christian Bale")
# # higher billed actors is row (1st), and lower billed actor is column (2nd)
edge_mat[2349,9808]

# solve
actor_nw=fp_1_power_solve.pageRank(edge_mat, uniq_actors,.69,1e-8,10)

# Use power meethod to get ordered vector of actors
ord_actors = actor_nw.powermethod()

# View segment of vector of actors
ord_actors[1:100] 
