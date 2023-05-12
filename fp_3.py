#-*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:51:09 2023

@author: Charl
"""

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# function to create and plot confusion matrix, and get # correct and most confused #s
def get_conf_mat(df, num, num_corr):
    confusion_matrix = metrics.confusion_matrix(df['Actual'], df['Predicted'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.title("Using " + str(num) + " Singular Vectors")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    # overall percentage correct
    print("% correct using", num, "singular vectors: {:.2f}%".format(df['Match'].values.sum() / len(df) * 100))
    # rank order of best estimated w/ % correct
    num_corr = corr_pred(confusion_matrix, num_corr)
    # print table of which #s were confused with which #s
    conf_num(confusion_matrix)
    return(confusion_matrix, num_corr)

# function to give rank order by best predicted w/ % correct and return #
# correctly predicted by number
def corr_pred(conf_mat, num_corr):
    per_corr = []
    for m in range(10):
        # check to see if num_corr has a full set of entries (0-9)
        if len(num_corr) > 9: num_corr[m][1] += conf_mat[m,m]
        else: num_corr.append([m, conf_mat[m,m]])
        per_corr.append([m, conf_mat[m,m] / conf_mat[m,:].sum() * 100])
    per_corr_df = pd.DataFrame(per_corr, columns = ["Actual", "Percent Correct"])
    print(per_corr_df.sort_values(by=['Percent Correct'], ascending=False).to_string(index=False))
    return(num_corr)

# function to give rank order by most confused #s w/ %
def conf_num(conf_mat):
    per_conf = []
    for m in range(10):
        all_wrong = np.delete(conf_mat, obj=m, axis=1)[m,:]
        per_conf.append([m, np.where(conf_mat[m,] == max(all_wrong))[0], max(all_wrong)])
    per_conf_df = pd.DataFrame(per_conf, columns = ["Actual","Most Confused With", "# of Occurences"])
    print(per_conf_df.sort_values(by=['# of Occurences'], ascending=False).to_string(index=False))

# get training and test sets
file_dir = "C:/Users/Charl/Documents/College/3 - Comp LA/Work/project/"
trng_set = np.loadtxt(file_dir + "handwriting_training_set.txt", dtype=float, delimiter=",")
trng_set_lab = np.loadtxt(file_dir + "handwriting_training_set_labels.txt", dtype=float)   
test_set = np.loadtxt(file_dir + "handwriting_test_set.txt", dtype=float, delimiter=",")
test_set_lab = np.loadtxt(file_dir + "handwriting_test_set_labels.txt", dtype=float)
test_set_lab[test_set_lab == 10] = 0

# create list to store Vt from SVD results for each number
svd_mat = []

# get SVD for each number and store Vt in svd_mat
for i in range(10):
    num_mat = trng_set[400*i:400*(i+1),]
    U, S, Vt = np.linalg.svd(num_mat)
    # Store Vt because it contains represents the relationship between each
    # column and its importance to the class -> (0-9) and S
    svd_mat.append([(i+1) % 10, Vt, S])

# create list to store test entry label, # of vectors used, best solution and 
# sorted list of solutions for each # of singular vectors used
sol_comp = []

# for each entry in test set, find distance between entry and 5, 10, 15 and 20 
# singular vectors for each number -> shortest distance is most likely solution
for j in [5, 10, 15, 20]:
    sol_vals = [] # list to temporarily store results for each # of SVs
    for k in range(len(test_set)):
        k_sol_vals = [] # list to temporarily store results for each entry in test
        for l in range(10):
            l_sol_vals = [] # list to temporarily store results for each digit (0-9)
            for m in range(j):
                # project test entry onto the mth vector in Vt for the digit (0-9)
                # currently being considered. 
                Vt_vec = svd_mat[l][1][m,:] 
                numerator = np.dot(test_set[k], Vt_vec) 
                denominator = np.dot(Vt_vec, Vt_vec)
                proj = numerator / denominator * Vt_vec
                # get distance from test entry to projection
                z_length = np.linalg.norm(test_set[k] - proj)
                l_sol_vals.append(z_length) # store distance from test entry to proj
            # store the mean value of distances between the test entry and the
            # projections for each class (0-9)
            k_sol_vals.append(np.mean(l_sol_vals))
        min_val = min(k_sol_vals) # get the lowest value (closest to test entry)
        # store the test entry label and the predicted class (0-9)
        sol_vals.append([test_set_lab[k], k_sol_vals.index(min_val) % 10])
    # store the # of SVs and the list of actual and predicted values
    sol_comp.append([j, sol_vals])

# create list to store total # correct for each number by # of SVs
num_corr = []

##### get results for 5 singular vectors
df5 = pd.DataFrame(sol_comp[0][1], columns = ['Actual', 'Predicted'])
df5['Match'] = df5['Actual'] == df5['Predicted']

# create confusion matrix to see results
confusion_mat5, num_corr = get_conf_mat(df5, 5, num_corr)


##### get results for 10 singular vectors
df10 = pd.DataFrame(sol_comp[1][1], columns = ['Actual', 'Predicted'])
df10['Match'] = df10['Actual'] == df10['Predicted']

# create confusion matrix to see results
confusion_mat10, num_corr = get_conf_mat(df10, 10, num_corr)


##### get results for 15 singular vectors
df15 = pd.DataFrame(sol_comp[2][1], columns = ['Actual', 'Predicted'])
df15['Match'] = df15['Actual'] == df15['Predicted']

# create confusion matrix to see results
confusion_mat15, num_corr = get_conf_mat(df15, 15, num_corr)


##### get results for 20 singular vectors
df20 = pd.DataFrame(sol_comp[3][1], columns = ['Actual', 'Predicted'])
df20['Match'] = df20['Actual'] == df20['Predicted']

# create confusion matrix to see results
confusion_mat20, num_corr = get_conf_mat(df20, 20, num_corr)


##### graphical comparison of success rates using 5, 10, 15, 20 SVDs
correctPerNum5 = []; correctPerNum10 = []; correctPerNum15 = []; correctPerNum20 = []
for i in range(10):
    correctPerNum5.append(confusion_mat5[i,i])
    correctPerNum10.append(confusion_mat10[i,i])
    correctPerNum15.append(confusion_mat15[i,i])   
    correctPerNum20.append(confusion_mat20[i,i])


# create df of # correct for each digit by # of SVs
df = pd.DataFrame({'5_SVs': correctPerNum5, '10_SVs': correctPerNum10,
                   '15_SVs': correctPerNum15, '20_SVs': correctPerNum20})
# create heatmap
p1 = sns.heatmap(df).set(title='Correct predictions by digit and # of SVs')


# create df of total correct by # of SVs
df2 = pd.DataFrame({'5_SVs': [sum(correctPerNum5)/1000], '10_SVs': [sum(correctPerNum10)/1000],
                   '15_SVs': [sum(correctPerNum15)/1000], '20_SVs': [sum(correctPerNum20)/1000]})
# create heatmap
p2 = sns.heatmap(df2).set(title='Total % Correct by # of SVs')


##### Look at some of the difficult numbers. 
# cumulative most difficult digits to classify
num_corr_df = pd.DataFrame(num_corr, columns = ['Actual', 'Number Correct'])
print(num_corr_df.sort_values(by=['Number Correct']).to_string(index=False))

# function to map number images
def num_img(num):
    # define figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=10)
    # iterate through pictures (20) to show sample of requested #
    for j, ax in enumerate(axes.flat):
    
        # get assortment of images for requested #
        num_array = trng_set[num*400 + 20*j, :]
        # reshape vectors into 20x20 matrices
        num_matrix = np.reshape(num_array, (20, 20)).T
        # Display the image and set the title
        ax.imshow(num_matrix, cmap='gray')
        ax.set_title('{}'.format(int(trng_set_lab[num*400 + 20*j])))
        # Remove the x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    #plt.tight_layout()
    plt.show()

# get images (2 is 800-1199, 5 is 2000-2399, 8 is 3200-3599)
num_img(2)
num_img(8)
num_img(5)


##### evidence that singular values are suitable basis for classifying digits
markers = ['.',',','o','v','^','p','*','+','x', 's']
colors = ['orange','green','black','blue','purple','brown','red','violet','teal','pink']

# plot all singular values for each number
for m in range(10):
    plt.scatter(np.repeat((m+1)%10, 20), svd_mat[m][2][0:20], c = colors[m],
            linewidths = 2,
            marker = markers[m],
            s = 100)
plt.show()

# plot mean of 20 largest singular values for each number
for m in range(9):
    plt.scatter((m+1)%10, np.mean(svd_mat[m][2][0:20]), c = colors[m],
            linewidths = 2,
            marker = markers[m],
            s = 100)
plt.show()


##### stage 2: compare largest SVD 1st, if 1 residual is signifcantly smaller
# than the others, use that number, o/w use stage 1 (5 singular vectors) 

sol_vals_1 = []
sol_vals_5 = []

# iterate through test set to find best predicted number
for k in range(len(test_set)):
    k_sol_vals = []
    # iterate through 0-9 to find distance from test number to space defined by SV
    for l in range(10):
        # project test entry onto the mth vector in Vt for the digit (0-9)
        # currently being considered. 
        Vt_vec = svd_mat[l][1][1,:] 
        numerator = np.dot(test_set[k], Vt_vec) 
        denominator = np.dot(Vt_vec, Vt_vec)
        proj = numerator / denominator * Vt_vec
        # get distance from test entry to projection
        z_length = np.linalg.norm(test_set[k] - proj)
        k_sol_vals.append(z_length) # store distance from test entry to proj
    min_val = min(k_sol_vals) # get shortest distance from test entry to proj
    min_val_index = k_sol_vals.index(min_val) # get pos of shortest dist (class)
    ordered_vals = np.sort(np.array(k_sol_vals)) # sort distances
    # if the smallest distance is more than 6% less than the next smallest, keep it
    if ordered_vals[0] < .94 * ordered_vals[1]:
        sol_vals_1.append([test_set_lab[k], k_sol_vals.index(min_val) % 10])
    else:
        k_sol_vals = []
        for l in range(10):
            l_sol_vals = []
            for m in range(j):
                # project test entry onto the mth vector in Vt for the digit (0-9)
                # currently being considered. 
                Vt_vec = svd_mat[l][1][m,:] 
                numerator = np.dot(test_set[k], Vt_vec) 
                denominator = np.dot(Vt_vec, Vt_vec)
                proj = numerator / denominator * Vt_vec
                # get distance from test entry to projection
                z_length = np.linalg.norm(test_set[k] - proj)
                l_sol_vals.append(z_length) # store distance from test entry to proj
            # store the mean value of distances between the test entry and the
            # projections for each class (0-9)
            k_sol_vals.append(np.mean(l_sol_vals))
        min_val = min(k_sol_vals) # get shortest distance
        # store the test entry label and the predicted class (0-9)
        sol_vals_5.append([test_set_lab[k], k_sol_vals.index(min_val) % 10])

# Get % correct for values predicted with 1 singular vector
print("Using 6% difference between the smallest and next smallest residual as" +
      " the demarcation between keeping the result from using 1 singular vector" +
      " and using 5 singular vectors, the # that were classified by 1 singular" +
      " vector is: " + str(len(sol_vals_1)) + 
      " which is: {:.2f}%".format(len(sol_vals_1) / (len(test_set)) * 100))
# Get % correct for values predicted with 5 singular vectors
print("The # that were classified by 5 singular vectors is: " + 
      str(len(sol_vals_5)) + 
      " which is: {:.2f}%".format(len(sol_vals_5) / (len(test_set)) * 100))

num_corr = []
# Create df and confusion matrix to see results for # predicted with 1 SV
df_st2_1 = pd.DataFrame(sol_vals_1, columns = ['Actual', 'Predicted'])
df_st2_1['Match'] = df_st2_1['Actual'] == df_st2_1['Predicted']
confusion_mat, num_corr = get_conf_mat(df_st2_1, 1, num_corr)
print("% correct using 1 singular vector: {:.2f}%".format(df_st2_1['Match'].values.sum() / len(df_st2_1) * 100))

# Create df and confusion matrix to see results for # predicted with 5 SVs
df_st2_5 = pd.DataFrame(sol_vals_5, columns = ['Actual', 'Predicted'])
df_st2_5['Match'] = df_st2_5['Actual'] == df_st2_5['Predicted']
confusion_mat, num_corr = get_conf_mat(df_st2_5, 1, num_corr)
print("% correct using 5 singular vectors: {:.2f}%".format(df_st2_5['Match'].values.sum() / len(df_st2_5) * 100))

# Get overall % correct
total_corr = df_st2_1['Match'].values.sum() + df_st2_5['Match'].values.sum()
print("% correct with 2 stage process: {:.2f}%".format(total_corr / (len(df_st2_1) + len(df_st2_5)) * 100))

# Result (93.5%) is higher than using 5 singular vectors for all but slightly
# less than using 10, 15 or 20 singular vectors
