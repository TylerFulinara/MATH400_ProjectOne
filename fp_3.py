# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:51:09 2023

@author: Charl
"""

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# function to create and plot confusion matrix
def get_conf_mat(df, num):
    confusion_matrix = metrics.confusion_matrix(df['Actual'], df['Predicted'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.title("Using " + str(num) + " Singular Vectors")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return(confusion_matrix)

# function to give rank order by best predicted w/ % correct and return #
# correctly predicted by number
def corr_pred(conf_mat, corr_num):
    per_corr = []
    for m in range(10):
        if len(corr_num) > 9: corr_num[m][1] += conf_mat[m,m]
        else: corr_num.append([m, conf_mat[m,m]])
        per_corr.append([m, conf_mat[m,m] / conf_mat[m,:].sum() * 100])
    per_corr_df = pd.DataFrame(per_corr, columns = ["Actual", "Percent Correct"])
    print(per_corr_df.sort_values(by=['Percent Correct'], ascending=False).to_string(index=False))
    return(corr_num)

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

# create list to store SVD results for each number
svd_mat = []

# get SVD for each number and store in svd_mat
for i in range(10):
    num_mat = trng_set[400*i:400*(i+1),]
    U, S, Vt = np.linalg.svd(num_mat)
    svd_mat.append([(i+1) % 10, U, S, Vt])

# create list to store test entry label, # of vectors used, best solution and 
# sorted list of solutions for each # of singular vectors used
sol_comp = []

# for each entry in test set, find distance between entry and 5, 10, 15 and 20 
# singular vectors for each number -> shortest distance is most likely solution
for j in [5, 10, 15, 20]:
    sol_vals = []
    for k in range(len(test_set)):
        k_sol_vals = []
        for l in range(10):
            num_svd_mat = np.dot(np.dot(svd_mat[l][1][:,:j],np.diag(svd_mat[l][2])[:j,:j]),
                                 svd_mat[l][3][:j,:])
            diff = np.linalg.norm(test_set[k,:] - num_svd_mat)
            k_sol_vals.append(diff)
        min_val = min(k_sol_vals)
        #print(test_set_lab[k], k_sol_vals.index(min_val))
        sol_vals.append([test_set_lab[k], k_sol_vals.index(min_val) % 10, k_sol_vals])
    sol_comp.append([j, sol_vals])


##### get results for 5 singular vectors
df5 = pd.DataFrame(sol_comp[0][1], columns = ['Actual', 'Predicted', 'Norms'])
df5['Match'] = df5['Actual'] == df5['Predicted']

# create confusion matrix to see results
confusion_mat = get_conf_mat(df5, 5)

# overall percentage correct
print("% correct using 5 singular vectors: {:.2f}%".format(df5['Match'].values.sum() / len(df5) * 100))

# create list to store total # correct for each number
num_corr = []

# rank order of best estimated w/ % correct
num_corr = corr_pred(confusion_mat, num_corr)

# rank order by most confused #s w/ %
conf_num(confusion_mat)


##### get results for 10 singular vectors
df10 = pd.DataFrame(sol_comp[1][1], columns = ['Actual', 'Predicted', 'Norms'])
df10['Match'] = df10['Actual'] == df10['Predicted']

# create confusion matrix to see results
confusion_mat = get_conf_mat(df10, 10)

# overall percentage correct
print("% correct using 10 singular vectors: {:.2f}%".format(df10['Match'].values.sum() / len(df10) * 100))

# rank order of best estimated w/ % correct
num_corr = corr_pred(confusion_mat, num_corr)

# rank order by most confused #s w/ %
conf_num(confusion_mat)

##### get results for 15 singular vectors
df15 = pd.DataFrame(sol_comp[2][1], columns = ['Actual', 'Predicted', 'Norms'])
df15['Match'] = df15['Actual'] == df15['Predicted']

# create confusion matrix to see results
confusion_mat = get_conf_mat(df15, 15)

# overall percentage correct
print("% correct using 15 singular vectors: {:.2f}%".format(df15['Match'].values.sum() / len(df15) * 100))

# rank order by best predicted w/ % correct
num_corr = corr_pred(confusion_mat, num_corr)

# rank order by most confused #s w/ %
conf_num(confusion_mat)


##### get results for 20 singular vectors
df20 = pd.DataFrame(sol_comp[3][1], columns = ['Actual', 'Predicted', 'Norms'])
df20['Match'] = df20['Actual'] == df20['Predicted']

# create confusion matrix to see results
confusion_mat = get_conf_mat(df20, 20)

# overall percentage correct
print("% correct using 20 singular vectors: {:.2f}%".format(df20['Match'].values.sum() / len(df20) * 100))

# rank order of best estimated w/ % correct
num_corr = corr_pred(confusion_mat, num_corr)

# rank order by most confused #s w/ %
conf_num(confusion_mat)

##### cumulative most diff digits to classify
num_corr_df = pd.DataFrame(num_corr, columns = ['Actual', 'Number Correct'])
print(num_corr_df.sort_values(by=['Number Correct']).to_string(index=False))

##### graphical comparison of success rates using 5, 10, 15, 20 SVDs
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 50)
print(x)
y = 2*x + 1

plt.plot(x, y)
plt.show()


##### Look at some of the dicult numbers. (5, 2 and 8)

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
num_img(5)
num_img(8)


##### evidence that singular values are suitable basis for classifying digits
# plot all singular values for each number
plt.scatter(np.repeat(0, 400), svd_mat[9][2], c ="pink",
            linewidths = 2,
            marker ="s",
            s = 50)
 
markers = ['.',',','o','v','^','p','*','+','x']
colors = ['orange','green','black','blue','purple','maroon','red','violet','teal']
for m in range(9):
    plt.scatter(np.repeat(m+1, 400), svd_mat[m][2], c = colors[m],
            linewidths = 2,
            marker = markers[m],
            s = 100)

plt.show()


# plot only most signifant singular value for each number
plt.scatter(0, svd_mat[9][2][0], c ="pink",
            linewidths = 2,
            marker ="s",
            s = 50)
 
markers = ['.',',','o','v','^','p','*','+','x']
colors = ['orange','green','black','blue','purple','maroon','red','violet','teal']
for m in range(9):
    plt.scatter(m+1, svd_mat[m][2][0], c = colors[m],
            linewidths = 2,
            marker = markers[m],
            s = 100)

plt.show()


##### stage 2: compare 1st to largest SVD, if 1 residual is signifcantly smaller
# than the others, use that number, o/w use stage 1 (5 singular vectors) 

sol_vals_1 = []
sol_vals_5 = []
for k in range(len(test_set)):
    k_sol_vals = []
    for l in range(10):
        num_svd_mat = np.dot(np.dot(svd_mat[l][1][:,:1],np.diag(svd_mat[l][2])[:1,:1]),
                             svd_mat[l][3][:1,:])
        diff = np.linalg.norm(test_set[k,:] - num_svd_mat)
        k_sol_vals.append(diff)
    min_val = min(k_sol_vals)
    min_val_index = k_sol_vals.index(min_val)
    ordered_vals = np.sort(np.array(k_sol_vals))
    #print(test_set_lab[k], k_sol_vals.index(min_val))
    if ordered_vals[0] < .92 * ordered_vals[1]:
        sol_vals_1.append([test_set_lab[k], k_sol_vals.index(min_val) % 10])
    else:
        k_sol_vals = []
        for l in range(10):
            num_svd_mat = np.dot(np.dot(svd_mat[l][1][:,:5],np.diag(svd_mat[l][2])[:5,:5]),
                                 svd_mat[l][3][:5,:])
            diff = np.linalg.norm(test_set[k,:] - num_svd_mat)
            k_sol_vals.append(diff)
        min_val = min(k_sol_vals)
        #print(test_set_lab[k], k_sol_vals.index(min_val))
        sol_vals_5.append([test_set_lab[k], k_sol_vals.index(min_val) % 10])


print("Using 8% difference between the smallest and next smallest residual as" +
      " the demarcation between keeping the result from using 1 singular vector" +
      " and using 5 singular vectors, the # that were classified by 1 singular" +
      " vector is: " + str(len(sol_vals_1)) + 
      " which is: {:.2f}%".format(len(sol_vals_1) / (len(test_set)) * 100))
print("The # that were classified by 5 singular vectors is: " + 
      str(len(sol_vals_5)) + 
      " which is: {:.2f}%".format(len(sol_vals_5) / (len(test_set)) * 100))

df_st2_1 = pd.DataFrame(sol_vals_1, columns = ['Actual', 'Predicted'])
df_st2_1['Match'] = df_st2_1['Actual'] == df_st2_1['Predicted']
confusion_mat = get_conf_mat(df_st2_1, 1)
print("% correct using 1 singular vector: {:.2f}%".format(df_st2_1['Match'].values.sum() / len(df_st2_1) * 100))

df_st2_5 = pd.DataFrame(sol_vals_5, columns = ['Actual', 'Predicted'])
df_st2_5['Match'] = df_st2_5['Actual'] == df_st2_5['Predicted']
confusion_mat = get_conf_mat(df_st2_1, 1)
print("% correct using 5 singular vectors: {:.2f}%".format(df_st2_5['Match'].values.sum() / len(df_st2_1) * 100))

total_corr = df_st2_1['Match'].values.sum() + df_st2_5['Match'].values.sum()
print("% correct with 2 stage process: {:.2f}%".format(total_corr / (len(df_st2_1) + len(df_st2_5)) * 100))

# Result (77.2%) is the same as using 5 singular vectors for all