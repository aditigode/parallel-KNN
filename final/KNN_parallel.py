
import numpy as np
import pandas as pd
from mpi4py import MPI
import timeit

#get the euclidean distannce
def euclidian_distance(a, b):

    return np.sqrt(np.sum((a - b) ** 2, axis=1))

#get k neighbor distances and their indices
def kneighbors(X_test, training_data,return_distance=False):
    n_neighbors = 5
    dist = []
    neigh_ind = []

    point_dist = [euclidian_distance(x_test, training_data) for x_test in X_test]

    for row in point_dist:
        enum_neigh = enumerate(row)
        sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:n_neighbors]

        ind_list = [tup[0] for tup in sorted_neigh]
        dist_list = [tup[1] for tup in sorted_neigh]

        dist.append(dist_list)
        neigh_ind.append(ind_list)

    if return_distance:
        return np.array(dist), np.array(neigh_ind)

    return np.array(neigh_ind),np.array(dist)

#get the predicted classes using K local indices
def predict_local_classes(neigh_ind,training_label_output_chunk):
    y_pred_neigh = np.array([training_label_output_chunk[i] for i in neigh_ind])
    return y_pred_neigh

#this function was only used for sequential execution
def predict(X_test,training_data, weights='uniform'):
    class_num = 3

    if weights == 'uniform':
        neighbors = kneighbors(X_test,training_data)

        y_pred = np.array([np.argmax(np.bincount(y_train[neighbor])) for neighbor in neighbors])

        return y_pred

    if weights == 'distance':

        dist, neigh_ind = kneighbors(X_test, return_distance=True)

        inv_dist = 1 / dist

        mean_inv_dist = inv_dist / np.sum(inv_dist, axis=1)[:, np.newaxis]

        proba = []

        for i, row in enumerate(mean_inv_dist):

            row_pred = self.y_train[neigh_ind[i]]

            for k in range(class_num):
                indices = np.where(row_pred == k)
                prob_ind = np.sum(row[indices])
                proba.append(np.array(prob_ind))

        predict_proba = np.array(proba).reshape(X_test.shape[0], class_num)

        y_pred = np.array([np.argmax(item) for item in predict_proba])

        return y_pred

#get final accuracy
def score(y_test,y_pred):
    # print("start")
    #y_pred = predict(X_test)
    # print("done")

    return float(sum(y_pred == y_test)) / float(len(y_test))



#start MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    df = pd.read_csv('creditcard_tiny2.csv', header=None)
    y = df.iloc[:, -1]
    # get labels in y
    y = np.array(y)
    # drop the last column of labels
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    # get number of samples and number of features
    n_samples, n_features = df.shape

    dataset = np.array(df)
    #Divide datasets in the ratio 7:3
    n_samples_div = int(n_samples * 0.7)

    X_train = dataset[:n_samples_div]
    X_test = dataset[n_samples_div:]


    y_train = y[:n_samples_div]
    y_test = y[n_samples_div:]
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    rows, cols = X_train.shape
    test_rows,test_cols = X_test.shape
    #K is defined as 5
    n_neighbors = 5

    outputData_dist = np.zeros((X_test.shape[0]*size, n_neighbors))
    outputData_labels = np.zeros((X_test.shape[0]*size ,n_neighbors))

    split = np.array_split(X_train, size, axis=0)  # Split input array by the number of available cores

    # this will create size of chunks to be sent to each process
    split_sizes = []
    for i in range(0, len(split), 1):
        split_sizes = np.append(split_sizes, len(split[i]))

    # this will give total number of elements to be sent to each process
    split_sizes_input = split_sizes * cols
    # this will give the starting position of global array in each local process
    displacements_input = np.insert(np.cumsum(split_sizes_input), 0, 0)[0:-1]

    # the following code is for gather
    split_sizes_output = split_sizes * cols
    displacements_output = np.insert(np.cumsum(split_sizes_output), 0, 0)[0:-1]

    # this will convert 2D numpy array into 1D continous block
    raveled = [np.ravel(arr) for arr in split]
    training_data = np.concatenate(raveled)

    split_train_label = np.array_split(y_train, size, axis=0)  # Split input array by the number of available cores

    # this will create size of chunks to be sent to each process
    split_sizes_train_label = []
    for i in range(0, len(split_train_label), 1):
        split_sizes_train_label = np.append(split_sizes_train_label, len(split_train_label[i]))

    # this will give total number of elements to be sent to each process
    split_sizes_input_train_label = split_sizes_train_label
    # this will give the starting position of global array in each local process
    displacements_input_train_label = np.insert(np.cumsum(split_sizes_input_train_label), 0, 0)[0:-1]
    raveled_labels = [np.ravel(arr) for arr in split_train_label]
    training_labels = np.concatenate(raveled_labels)
    displacements_output = np.insert(np.cumsum(split_sizes_output), 0, 0)[0:-1]


else:
    # Create variables for other cores(where rank is not zero)
    split_sizes_input = None
    displacements_input = None
    split_sizes_output = None
    displacements_output = None
    split = None
    training_data = None
    outputData_dist = None
    outputData_labels = None
    rows = None
    cols = None
    X_test= None
    training_labels = None
    split_train_label = None
    displacements_input_train_label = None
    split_sizes_train_label = None
    split_sizes_input_train_label = None

end,begin=0,0
begin=timeit.default_timer()
# broadcast sizes of split and displacements to all processes
split = comm.bcast(split, root=0)  # Broadcast split array to other cores
split_sizes = comm.bcast(split_sizes_input, root=0)
displacements = comm.bcast(displacements_input, root=0)
split_sizes_output = comm.bcast(split_sizes_output, root=0)
displacements_output = comm.bcast(displacements_output, root=0)
cols = comm.bcast(cols, root=0)
rows = comm.bcast(rows, root=0)
X_test=comm.bcast(X_test, root=0)
split_train_label = comm.bcast(split_train_label, root=0)
displacements_input_train_label = comm.bcast(displacements_input_train_label, root=0)
split_sizes_input_train_label = comm.bcast(split_sizes_input_train_label, root=0)



training_output_chunk = np.zeros(np.shape(split[rank]))  # Create array to receive subset of data on each core, where rank specifies the core

#scatter the training data to all the processes. ScatterV is used because division of training data will not be always equal
comm.Scatterv([training_data, split_sizes_input, displacements_output, MPI.DOUBLE], training_output_chunk, 0)
training_label_output_chunk = np.zeros(np.shape(split_train_label[rank]))

#scatter the training labels to all the processes. ScatterV is used because division of training data will not be always equal
comm.Scatterv([training_labels, split_sizes_input_train_label, displacements_input_train_label, MPI.DOUBLE], training_label_output_chunk, 0)

# get k neighbours local distances and local indices
neigh_ind, dist = kneighbors(X_test,training_output_chunk)
predicted_local_labels = predict_local_classes(neigh_ind,training_label_output_chunk)

output_dist = np.zeros((len(dist), 5))  # Create output array on each core
output_labels = np.zeros((len(predicted_local_labels), 5))  # Create output array on each core
# #
for i in range(0, np.shape(dist)[0], 1):
    output_dist[i, 0:5] = dist[i]
    output_labels[i, 0:5] = predicted_local_labels[i]

split_sizes_output_gather = [rows] * size
displacements_output_gather = np.insert(np.cumsum(split_sizes_output_gather), 0, 0)[0:-1]

#all processes will wait until all processes have reached this point
comm.Barrier()

#gather is used to send local K distances and local k predicted labels to the root/master process
comm.Gather(output_dist,outputData_dist, root=0)
comm.Gather(output_labels,outputData_labels, root=0)


if rank == 0:
    #sort all k distances of processes in an ascending order, choose K top distances and take majority votes of those processes to assign a final label to the test data point
    new_ele = np.split(outputData_labels, size)
    new_ele2 = np.concatenate(new_ele, axis=1)

    new_ele3 = np.split(outputData_dist, size)
    new_ele4 = np.concatenate(new_ele3, axis=1)

    indices = (np.argsort(new_ele4))

    y_pred_final = np.zeros((test_rows))
    temp = np.zeros((n_neighbors))
    #print("temp is",temp)
    for i in range(test_rows):

        for j in range(n_neighbors):
            temp[j] = new_ele2[i][indices[i, j]]

        temp = temp.astype(int)
        y_pred_final[i] = np.argmax(np.bincount(temp))
    print("number of processes are",size)
    print("score is",score(y_test,y_pred_final))


end = timeit.default_timer()
local_time = (end-begin)
global_time=0
#take average of the time taken by all processes
global_time = comm.reduce(local_time,MPI.SUM,root=0)
if rank==0:
    print("average time required is",global_time/size)


