# Parallel-KNN using mpi4py
### A parallel version of K Nearest Neighbor algorithm using Message Passing Interface in Python
I took a course last semester named High Performance Computing where I learned using shared-memory and distributed-memory architectures using APIs like OpenMP and MPI. I was fascinated by how repetitive work can be done done simultaneously on multiple processors at the same time to save a lot of computation time. I wondered if I can apply this knowledge of parallel computing to make a Machine learning algorithm overcome its "slowness". I implemented K-Nearest Neighbor algorithm in parallel using Message Passing Interface(mpi4py) for it to run faster even when the size of the dataset is extremely large. I used 48 processes on a supercomputer to achieve 17.76 times speedup of KNN as compared to its parallel version while producing the same accuracy as the sequential version. The main disadvantage of KNN was overcome by running it on parallel.

#### K-Nearest Neighbor classifier - Parallel algorithm:     
Step 1.	Load the training and test data     
Step 2.	 Set the value of K           
Step 3.	Classes are predicted for the test values by doing the following steps:               
1.	Root broadcasts K value and the testing data to all processes. Root also divides the training data set equally and sends these training data partitions to all processes        
2.	All processes calculate the Euclidean distances of all test values from each of the partitioned training set values locally. The first local K values and their predicted classes are chosen and returned back to the root process
3.	After the root process receives data from all workers, it sorts to find K top values globally, takes the majority of votes of those K top values, and then assigns a class to the new test value     
Step 4.	Calculate accuracy of the model at root and exit    

#### MPI functions used:
comm.bcast: To broadcast test data and k value to all the processes                
comm.Scatterv: Scatterv was used instead of Scatter because size of the send buffer is not always divisible by the number of processes. Scatterv was used to send partitions of training data(features) and training samples to all the processes      
comm.Barrier: Barrier was used to ensure that every process has completed their given work before bring the data back to the root process(in gather)           
comm.Gather: All local processes will calculate local k distances and then get k predicted labels. Gather was used to bring this local data to a single process to find global K distances             


#### Performance Evaluation:
To compare the results between the sequential and parallel execution of KNN, I have done an analysis of the parallel version by varying the number of processors working concurrently and comparing those outcomes with the sequential version.              

![image](https://user-images.githubusercontent.com/89469875/156885752-9246468d-e33c-4f5b-ac90-82ba12a610cb.png)

For the credit-card dataset with 27500 entries, time to solution was highest for the sequential execution. For 2 processors, the time to solution dropped down to 36.5 seconds from 71.4 seconds. The computation time kept decreasing rapidly as the number of processors kept getting increased until 20 processors. Time increased momentarily for 30 processors, and then there was again a steady decline until 48 processors.              

![image](https://user-images.githubusercontent.com/89469875/156885769-fde242c2-a413-437d-b916-63a1598d78c0.png)

Figure 2 represents the speedup achieved with parallel execution of KNN algorithm. As you see, the speedup went up as the number of processors increased. This steady rise stopped when the number of processors were increased from 20 to 30. There was again an increase after 30 processors. The maximum speedup achieved with the dataset was 17.7x with 48 processors.                      


