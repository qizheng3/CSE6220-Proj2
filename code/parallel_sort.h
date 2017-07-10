/**
 * @file    parallel_sort.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Declares the parallel sorting function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#ifndef PARALLEL_SORT_H
#define PARALLEL_SORT_H

#include <mpi.h>
#include <vector>
using namespace std;


// Set up the "matrix" for Alltoallv sending and receiving
// (data transfer) between processors
void send_recv_matrix(int* sCount, int* rCount, int myLen, int recvLen, MPI_Comm &comm);

// Partition the elements in the array to either the smaller-or-equal 
// side or the larger-than side of the pivot
void array_partition(int* front, int* end, int pvt, int &nEq, int &nSm);

// Partition the processors based on the proportion of total counts of 
// smaller-or-euqal and lager-than elements
void div_processor(int p, int nSm, int nLg, int &proSm, int &proLg, MPI_Comm comm);

// Key recursive function: used to arrange data using pivoting
int sub_parallel_sort(int arrLen, vector<int> &array, MPI_Comm comm);

/**
 * @brief   Parallel, distributed sorting over all processors in `comm`. Each
 *          processor has the local input [begin, end).
 *
 * Note that `end` is given one element beyond the input. This corresponds to
 * the API of C++ std::sort! You can get the size of the local input with:
 * int local_size = end - begin;
 *
 * @param begin Pointer to the first element in the input sequence.
 * @param end   Pointer to one element past the input sequence. Don't access this!
 * @param comm  The MPI communicator with the processors participating in the
 *              sorting.
 */
void parallel_sort(int * begin, int* end, MPI_Comm comm);

/*********************************************************************
 *              Declare your own helper functions here               *
 *********************************************************************/

// ...

#endif // PARALLEL_SORT_H
