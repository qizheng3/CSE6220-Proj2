/**
 * @file    parallel_sort.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the parallel, distributed sorting function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "parallel_sort.h"
#include "utils.h"
#include <cassert>

using namespace std;


// Set up the "matrix" for Alltoallv sending and receiving
// (data transfer) between processors
void send_recv_matrix(int* sCount, int* rCount, int myLen, int recvLen, MPI_Comm &comm)
{
    int localR, p;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &localR);

    int prefix_mL[p];
    int prefix_rL[p];
    int tempSt;

    //Perform scan followed by allgather to gather the prefix sum of
    // both self current size and destination size
    MPI_Scan(&myLen, &tempSt, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allgather(&tempSt, 1, MPI_INT, prefix_mL, 1, MPI_INT, comm);

    
    MPI_Scan(&recvLen, &tempSt, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allgather(&tempSt, 1, MPI_INT, prefix_rL, 1, MPI_INT, comm);

    // determine the front and end index of the data block to transfer
    int front = 0, end = 0;
    if (localR == 0)
        ;
    else
        while(prefix_rL[front] < prefix_mL[localR-1])
            front++;
    
    end = front;

    while(prefix_rL[end] < prefix_mL[localR])
        end++;

    // Fill in the SendCounts array
    // Ohter arrays (send displacements, recv counts,  recv displacements)
    // can all be obtained based on the send counts array
    for(int i = 0; i < front; i++)
        sCount[i] = 0;

    if(end==front)
        sCount[front] = myLen;
    else
    {
        // processor where the front of the array goes
		if (localR ==0)
			sCount[front] = prefix_rL[front];
		else
			sCount[front] = prefix_rL[front] - prefix_mL[localR-1];
        // processor(s) in between
        for(int i = front + 1; i < end; i++)
            sCount[i] = prefix_rL[i] - prefix_rL[i-1];
        // processor where the end of the array goes
        sCount[end] = prefix_mL[localR] - prefix_rL[end - 1];
    }
    for(int i = end + 1; i < p; i++)
        sCount[i] = 0;

    for(int i = 0; i < p; i++)
        MPI_Gather(&sCount[i], 1, MPI_INT, rCount, 1, MPI_INT, i, comm);
    return ;
}


// Partition the elements in the array to either the smaller-or-equal
// side or the larger-than side of the pivot
void array_partition(int* front, int* end, int pvt, int &nEq, int &nSm)
{
    int j = 0;
    nEq = 0;
    for(int i = 0; i < end - front; i++)
    {
        if(front[i] <= pvt)
        {
            nEq += (front[i] == pvt? 1:0);
            swap(front[i], front[j]);
            j++;
        }
    }
    nSm = j;
    return;
}


// Partition the processors based on the proportion of total counts of 
// smaller-or-euqal and lager-than elements
void div_processor(int p, int nSm, int nLg, int &proSm, int &proLg, MPI_Comm comm)
{
    int totalSm, totalLg;
    MPI_Allreduce(&nSm, &totalSm, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&nLg, &totalLg, 1, MPI_INT, MPI_SUM, comm);
    proSm = (int)round((totalSm * p)/(totalSm + totalLg));

    if (proSm == 0 && totalSm > 0)
        proSm += 1;

    else if (proSm == p && totalLg > 0)
        proSm -= 1;

    proLg = p - proSm;
}


// Key recursive function: used to arrange data using pivoting
int sub_parallel_sort(int arrLen, vector<int> &array, MPI_Comm comm)
{
    int localR;
    int p;
    int totalLen, localLen;

    // Figure out the local length and total array length
    localLen = arrLen;
    MPI_Allreduce(&localLen, &totalLen, 1, MPI_INT, MPI_SUM, comm);
    MPI_Comm_rank(comm, &localR);
    MPI_Comm_size(comm, &p);


    // Corner Case #1: # processors > # numbers
    // use the fist n processor for sorting, the remainings are at idle
    if (p > totalLen)
    {
        MPI_Comm newComm;
        if (localR < totalLen)
        {
            MPI_Comm_split(comm, 1, localR, &newComm);
            return sub_parallel_sort(arrLen, array, newComm);
        }
        else{
            MPI_Comm_split(comm, MPI_UNDEFINED, localR, &newComm);
            return 0;
        }
    }

    // Corner Case #2: p = 1. sequential quicksort
    if (p == 1)
    {
        sort(array.begin(), array.end());
        return arrLen;
    }

    // 1 < p <= n: allocate all processors, and sort the numbers recursively

    // Generate the pivoting number
    int pivot, p_proc;
    // Randomly pick a processor
    if (localR == 0)
        p_proc = rand() % p;
    MPI_Bcast(&p_proc, 1, MPI_INT, 0, comm);
    
    // Randomly pick a number on that processor
    // This will be our pivot (randomness ensured)
    if (localR == p_proc){
        pivot = array[rand() % arrLen];
    }   
    MPI_Bcast(&pivot, 1, MPI_INT, p_proc, comm);


    // Deal with the <= part
    int nSm = 0, nEq = 0;   
    array_partition(&array[0], &array[0] + arrLen, pivot, nEq, nSm);
    int nTotalSm = 0, nTotalEq = 0;
    MPI_Allreduce(&nSm, &nTotalSm, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&nEq, &nTotalEq, 1, MPI_INT, MPI_SUM, comm);


    // If all elements are queal
    // No need to sort
    if (nTotalEq == totalLen)
        return localLen;

    // Allocate processors
    // Deal with the local array
    int nProSm = 0, nProLg = 0;
    div_processor(p, nSm, localLen - nSm, nProSm, nProLg, comm);


    int newLen;
    if (localR < nProSm)
        newLen = nTotalSm/nProSm + ((localR < nTotalSm % nProSm) ? 1 : 0);
    else
        newLen = (totalLen - nTotalSm)/nProLg + ((localR - nProSm) < ((totalLen - nTotalSm) % nProLg) ? 1 : 0);
    
    
    // Figure out where to send and reveive data
    // Divide the big problem to subproblems
    int sCount[p];
    int sDispl[p];
    int rCount[p];
    int rDispl[p];
    int tmp_sCount[p];
    int tmp_rCount[p];

    int localSmLen = newLen;
    int localLgLen = newLen;

    if (localR > nProSm)
        localSmLen = 0;
    if (localR < nProSm)
        localLgLen = 0;

    send_recv_matrix(tmp_sCount, tmp_rCount, nSm, localSmLen, comm);
    send_recv_matrix(sCount, rCount, localLen - nSm, localLgLen, comm);
    // combine two parts
    for (int i = 0; i< p; i++)
    {
        sCount[i] = sCount[i] + tmp_sCount[i];
        rCount[i] = rCount[i] + tmp_rCount[i];
    }
    //calculate displacements array
    sDispl[0] = 0; rDispl[0] = 0;
    for(int i = 1; i < p; i++)
    {
        sDispl[i] = sDispl[i-1] + sCount[i-1];
        rDispl[i] = rDispl[i-1] + rCount[i-1];
    }

    // Exchange data among processors
    vector <int> tempSt(newLen);
    MPI_Alltoallv(&array[0], sCount, sDispl, MPI_INT, &tempSt[0], rCount, rDispl, MPI_INT, comm);

    array.resize(newLen);
    array = tempSt;

    // Prepare the new communicator for subproblems
    MPI_Comm newComm;
    MPI_Comm_split(comm, (localR < nProSm), localR, &newComm);
    // get the new length from child(recursive) processes
    int lnew = sub_parallel_sort(newLen, array, newComm);
    // free the communicator!! Otherwise when p is large, the
    MPI_Comm_free(&newComm);
    
    return lnew;
    
}



// paralle quick sort main function, used to call recursive subfunction
void parallel_sort(int* begin, int* end, MPI_Comm comm)
{
    srand(time(NULL));
    int p;
    int assignLen = end - begin;
    MPI_Comm_size(comm, &p);
    
    //fetch the data
    vector<int> localArr(assignLen);

    int i = 0;
    for(; i < assignLen; i++){
      localArr[i] = begin[i];
    }
    assert(&begin[i] == end && "Data read in incorrectly!");

    //Sort the data recursively
    int newLen = sub_parallel_sort(assignLen, localArr, comm);

    // Prepare info for Alltoallv data transfer matrix
    int sCount[p];
    int sDispl[p];
    int rCount[p];
    int rDispl[p];

    send_recv_matrix(sCount, rCount, newLen, assignLen, comm);
    
    sDispl[0] = 0; rDispl[0] = 0;
    for(int i = 1; i < p; i++)
    {
        sDispl[i] = sDispl[i-1] + sCount[i-1];
        rDispl[i] = rDispl[i-1] + rCount[i-1];
    }

    //Use alltoallv to transfer data
    MPI_Alltoallv(&localArr[0], sCount, sDispl, MPI_INT, begin, rCount, rDispl, MPI_INT, comm);

    return;
}




