#include "Barrier.h"
#include <cstdlib>
#include <cstdio>

/**
 * Constructor for Barrier
 * @param numThreads - total number of threads the job is working with
 */
Barrier::Barrier(int numThreads)
 : startMutex(PTHREAD_MUTEX_INITIALIZER)
 , startCv(PTHREAD_COND_INITIALIZER)
 , waitMutex(PTHREAD_MUTEX_INITIALIZER)
 , waitCv(PTHREAD_COND_INITIALIZER)
 , startCount(0)
 , waitCount(0)
 , numThreads(numThreads)
 , jobDone(false)
{ }


/**
 * Barrier destructor
 */
Barrier::~Barrier()
{
	if (pthread_mutex_destroy(&startMutex) != 0 || pthread_mutex_destroy(&waitMutex) != 0) {
		fprintf(stderr, "[[Barrier]] error on pthread_mutex_destroy");
		exit(1);
	}
	if (pthread_cond_destroy(&startCv) != 0 || pthread_cond_destroy(&waitCv) != 0){
		fprintf(stderr, "[[Barrier]] error on pthread_cond_destroy");
		exit(1);
	}
}


/**
* A barrier for waitForJob
* @param done - true iff called by the last thread after job is done
*/
void Barrier::waitBarrier(bool done)
{
    if (pthread_mutex_lock(&waitMutex) != 0){
        fprintf(stderr, "[[Barrier]] error on pthread_mutex_lock");
        exit(1);
    }
    if (!done && !jobDone) {
        if (pthread_cond_wait(&waitCv, &waitMutex) != 0){
            fprintf(stderr, "[[Barrier]] error on pthread_cond_wait");
            exit(1);
        }
    } else {
        if (pthread_cond_broadcast(&waitCv) != 0) {
            fprintf(stderr, "[[Barrier]] error on pthread_cond_broadcast");
            exit(1);
        }
        jobDone = true;
    }
    if (pthread_mutex_unlock(&waitMutex) != 0) {
        fprintf(stderr, "[[Barrier]] error on pthread_mutex_unlock");
        exit(1);
    }
}


/**
 * A barrier for 2 places: before start mapping and before start reducing
 */
void Barrier::startBarrier()
{
    if (pthread_mutex_lock(&startMutex) != 0){
        fprintf(stderr, "[[Barrier]] error on pthread_mutex_lock");
        exit(1);
    }
    if (++startCount < numThreads) {
        if (pthread_cond_wait(&startCv, &startMutex) != 0){
            fprintf(stderr, "[[Barrier]] error on pthread_cond_wait");
            exit(1);
        }
    } else {
        if (pthread_cond_broadcast(&startCv) != 0)
        {
            fprintf(stderr, "[[Barrier]] error on pthread_cond_broadcast");
            exit(1);
        }
        startCount = 0;
    }
    if (pthread_mutex_unlock(&startMutex) != 0) {
        fprintf(stderr, "[[Barrier]] error on pthread_mutex_unlock");
        exit(1);
    }
}
