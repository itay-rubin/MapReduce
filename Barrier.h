#ifndef BARRIER_H
#define BARRIER_H

#include <pthread.h>


// a multiple use mapBarrier

class Barrier {
public:

    /**
     * Constructor for Barrier
     * @param numThreads - total number of threads the job is working with
     */
	Barrier(int numThreads);

	/**
	 * Barrier destructor
	 */
	~Barrier();

	/**
	 * A barrier for 2 places: before start mapping and before start reducing
	 */
	void startBarrier();

	/**
	 * A barrier for waitForJob
	 * @param done - true iff called by the last thread after job is done
	 */
	void waitBarrier(bool done);

private:

    pthread_mutex_t startMutex;
    pthread_cond_t startCv;
    pthread_mutex_t waitMutex;
    pthread_cond_t waitCv;
    int startCount;
	int waitCount;
	int numThreads;
	bool jobDone;
};

#endif //BARRIER_H
