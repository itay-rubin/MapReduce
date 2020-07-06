//
// Created by itayr on 5/25/2020.
//

#ifndef OS_EX3_JOBCONTEXT_H
#define OS_EX3_JOBCONTEXT_H

#include <unordered_map>
#include <atomic>
#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include "Barrier.h"
#define ERR_MSG_MUTEX_LOCK "system error: can't lock mapMutex"
#define ERR_MSG_MUTEX_UNLOCK "system error: can't unlock mapMutex"

/**
 * A struct wrapping a pthread
 */
typedef struct
{
    /**
     * Mutex for the thread's output vector
     */
    pthread_mutex_t mutex;

    /**
     * Output vector of the mapping phase
     */
    std::vector<IntermediatePair> output;

    /**
     * Counts how many pairs the thread has mapped
     */
    int counter;

} MappingThread;


/**
 * A class wrapping a multithreaded job
 */
class JobContext
{

private:

    const MapReduceClient& _client;

    const InputVec& _inputVec;

    int _multiThreadLevel;

    std::map<pthread_t, MappingThread> _mappingThreads;

    pthread_t _shuffle;

    IntermediateMap _intermediateMap;

    std::vector<K2*> _intermediateKeys;

    std::atomic<uint64_t> _atomic;

    Barrier _barrier;

    /**
     * Initializes a new mapping thread
     */
    void _initMapThread();

    /**
     * Prints error message and exits with failure code
     * @param msg: string describing the error
     */
    static void _errorHandler(const std::string& msg);

    /**
     * For shuffling thread. Iterating over all mapping threads' output vectors
     * @param shuffleStage: true iff in shuffle stage
     */
    void _iterateOverVecs(bool shuffleStage);

    /**
     * Transferring pairs from the thread's output vec to the intermediate map
     * @param vec: current vector we're processing
     * @param shuffleStage: true iff in shuffle stage
     */
    void _shuffleVec(std::vector<IntermediatePair> &vec, bool shuffleStage);


    /**
     * Initializing to shuffle stage
     */
    void _initShuffleStage();

    /**
     * Fills the _intermediateKeys with unique keys from the Intermediate Map and sets variables for reduce stage
     */
    void _initReduce();


public:

    OutputVec& outputVec;

    pthread_mutex_t outputMutex;

    std::atomic<int> doneCounter;

    /**
     * Constructor for the JobContext
     * @param client: The implementation of MapReduceClient class
     * @param inputVec: A vector of type std::vector<std::pair<K1*, V1*>>, the input elements
     * @param outputVec: A vector of type std::vector<std::pair<K3*, V3*>>, to which the output elements will be added
     *                   before returning
     * @param multiThreadLevel: The number of worker threads to be used for running the algorithm
     */
    JobContext(const MapReduceClient& client, const InputVec& inputVec, OutputVec& outputVec, int multiThreadLevel);

    /**
     * Destructor - destroying output vector mapMutex and mappers mutexes
     */
    ~JobContext();

    /**
     * Starts the mapping process
     * @param arg: The JobContext of the job using the function
     * @return: dummy retval
     */
    static void* startMap(void* arg);

    /**
     * Starts the shuffling process
     * @param arg: The JobContext of the job using the function
     * @return: dummy retval
     */
    static void* startShuffle(void* arg);

    /**
     * Starts the reducing process
     */
    void startReduce();

    /**
     * @param tid of the wanted thread
     * @return MappingThread struct according to the tid
     */
    MappingThread& getThread(pthread_t tid);

    /**
     * @return the JobState of this job
     */
    JobState getState();

    /**
     * Waits for the job to finish
     * @param first: true iff the first thread requesting waitForJob
     */
    void wait(bool first);

};


#endif //OS_EX3_JOBCONTEXT_H
