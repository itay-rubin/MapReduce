//
// Created by itayr on 5/25/2020.
//

#include "JobContext.h"
#include <iostream>
#include <bitset>


#define ERR_MSG_CR_THREAD "system error: can't create a new thread"
#define ERR_MSG_MUTEX_DESTROY_OUTPUT "system error: can't destroy outputMutex"
#define ERR_MSG_MUTEX_DESTROY_THREAD "system error: can't destroy thread mutex"
#define BAD_ALLOC_MSG "system error: can't allocate memory for job"
#define ERR_MSG_JOIN "system error: can't join thread"

#define TOTAL_AND   0x3fffffff80000000
#define PROGRESS_AND 0x7fffffff
#define TOTAL_SHIFT  31
#define STAGE_SHIFT 62


/**
 * Constructor for the JobContext
 * @param client: The implementation of MapReduceClient class
 * @param inputVec: A vector of type std::vector<std::pair<K1*, V1*>>, the input elements
 * @param outputVec: A vector of type std::vector<std::pair<K3*, V3*>>, to which the output elements will be added
 *                   before returning
 * @param multiThreadLevel: The number of worker threads to be used for running the algorithm
 */
JobContext::JobContext(const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec,
                       int multiThreadLevel): _client(client),
                                              _inputVec(inputVec),
                                              _multiThreadLevel(multiThreadLevel),
                                              _shuffle(),
                                              _atomic(((uint64_t)inputVec.size() << (uint64_t)TOTAL_SHIFT) |
                                                      ((uint64_t)MAP_STAGE << (uint64_t) STAGE_SHIFT)),
                                              _barrier(multiThreadLevel),
                                              outputVec(outputVec),
                                              outputMutex(PTHREAD_MUTEX_INITIALIZER),
                                              doneCounter(0)
{
    for (int i = 0; i < multiThreadLevel - 1; ++i) // Create mapping threads
    {
        _initMapThread();
    }

    if (pthread_create(&_shuffle, nullptr, startShuffle, this) != 0) // Create shuffling thread
    {
        _errorHandler(ERR_MSG_CR_THREAD);
    }
}


/**
 * Destructor - destroying output vector mapMutex and mappers mutexes
 */
JobContext::~JobContext()
{
    for (auto& thread: _mappingThreads)
    {
        if (pthread_mutex_destroy(&(thread.second.mutex)) != 0) {
            _errorHandler(ERR_MSG_MUTEX_DESTROY_THREAD);
        }
    }

    if (pthread_mutex_destroy(&outputMutex) != 0)
    {
        _errorHandler(ERR_MSG_MUTEX_DESTROY_OUTPUT);
    }
}


/**
 * Initializes a new mapping thread
 */
void JobContext::_initMapThread()
{
    pthread_t tid;

    if (pthread_create(&tid, nullptr, startMap, this) != 0)
    {
        _errorHandler(ERR_MSG_CR_THREAD);
    }

    _mappingThreads[tid] = {PTHREAD_MUTEX_INITIALIZER, std::vector<IntermediatePair>(), 0};
}


/**
 * Prints error message and exits with failure code
 * @param msg: string describing the error
 */
void JobContext::_errorHandler(const std::string &msg)
{
    std::cerr << msg << std::endl;
    exit(EXIT_FAILURE);
}


/**
 * Starts the mapping process
 * @param arg: the JobContext of the job using the function
 * @return: dummy retval
 */
void *JobContext::startMap(void* arg)
{
    auto job = (JobContext*) arg;
    job->_barrier.startBarrier();
    uint64_t cur = (job->_atomic)++;

    while ((cur & (uint64_t)PROGRESS_AND) < ((cur & (uint64_t)TOTAL_AND) >> (uint64_t)TOTAL_SHIFT))
    {
        const InputPair& curPair = job->_inputVec[cur & (uint64_t)PROGRESS_AND];
        job->_client.map(curPair.first, curPair.second, arg);

        cur = (job->_atomic)++;
    }

    job->_barrier.startBarrier();
    job->startReduce();

    return nullptr;
}


/**
 * Starts the shuffling process
 * @param arg: The JobContext of the job using the function
 * @return: dummy retval
 */
void *JobContext::startShuffle(void *arg)
{
    auto* job = (JobContext*) arg;
    job->_barrier.startBarrier();
    uint64_t cur = job->_atomic.load();

    while ((cur & (uint64_t)PROGRESS_AND) <
                    (((cur & (uint64_t)TOTAL_AND) >> (uint64_t)TOTAL_SHIFT) + (uint64_t)(job->_multiThreadLevel - 1)))
    {
        job->_iterateOverVecs(false);
        cur = job->_atomic.load();
    }

    job->_initShuffleStage();
    job->_iterateOverVecs(false);
    job->_initReduce();

    return nullptr;
}


/**
 * Starts the reducing process
 */
void JobContext::startReduce()
{
    uint64_t cur = _atomic++;

    while ((cur & (uint64_t)PROGRESS_AND) < ((cur & ((uint64_t)TOTAL_AND)) >> (uint64_t)TOTAL_SHIFT))
    {
        K2* key = _intermediateKeys[cur & (uint64_t)PROGRESS_AND];
        _client.reduce(key, _intermediateMap[key], this);
        cur = _atomic++;
    }
}


/**
 * For shuffling thread. Iterating over all mapping threads' output vectors
 * @param shuffleStage: true iff in shuffle stage
 */
void JobContext::_iterateOverVecs(bool shuffleStage)
{
    for (auto& thread: _mappingThreads)
    {
        if (pthread_mutex_lock(&thread.second.mutex) != 0)
        {
            _errorHandler(ERR_MSG_MUTEX_LOCK);
        }

        _shuffleVec(thread.second.output, shuffleStage);

        if (pthread_mutex_unlock(&thread.second.mutex) != 0)
        {
            _errorHandler(ERR_MSG_MUTEX_UNLOCK);
        }
    }
}


/**
 * Transferring pairs from the thread's output vec to the intermediate map
 * @param vec - current vector we're processing
 * @param shuffleStage: true iff in shuffle stage
 */
void JobContext::_shuffleVec(std::vector<IntermediatePair> &vec, bool shuffleStage)
{
    for (const auto& pair: vec)
    {
        _intermediateMap[pair.first].push_back(pair.second);
        if (shuffleStage) _atomic++;
    }
    vec.clear();
}


/**
 * Initializing to shuffle stage
 */
void JobContext::_initShuffleStage()
{
    uint64_t stage = (uint64_t)SHUFFLE_STAGE << (uint64_t)STAGE_SHIFT;

    uint64_t total = 0;
    for (const auto& mapper: _mappingThreads)
    {
        total += (uint64_t)(mapper.second.counter);
    }

    total = total << (uint64_t)TOTAL_SHIFT;

    uint64_t progress = _intermediateMap.size();

    _atomic.store((uint64_t)stage + (uint64_t)total + (uint64_t)progress);
}


/**
 * Waits for the job to finish
 * @param first: true iff the first thread requesting waitForJob
 */
void JobContext::wait(bool first)
{
    if (first)
    {
        for (auto& thread: _mappingThreads)
        {
            if (pthread_join(thread.first, nullptr) != 0)
            {
                _errorHandler(ERR_MSG_JOIN);
            }
        }
        if (pthread_join(_shuffle, nullptr) != 0)
        {
            _errorHandler(ERR_MSG_JOIN);
        }
        _barrier.waitBarrier(true);
    }
    else
    {
        _barrier.waitBarrier(false);
    }
}



/**
 * Fills the _intermediateKeys with unique keys from the Intermediate Map and sets variables for reduce stage
 */
void JobContext::_initReduce()
{
    for (const auto& pair: _intermediateMap)
    {
        _intermediateKeys.push_back(pair.first);
    }

    _atomic.store(((uint64_t)REDUCE_STAGE << (uint64_t)STAGE_SHIFT) |
                  ((uint64_t)_intermediateKeys.size() << (uint64_t)TOTAL_SHIFT));

    _barrier.startBarrier();
    startReduce();
}


/**
 * @param tid of the wanted thread
 * @return MappingThread struct according to the tid
 */
MappingThread &JobContext::getThread(pthread_t tid)
{
    return _mappingThreads[tid];
}


/**
 * @return the JobState of this job
 */
JobState JobContext::getState()
{
    uint64_t cur = _atomic;
    auto stage = (stage_t)(cur >> (uint64_t)STAGE_SHIFT);
    float percentage = 0;

    switch(stage)
        {
            case UNDEFINED_STAGE:
            {
                percentage = 0.0;
                break;
            }

            case MAP_STAGE:
            {
                percentage= (float)(cur & (uint64_t)PROGRESS_AND) /
                            (float)( ((cur & (uint64_t)TOTAL_AND) >> (uint64_t)TOTAL_SHIFT) + _multiThreadLevel - 1);
                break;
            }

            case SHUFFLE_STAGE:
            {
                percentage = (float)(cur & (uint64_t)PROGRESS_AND) /
                             (float)((cur & (uint64_t)TOTAL_AND) >> (uint64_t)TOTAL_SHIFT);
                break;
            }

            case REDUCE_STAGE:
                {
                percentage = (float)(cur & (uint64_t)PROGRESS_AND) /
                             (float)(((cur & (uint64_t)TOTAL_AND) >> (uint64_t)TOTAL_SHIFT) +
                             _multiThreadLevel);
                break;
                }
        }

    return {stage, percentage * 100};
}




