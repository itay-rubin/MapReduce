#include "MapReduceFramework.h"
#include "JobContext.h"
#include <iostream>

#define BAD_ALLOC_MSG "system error: can't allocate memory for job"

/**
 * Prints error message and exits with failure code
 * @param msg: string describing the error
 */
void errorHandler(const std::string& msg)
{
        std::cerr << msg << std::endl;
        exit(EXIT_FAILURE);
}


void emit2 (K2* key, V2* value, void* context)
{
    auto job = (JobContext*) context;
    MappingThread& thread = job->getThread(pthread_self());

    if (pthread_mutex_lock(&thread.mutex) != 0)
    {
        errorHandler(ERR_MSG_MUTEX_LOCK);
    }

    thread.output.emplace_back(key, value);

    if (pthread_mutex_unlock(&thread.mutex) != 0)
    {
        errorHandler(ERR_MSG_MUTEX_UNLOCK);
    }
    thread.counter++;
}


void emit3 (K3* key, V3* value, void* context)
{
    auto job = (JobContext*) context;

    if (pthread_mutex_lock(&(job->outputMutex)) != 0)
    {
        errorHandler(ERR_MSG_MUTEX_LOCK);
    }

    job->outputVec.push_back({key, value});

    if (pthread_mutex_unlock(&(job->outputMutex)) != 0)
    {
        errorHandler(ERR_MSG_MUTEX_UNLOCK);
    }
}


JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel)
{
    JobContext* job;
    try
    {
        job = new JobContext(client, inputVec, outputVec, multiThreadLevel);
    }
    catch (std::bad_alloc& e)
    {
        errorHandler(BAD_ALLOC_MSG);
    }
    return job;
}


void waitForJob(JobHandle job)
{
    auto casted = (JobContext*) job;
    casted->wait(!(casted->doneCounter++));
}


void getJobState(JobHandle job, JobState* state)
{
    auto casted = (JobContext*) job;
    *state = casted->getState();
}


void closeJobHandle(JobHandle job)
{
    waitForJob(job);
    delete ((JobContext*) job);
}