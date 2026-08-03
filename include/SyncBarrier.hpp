#pragma once
#include <pthread.h>

// POSIX condition-variable barrier.
// Blocks the calling thread in wait() until exactly 'count' threads
// have each called signal() once.
//
// Usage pattern (same as in GlobalLocator and ORBFeatureEstimator):
//
//   SyncBarrier barrier(N);
//   for (int i = 0; i < N; ++i)
//       pool.enqueue(worker, &args[i]);   // each worker calls barrier.signal()
//   barrier.wait();                        // caller blocks here until all N finish
//
struct SyncBarrier {
    int             tasks_remaining;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;

    explicit SyncBarrier(int count) : tasks_remaining(count) {
        pthread_mutex_init(&mutex, nullptr);
        pthread_cond_init(&cond,  nullptr);
    }

    ~SyncBarrier() {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cond);
    }

    void signal() {
        pthread_mutex_lock(&mutex);
        if (--tasks_remaining == 0)
            pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mutex);
    }

    void wait() {
        pthread_mutex_lock(&mutex);
        while (tasks_remaining > 0)
            pthread_cond_wait(&cond, &mutex);
        pthread_mutex_unlock(&mutex);
    }

    // A barrier is a unique synchronisation point — copying makes no sense.
    SyncBarrier(const SyncBarrier&)            = delete;
    SyncBarrier& operator=(const SyncBarrier&) = delete;
};