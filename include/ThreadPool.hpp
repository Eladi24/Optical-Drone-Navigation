#pragma once
#include <pthread.h>
#include <vector>
#include <queue>
#include <stdexcept>
#include <iostream>

class ThreadPool {
public:
    // Task moved inside the class to avoid polluting the global namespace.
    struct Task {
        void* (*function)(void*);
        void* arg;
    };

    // Spawns num_threads worker threads immediately.
    // Throws std::runtime_error if zero threads could be created.
    explicit ThreadPool(size_t num_threads);

    // Signals all workers to drain remaining tasks and exit, then joins them.
    ~ThreadPool();

    // Submit a task for execution by the next available worker.
    // Prints a warning and returns immediately if the pool is already stopping.
    void enqueue(void* (*function)(void*), void* arg);

    // Actual number of live threads (may be less than requested if creation failed).
    size_t threadCount() const { return _threads.size(); }

    // Non-copyable, non-movable — the workers hold a raw pointer to this object.
    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    std::vector<pthread_t> _threads;
    std::queue<Task>       _tasks;
    pthread_mutex_t        _queue_mutex;
    pthread_cond_t         _condition;
    bool                   _stop;       // always accessed under _queue_mutex

    static void* worker(void* arg);
};