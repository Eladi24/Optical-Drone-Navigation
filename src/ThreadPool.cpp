#include "ThreadPool.hpp"

ThreadPool::ThreadPool(size_t num_threads) : _stop(false) {
    pthread_mutex_init(&_queue_mutex, nullptr);
    pthread_cond_init(&_condition,    nullptr);

    _threads.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        pthread_t thread;
        if (pthread_create(&thread, nullptr, worker, this) == 0) {
            _threads.push_back(thread);
        } else {
            std::cerr << "❌ ThreadPool: failed to create thread " << i << std::endl;
        }
    }

    // A pool with zero live threads can never execute any task.
    // Fail hard here rather than letting callers silently deadlock on a barrier.
    if (_threads.empty()) {
        pthread_mutex_destroy(&_queue_mutex);
        pthread_cond_destroy(&_condition);
        throw std::runtime_error("ThreadPool: failed to create any worker threads");
    }
}

ThreadPool::~ThreadPool() {
    // Set the stop flag under the lock so workers that wake up spuriously see it.
    pthread_mutex_lock(&_queue_mutex);
    _stop = true;
    pthread_mutex_unlock(&_queue_mutex);

    // Broadcast rather than signal: every sleeping thread must wake up and exit.
    pthread_cond_broadcast(&_condition);

    for (pthread_t& thread : _threads)
        pthread_join(thread, nullptr);

    pthread_mutex_destroy(&_queue_mutex);
    pthread_cond_destroy(&_condition);
}

void ThreadPool::enqueue(void* (*function)(void*), void* arg) {
    pthread_mutex_lock(&_queue_mutex);

    // Guard against submitting work after destruction has started.
    // This is a caller bug, but catching it prevents a silent data race.
    if (_stop) {
        pthread_mutex_unlock(&_queue_mutex);
        std::cerr << "⚠️  ThreadPool::enqueue called on a stopped pool — task dropped"
                  << std::endl;
        return;
    }

    _tasks.push({function, arg});
    pthread_mutex_unlock(&_queue_mutex);

    // Signal one worker — one task, one wakeup.
    pthread_cond_signal(&_condition);
}

void* ThreadPool::worker(void* arg) {
    ThreadPool* pool = static_cast<ThreadPool*>(arg);

    while (true) {
        Task task;

        pthread_mutex_lock(&pool->_queue_mutex);

        // Sleep until there is work or the pool is stopping.
        // The while loop handles spurious wakeups correctly.
        while (pool->_tasks.empty() && !pool->_stop)
            pthread_cond_wait(&pool->_condition, &pool->_queue_mutex);

        // If we were woken to stop and there is nothing left to do, exit.
        // Note: tasks that were already queued before stop() are drained first.
        if (pool->_stop && pool->_tasks.empty()) {
            pthread_mutex_unlock(&pool->_queue_mutex);
            return nullptr;
        }

        task = pool->_tasks.front();
        pool->_tasks.pop();

        pthread_mutex_unlock(&pool->_queue_mutex);

        task.function(task.arg);
    }
}