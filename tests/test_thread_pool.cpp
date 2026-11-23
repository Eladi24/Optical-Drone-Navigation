// Hand-rolled pthreads ThreadPool (used by ORBFeatureEstimator and SplitPipelineEstimator's
// per-candidate matching). Also the main ThreadSanitizer target.
#include "ThreadPool.hpp"
#include "SyncBarrier.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <vector>

namespace {
void* increment(void* arg)
{
    static_cast<std::atomic<int>*>(arg)->fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}
}  // namespace

TEST(ThreadPool, SpawnsTheRequestedThreads)
{
    ThreadPool pool(4);
    EXPECT_EQ(pool.threadCount(), 4u);
}

TEST(ThreadPool, DestructorDrainsAllQueuedTasks)
{
    std::atomic<int> counter{0};
    {
        ThreadPool pool(4);
        for (int i = 0; i < 2000; ++i) pool.enqueue(increment, &counter);
    }  // ~ThreadPool drains the queue, then joins
    EXPECT_EQ(counter.load(), 2000);
}

namespace {
struct BarrierTask {
    int index;
    std::vector<int>* out;
    SyncBarrier* barrier;
};
void* writeAndSignal(void* arg)
{
    auto* t = static_cast<BarrierTask*>(arg);
    (*t->out)[t->index] = t->index * t->index;
    t->barrier->signal();
    return nullptr;
}
}  // namespace

// The exact pattern SplitPipelineEstimator uses: n tasks, each writes its own slot, the caller
// blocks on a barrier, then reads every slot.
TEST(ThreadPool, BarrierPatternSeesEveryResult)
{
    ThreadPool pool(4);
    for (int round = 0; round < 50; ++round) {
        const int n = 10;
        std::vector<int> out(n, -1);
        SyncBarrier barrier(n);
        std::vector<BarrierTask> tasks(n);
        for (int i = 0; i < n; ++i) {
            tasks[i] = {i, &out, &barrier};
            pool.enqueue(writeAndSignal, &tasks[i]);
        }
        barrier.wait();
        for (int i = 0; i < n; ++i) ASSERT_EQ(out[i], i * i) << "round " << round;
    }
}
