#ifndef NEWTON_PARALLEL_H
#define NEWTON_PARALLEL_H

namespace newton
{
typedef void (*ParallelFunction)(void*, int);

class ParallelExecutor
{
public:
	virtual ~ParallelExecutor() {}
	virtual int workerCapacity() const = 0;
	virtual int acquireWorkerCount() = 0;
	virtual void parallelFor(int count, ParallelFunction function, void* context) = 0;
	virtual void endParallelRegion() = 0;
};
}

#endif
