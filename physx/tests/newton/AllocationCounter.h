#ifndef NEWTON_ALLOCATION_COUNTER_H
#define NEWTON_ALLOCATION_COUNTER_H

// Test-only Debug CRT hook covering malloc, realloc and C++ new in this process.
#include <atomic>
#include <crtdbg.h>

namespace allocationAudit
{
inline std::atomic<bool> enabled(false);
inline std::atomic<size_t> allocations(0), frees(0);

inline int hook(int type, void*, size_t, int, long, const unsigned char*, int)
{
	if(enabled.load(std::memory_order_relaxed))
	{
		if(type == _HOOK_ALLOC || type == _HOOK_REALLOC)
		{
			allocations.fetch_add(1, std::memory_order_relaxed);
		}
		if(type == _HOOK_FREE)
		{
			frees.fetch_add(1, std::memory_order_relaxed);
		}
	}
	return 1;
}

inline void begin()
{
	allocations = 0;
	frees = 0;
	enabled = true;
}

inline bool install()
{
	_CrtSetAllocHook(hook);
	begin();
	void* control = _malloc_dbg(16, _NORMAL_BLOCK, __FILE__, __LINE__);
	_free_dbg(control, _NORMAL_BLOCK);
	enabled = false;
	return allocations == 1 && frees == 1;
}
}
#endif
