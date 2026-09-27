#ifndef ANVIL_STORAGE_H
#define ANVIL_STORAGE_H

#include "AnvilMath.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace anvil
{
template<typename T, typename Allocator>
void reserveStorage(std::vector<T, Allocator>& storage, std::uint32_t capacity)
{
	const std::uint32_t currentCapacity = std::uint32_t(storage.capacity());
	if(capacity > currentCapacity)
	{
		storage.reserve(std::max(capacity, 2u * currentCapacity));
	}
}

// Storage aligned beyond what operator new guarantees, without C++17 aligned allocation.
// The block's start precedes the aligned address. Allocation failure aborts, as a
// vector without exceptions would.
template<typename T, std::size_t Alignment>
struct AlignedAllocator
{
	typedef T value_type;
	template<typename U>
	struct rebind
	{
		typedef AlignedAllocator<U, Alignment> other;
	};
	AlignedAllocator() noexcept {}
	template<typename U>
	AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
	T* allocate(std::size_t count)
	{
		void* block = std::malloc(count * sizeof(T) + Alignment + sizeof(void*));
		if(!block)
		{
			std::abort();
		}
		const std::uintptr_t aligned = (std::uintptr_t(block) + sizeof(void*) + Alignment - 1) & ~std::uintptr_t(Alignment - 1);
		reinterpret_cast<void**>(aligned)[-1] = block;
		return reinterpret_cast<T*>(aligned);
	}
	void deallocate(T* pointer, std::size_t) noexcept
	{
		if(pointer)
		{
			std::free(reinterpret_cast<void**>(pointer)[-1]);
		}
	}
	template<typename U>
	bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
	template<typename U>
	bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
};
}
#endif
