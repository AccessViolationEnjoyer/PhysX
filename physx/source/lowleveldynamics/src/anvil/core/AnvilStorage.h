#ifndef ANVIL_STORAGE_H
#define ANVIL_STORAGE_H

#include "AnvilMath.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace anvil
{
template<typename T>
void reserveStorage(std::vector<T>& storage, std::uint32_t capacity)
{
	const std::uint32_t currentCapacity = std::uint32_t(storage.capacity());
	if(capacity > currentCapacity)
	{
		storage.reserve(std::max(capacity, 2u * currentCapacity));
	}
}
}
#endif
