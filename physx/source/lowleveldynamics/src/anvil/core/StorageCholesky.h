#ifndef ANVIL_STORAGE_CHOLESKY_H
#define ANVIL_STORAGE_CHOLESKY_H

#include "AnvilMath.h"
#include <cmath>
#include <vector>

namespace anvil
{
class StorageCholesky
{
public:
	SparseStorage m_matrix;

	void analyzePattern(const SparseStorage& matrix)
	{
		const int size = matrix.rows();
		m_matrix.resize(size, size);
		m_parent.resize(size);
		m_nonZerosPerColumn.resize(size);
		m_tags.resize(size);
		for(int column = 0; column < size; ++column)
		{
			m_parent[column] = -1;
			m_tags[column] = column;
			m_nonZerosPerColumn[column] = 0;
			for(SparseStorage::InnerIterator entry(matrix, column); entry; ++entry)
			{
				int row = entry.index();
				if(row >= column)
				{
					continue;
				}
				for(; m_tags[row] != column; row = m_parent[row])
				{
					if(m_parent[row] == -1)
					{
						m_parent[row] = column;
					}
					++m_nonZerosPerColumn[row];
					m_tags[row] = column;
				}
			}
		}
		int* outer = m_matrix.outerIndexPtr();
		outer[0] = 0;
		for(int column = 0; column < size; ++column)
		{
			outer[column + 1] = outer[column] + m_nonZerosPerColumn[column] + 1;
		}
		m_matrix.resizeNonZeros(outer[size]);
	}

	bool factorize(const SparseStorage& matrix)
	{
		const int size = matrix.rows();
		const int* outer = m_matrix.outerIndexPtr();
		int* inner = m_matrix.innerIndexPtr();
		double* values = m_matrix.valuePtr();
		m_y.resize(size);
		m_pattern.resize(size);
		m_tags.resize(size);
		double* y = m_y.data();
		int* pattern = m_pattern.data();
		int* tags = m_tags.data();
		for(int column = 0; column < size; ++column)
		{
			y[column] = 0.0;
			int top = size;
			tags[column] = column;
			m_nonZerosPerColumn[column] = 0;
			for(SparseStorage::InnerIterator entry(matrix, column); entry; ++entry)
			{
				int row = entry.index();
				if(row > column)
				{
					continue;
				}
				y[row] += entry.value();
				int length = 0;
				for(; tags[row] != column; row = m_parent[row])
				{
					pattern[length++] = row;
					tags[row] = column;
				}
				while(length)
				{
					pattern[--top] = pattern[--length];
				}
			}
			double diagonal = y[column];
			y[column] = 0.0;
			for(; top < size; ++top)
			{
				const int row = pattern[top];
				double yi = y[row];
				y[row] = 0.0;
				yi /= values[outer[row]];
				const int end = outer[row] + m_nonZerosPerColumn[row];
				for(int entry = outer[row] + 1; entry < end; ++entry)
				{
					y[inner[entry]] -= values[entry] * yi;
				}
				diagonal -= yi * yi;
				inner[end] = column;
				values[end] = yi;
				++m_nonZerosPerColumn[row];
			}
			const int entry = outer[column] + m_nonZerosPerColumn[column]++;
			inner[entry] = column;
			if(!(diagonal > 0.0))
			{
				return false;
			}
			values[entry] = std::sqrt(diagonal);
		}
		return true;
	}

private:
	std::vector<int> m_parent;
	std::vector<int> m_nonZerosPerColumn;
	std::vector<int> m_tags;
	std::vector<int> m_pattern;
	std::vector<double> m_y;
};
}

#endif
