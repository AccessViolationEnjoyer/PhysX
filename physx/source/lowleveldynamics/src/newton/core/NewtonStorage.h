#ifndef NEWTON_STORAGE_H
#define NEWTON_STORAGE_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <new>
#include <vector>

namespace newton
{
template<typename T>
void reserveStorage(std::vector<T>& storage, size_t capacity)
{
	if(capacity > storage.capacity())
		storage.reserve(std::max(capacity, 2 * storage.capacity()));
}

// Eigen's owning dynamic vectors reallocate whenever their size changes.
// Keep capacity in std::vector and expose only the active range to Eigen.
class VectorStorage : public Eigen::Map<Eigen::VectorXd>
{
	typedef Eigen::Map<Eigen::VectorXd> Base;
public:
	VectorStorage() : Base(NULL, 0) {}
	explicit VectorStorage(Eigen::Index size) : Base(NULL, 0) { resize(size); }
	VectorStorage(const VectorStorage& other) : Base(NULL, 0) { *this = other; }
	VectorStorage(VectorStorage&& other) noexcept : Base(NULL, 0) { swap(other); }
	VectorStorage& operator=(const VectorStorage& other)
	{
		if(this != &other)
		{
			resize(other.size());
			Base::operator=(other);
		}
		return *this;
	}
	template<typename Derived>
	VectorStorage& operator=(const Eigen::MatrixBase<Derived>& other)
	{
		resize(other.size());
		Base::operator=(other);
		return *this;
	}
	void resize(Eigen::Index size)
	{
		reserveStorage(m_values, size_t(size));
		m_values.resize(size_t(size));
		new(static_cast<Base*>(this)) Base(m_values.data(), size);
	}
	using Base::setZero;
	void setZero(Eigen::Index size) { resize(size); Base::setZero(); }
	void swap(VectorStorage& other) noexcept
	{
		m_values.swap(other.m_values);
		new(static_cast<Base*>(this)) Base(m_values.data(), Eigen::Index(m_values.size()));
		new(static_cast<Base*>(&other)) Base(other.m_values.data(), Eigen::Index(other.m_values.size()));
	}
private:
	std::vector<double> m_values;
};

// SparseMatrix retains its values, but normally reallocates column offsets on
// every dimension change. Keep their capacity too. Storage stays compressed.
class SparseStorage : public Eigen::SparseMatrix<double>
{
	typedef Eigen::SparseMatrix<double> Base;
public:
	SparseStorage() : m_columnCapacity(0) {}
	SparseStorage(const SparseStorage& other) : Base(other), m_columnCapacity(other.cols()) {}
	SparseStorage& operator=(const SparseStorage& other)
	{
		if(this != &other)
		{
			resize(other.rows(), other.cols());
			resizeNonZeros(other.nonZeros());
			std::copy_n(other.outerIndexPtr(), other.cols() + 1, outerIndexPtr());
			std::copy_n(other.innerIndexPtr(), other.nonZeros(), innerIndexPtr());
			std::copy_n(other.valuePtr(), other.nonZeros(), valuePtr());
		}
		return *this;
	}
	void resize(Eigen::Index rows, Eigen::Index columns)
	{
		if(columns > m_columnCapacity)
		{
			m_columnCapacity = std::max(columns, 2 * m_columnCapacity);
			Base::resize(rows, m_columnCapacity);
		}
		m_innerSize = rows;
		m_outerSize = columns;
		setZero();
	}
	void reserve(Eigen::Index entries)
	{
		if(entries > m_data.allocatedSize())
			m_data.reserve(std::max(entries, 2 * m_data.allocatedSize()) - m_data.size());
	}
	void resizeNonZeros(Eigen::Index entries)
	{
		reserve(entries);
		m_data.resize(entries);
	}
private:
	Eigen::Index m_columnCapacity;
};
}
#endif
