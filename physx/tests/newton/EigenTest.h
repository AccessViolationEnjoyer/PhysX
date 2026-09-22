#pragma once

#include "NewtonSolver.h"
#include <Eigen/Core>

template<int Rows, int Columns>
Eigen::Matrix<double, Rows, Columns> toEigen(const newton::Matrix<Rows, Columns>& source)
{
	Eigen::Matrix<double, Rows, Columns> result;
	for(int column = 0; column < Columns; ++column)
	{
		for(int row = 0; row < Rows; ++row)
		{
			result(row, column) = source(row, column);
		}
	}
	return result;
}

inline Eigen::VectorXd toEigen(const newton::VectorStorage& source)
{
	return Eigen::Map<const Eigen::VectorXd>(source.data(), source.size());
}

inline Eigen::MatrixXd toEigen(const newton::SparseStorage& source)
{
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(source.rows(), source.cols());
	const int columnCount = int(source.cols());
	for(int column = 0; column < columnCount; ++column)
	{
		for(newton::SparseStorage::InnerIterator entry(source, column); entry; ++entry)
		{
			result(entry.row(), column) = entry.value();
		}
	}
	return result;
}
