// Six-coordinate supernodal LLT for the solver's complete rigid-body blocks.
// Scalar CSC output preserves the existing solves and signed-update path.
namespace newton
{
class BlockCholesky : public StorageCholesky
{
	typedef Eigen::Matrix<double, 6, 6> Block;
	enum
	{
		MIN_PARALLEL_FACTOR_UPDATES = 500000,
		MIN_PARALLEL_PIVOT_UPDATES = 256
	};
public:
	bool usesBlocks() const { return m_useBlocks; }
	void setParallelExecutor(ParallelExecutor* executor) { m_parallelExecutor = executor; }

	void analyzePattern(const SparseStorage& ap, bool doLDLT)
	{
		(void)doLDLT;
		const int bodies = int(ap.cols()) / 6;
		m_parent.resize(bodies);
		m_tags.resize(bodies);
		m_counts.resize(bodies);
		m_pattern.resize(bodies);
		m_blockOuter.resize(bodies + 1);
		m_rowOuter.resize(bodies + 1);
		// Body-level elimination tree and column counts. Every allocated body
		// pair is a full block, so the first scalar column supplies its graph.
		for(int k = 0; k < bodies; ++k)
		{
			m_parent[k] = -1;
			m_tags[k] = k;
			m_counts[k] = 0;
			for(Sparse::InnerIterator entry(ap, 6 * k); entry; ++entry)
			{
				int i = int(entry.row()) / 6;
				if(i >= k)
					continue;
				for(; m_tags[i] != k; i = m_parent[i])
				{
					if(m_parent[i] == -1)
						m_parent[i] = k;
					++m_counts[i];
					m_tags[i] = k;
				}
			}
		}
		m_blockOuter[0] = 0;
		for(int k = 0; k < bodies; ++k)
			m_blockOuter[k + 1] = m_blockOuter[k] + m_counts[k] + 1;
		double scalarWork = 0.0, scalarEntries = 0.0;
		for(int body = 0; body < bodies; ++body)
			for(int axis = 0; axis < 6; ++axis)
			{
				const double count = 6 * (m_counts[body] + 1) - axis;
				scalarEntries += count;
				scalarWork += (count - 1.0) * (count + 2.0);
			}
		// Dense block kernels amortize their packing/export work on columns
		// with substantial fill. Thin sparse factors use the established
		// scalar path directly; the decision uses only symbolic work.
		m_useBlocks = scalarWork >= 32.0 * scalarEntries;
		if(!m_useBlocks)
		{
			StorageCholesky::analyzePattern(ap, false);
			return;
		}
		const int blocks = m_blockOuter[bodies];
		m_blockRows.resize(blocks);
		m_blockColumns.resize(blocks);
		m_blocks.resize(blocks);
		m_work.resize(bodies);
		m_rowEntries.clear();
		reserveStorage(m_rowEntries, size_t(blocks - bodies));
		std::fill(m_tags.begin(), m_tags.end(), -1);
		std::fill(m_counts.begin(), m_counts.end(), 0);
		// Store each row's topological update order once. Numeric work then
		// needs neither elimination-tree traversal nor sparse index searches.
		for(int k = 0; k < bodies; ++k)
		{
			int top = bodies;
			m_tags[k] = k;
			m_rowOuter[k] = int(m_rowEntries.size());
			const int diagonal = m_blockOuter[k];
			m_blockRows[diagonal] = k;
			m_blockColumns[diagonal] = k;
			for(Sparse::InnerIterator entry(ap, 6 * k); entry; ++entry)
			{
				int i = int(entry.row()) / 6;
				if(i >= k)
					continue;
				int length = 0;
				for(; m_tags[i] != k; i = m_parent[i])
				{
					m_pattern[length++] = i;
					m_tags[i] = k;
				}
				while(length)
					m_pattern[--top] = m_pattern[--length];
			}
			for(int index = top; index < bodies; ++index)
			{
				const int i = m_pattern[index];
				const int address = m_blockOuter[i] + 1 + m_counts[i]++;
				m_blockRows[address] = k;
				m_blockColumns[address] = i;
				m_rowEntries.push_back(address);
			}
		}
		m_rowOuter[bodies] = int(m_rowEntries.size());
		m_updateOuter.resize(bodies + 1);
		m_updateOuter[0] = 0;
		for(int body = 0; body < bodies; ++body)
		{
			const int count = m_blockOuter[body + 1] - m_blockOuter[body] - 1;
			m_updateOuter[body + 1] = m_updateOuter[body] + count * (count + 1) / 2;
		}
		// The right-looking factor has higher serial cost and extra symbolic
		// storage. Use it only when parallel work repays both overheads.
		if(m_updateOuter[bodies] >= MIN_PARALLEL_FACTOR_UPDATES && m_parallelExecutor != NULL && m_parallelExecutor->workerCount() > 1)
		{
			m_updateTargets.resize(m_updateOuter[bodies]);
			for(int body = 0; body < bodies; ++body)
			{
				const int first = m_blockOuter[body] + 1;
				const int count = m_blockOuter[body + 1] - first;
				int update = m_updateOuter[body];
				for(int column = 0; column < count; ++column)
				{
					const int targetColumn = m_blockRows[first + column];
					for(int row = column; row < count; ++row)
						m_updateTargets[update++] = findBlock(targetColumn, m_blockRows[first + row]);
				}
			}
			m_inputBlocks.resize(ap.nonZeros());
			for(int column = 0; column < ap.outerSize(); ++column)
				for(int entry = ap.outerIndexPtr()[column]; entry < ap.outerIndexPtr()[column + 1]; ++entry)
				{
					const int blockColumn = ap.innerIndexPtr()[entry] / 6;
					const int blockRow = column / 6;
					m_inputBlocks[entry] = blockColumn == blockRow ? m_blockOuter[blockRow] : findBlock(blockColumn, blockRow);
				}
		}
		else
		{
			m_updateTargets.clear();
			m_inputBlocks.clear();
		}

		// Scalar storage is an export view for the established triangular
		// solve and rank-update kernels; its sparsity pattern is invariant.
		const int size = 6 * bodies;
		m_matrix.resize(size, size);
		int entries = 0;
		for(int body = 0; body < bodies; ++body)
			for(int axis = 0; axis < 6; ++axis)
			{
				m_matrix.outerIndexPtr()[6 * body + axis] = entries;
				entries += 6 * (m_blockOuter[body + 1] - m_blockOuter[body]) - axis;
			}
		m_matrix.outerIndexPtr()[size] = entries;
		m_matrix.resizeNonZeros(entries);
		int* inner = m_matrix.innerIndexPtr();
		for(int body = 0; body < bodies; ++body)
			for(int axis = 0; axis < 6; ++axis)
			{
				int entry = m_matrix.outerIndexPtr()[6 * body + axis];
				for(int row = axis; row < 6; ++row)
					inner[entry++] = 6 * body + row;
				for(int block = m_blockOuter[body] + 1; block < m_blockOuter[body + 1]; ++block)
					for(int row = 0; row < 6; ++row)
						inner[entry++] = 6 * m_blockRows[block] + row;
			}
	}

	template<bool DoLDLT>
	bool factorize(const SparseStorage& ap)
	{
		static_assert(!DoLDLT, "BlockCholesky implements LLT");
		if(!m_useBlocks)
			return StorageCholesky::factorize<false>(ap);
		if(m_parallelExecutor != NULL && !m_updateTargets.empty())
			return factorizeParallel(ap);
		const int bodies = int(ap.cols()) / 6;
		for(int body = 0; body < bodies; ++body)
			m_work[body].setZero();
		for(int k = 0; k < bodies; ++k)
		{
			// Scatter the upper input into the corresponding lower block row.
			for(int axis = 0; axis < 6; ++axis)
				for(Sparse::InnerIterator entry(ap, 6 * k + axis); entry; ++entry)
					m_work[int(entry.row()) / 6](axis, int(entry.row()) % 6) = entry.value();
			Block diagonal = m_work[k];
			m_work[k].setZero();
			for(int rowEntry = m_rowOuter[k]; rowEntry < m_rowOuter[k + 1]; ++rowEntry)
			{
				const int address = m_rowEntries[rowEntry];
				const int i = m_blockColumns[address];
				Block value = m_work[i];
				m_work[i].setZero();
				const Block& lower = m_blocks[m_blockOuter[i]];
				// Solve value * L' = work using contiguous six-value columns.
				// These fixed blocks need neither packed GEMM panels nor a transpose.
				for(int column = 0; column < 6; ++column)
				{
					for(int inner = 0; inner < column; ++inner)
						value.col(column) -= lower(column, inner) * value.col(inner);
					value.col(column) /= lower(column, column);
				}
				for(int previous = m_blockOuter[i] + 1; previous < address; ++previous)
				{
					Block& work = m_work[m_blockRows[previous]];
					const Block& factor = m_blocks[previous];
					for(int column = 0; column < 6; ++column)
					{
						Vec6 accumulated = work.col(column);
						for(int inner = 0; inner < 6; ++inner)
							accumulated -= factor(column, inner) * value.col(inner);
						work.col(column) = accumulated;
					}
				}
				// Accumulate the small symmetric product directly; only the lower
				// triangle is consumed by LLT.
				for(int column = 0; column < 6; ++column)
					for(int row = column; row < 6; ++row)
					{
						double product = 0.0;
						for(int inner = 0; inner < 6; ++inner)
							product += value(row, inner) * value(column, inner);
						diagonal(row, column) -= product;
					}
				m_blocks[address] = value;
			}
			Eigen::LLT<Block, Eigen::Lower> factor(diagonal);
			if(factor.info() != Eigen::Success)
				return scalarFallback(ap);
			m_blocks[m_blockOuter[k]] = factor.matrixL();
		}
		double* values = m_matrix.valuePtr();
		for(int body = 0; body < bodies; ++body)
			for(int axis = 0; axis < 6; ++axis)
			{
				int entry = m_matrix.outerIndexPtr()[6 * body + axis];
				const Block& diagonal = m_blocks[m_blockOuter[body]];
				for(int row = axis; row < 6; ++row)
					values[entry++] = diagonal(row, axis);
				for(int block = m_blockOuter[body] + 1; block < m_blockOuter[body + 1]; ++block)
					for(int row = 0; row < 6; ++row)
						values[entry++] = m_blocks[block](row, axis);
			}
		return true;
	}

private:
	struct ParallelUpdate
	{
		BlockCholesky* factor;
		int body;
		int first;
		int end;
		int count;
	};

	int findBlock(int column, int row) const
	{
		int begin = m_blockOuter[column], end = m_blockOuter[column + 1];
		while(begin < end)
		{
			const int middle = begin + (end - begin) / 2;
			if(m_blockRows[middle] < row)
				begin = middle + 1;
			else
				end = middle;
		}
		return begin;
	}

	void updateTrailingColumn(int body, int first, int end, int count, int localColumn)
	{
		const int column = first + localColumn;
		int update = m_updateOuter[body] + localColumn * count - localColumn * (localColumn - 1) / 2;
		for(int row = column; row < end; ++row)
		{
			Block& target = m_blocks[m_updateTargets[update++]];
			const Block& left = m_blocks[row];
			const Block& right = m_blocks[column];
			for(int axis = 0; axis < 6; ++axis)
			{
				Vec6 accumulated = target.col(axis);
				for(int inner = 0; inner < 6; ++inner)
					accumulated -= right(axis, inner) * left.col(inner);
				target.col(axis) = accumulated;
			}
		}
	}

	static void updateTrailingColumnParallel(void* context, int column)
	{
		ParallelUpdate& update = *static_cast<ParallelUpdate*>(context);
		update.factor->updateTrailingColumn(update.body, update.first, update.end, update.count, column);
	}

	bool factorizeParallel(const SparseStorage& ap)
	{
		const int bodies = int(ap.cols()) / 6;
		for(size_t block = 0; block < m_blocks.size(); ++block)
			m_blocks[block].setZero();
		for(int column = 0; column < ap.outerSize(); ++column)
			for(int entry = ap.outerIndexPtr()[column]; entry < ap.outerIndexPtr()[column + 1]; ++entry)
				m_blocks[m_inputBlocks[entry]](column % 6, ap.innerIndexPtr()[entry] % 6) = ap.valuePtr()[entry];
		for(int body = 0; body < bodies; ++body)
		{
			Eigen::LLT<Block, Eigen::Lower> factor(m_blocks[m_blockOuter[body]]);
			if(factor.info() != Eigen::Success)
			{
				m_parallelExecutor->endParallelRegion();
				return scalarFallback(ap);
			}
			const Block lower = factor.matrixL();
			m_blocks[m_blockOuter[body]] = lower;
			const int first = m_blockOuter[body] + 1;
			const int end = m_blockOuter[body + 1];
			for(int address = first; address < end; ++address)
			{
				Block value = m_blocks[address];
				for(int column = 0; column < 6; ++column)
				{
					for(int inner = 0; inner < column; ++inner)
						value.col(column) -= lower(column, inner) * value.col(inner);
					value.col(column) /= lower(column, column);
				}
				m_blocks[address] = value;
			}
			const int count = end - first;
			if(count * (count + 1) / 2 >= MIN_PARALLEL_PIVOT_UPDATES)
			{
				ParallelUpdate update;
				update.factor = this;
				update.body = body;
				update.first = first;
				update.end = end;
				update.count = count;
				m_parallelExecutor->parallelFor(count, updateTrailingColumnParallel, &update);
			}
			else
				for(int column = 0; column < count; ++column)
					updateTrailingColumn(body, first, end, count, column);
		}
		m_parallelExecutor->endParallelRegion();
		double* values = m_matrix.valuePtr();
		for(int body = 0; body < bodies; ++body)
			for(int axis = 0; axis < 6; ++axis)
			{
				int entry = m_matrix.outerIndexPtr()[6 * body + axis];
				const Block& diagonal = m_blocks[m_blockOuter[body]];
				for(int row = axis; row < 6; ++row)
					values[entry++] = diagonal(row, axis);
				for(int block = m_blockOuter[body] + 1; block < m_blockOuter[body + 1]; ++block)
					for(int row = 0; row < 6; ++row)
						values[entry++] = m_blocks[block](row, axis);
			}
		return true;
	}

	bool scalarFallback(const SparseStorage& ap)
	{
		// The established scalar LLT remains available if different rounding
		// of dense block arithmetic encounters a nonpositive pivot.
		StorageCholesky::analyzePattern(ap, false);
		return StorageCholesky::factorize<false>(ap);
	}

	bool m_useBlocks = false;
	ParallelExecutor* m_parallelExecutor = NULL;
	std::vector<int> m_parent, m_tags, m_counts, m_pattern;
	std::vector<int> m_blockOuter, m_blockRows, m_blockColumns, m_rowOuter, m_rowEntries;
	std::vector<int> m_updateOuter, m_updateTargets, m_inputBlocks;
	std::vector<Block> m_blocks, m_work;
};
}
