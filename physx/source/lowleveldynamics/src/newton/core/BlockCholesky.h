// Six-coordinate supernodal LLT for the solver's complete rigid-body blocks.
// Scalar storage is exported only when signed rank updates invalidate the retained blocks.
namespace newton
{
class BlockCholesky : public StorageCholesky
{
	typedef Mat6 Block;
	enum
	{
		MIN_PARALLEL_FACTOR_UPDATES = 60000,
		MIN_PARALLEL_PIVOT_UPDATES = 64,
		MIN_PARALLEL_SOLVE_UPDATES = 1000000
	};
public:
	bool usesBlocks() const { return m_useBlocks; }
	bool hasCurrentBlocks() const { return m_blocksCurrent; }
	void swapInverseDiagonal(std::vector<double>& inverseDiagonal) { m_inverseDiagonal.swap(inverseDiagonal); }
	int parallelWorkerCount() const { return m_parallelWorkers; }
	void setParallelExecutor(ParallelExecutor* executor) { m_parallelExecutor = executor; m_parallelWorkers = 0; }

	void beginScalarUpdates()
	{
		ensureScalarFactor();
		m_blocksCurrent = false;
	}

	void solveBlocks(double* solution)
	{
		if(m_parallelExecutor == NULL || m_parallelWorkers < 2 || m_updateOuter.back() < MIN_PARALLEL_SOLVE_UPDATES)
		{
			solveBlocksSerial(solution);
			return;
		}
		ParallelSolve solve;
		solve.factor = this;
		solve.solution = solution;
		const int levels = int(m_solveLevelOuter.size()) - 1;
		for(int level = 0; level < levels; ++level)
		{
			solve.first = m_solveLevelOuter[level];
			m_parallelExecutor->parallelFor(m_solveLevelOuter[level + 1] - solve.first, solveForwardParallel, &solve);
		}
		for(int level = levels - 1; level >= 0; --level)
		{
			solve.first = m_solveLevelOuter[level];
			m_parallelExecutor->parallelFor(m_solveLevelOuter[level + 1] - solve.first, solveBackwardParallel, &solve);
		}
	}

	void analyzePattern(const SparseStorage& ap)
	{
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
				{
					continue;
				}
				for(; m_tags[i] != k; i = m_parent[i])
				{
					if(m_parent[i] == -1)
					{
						m_parent[i] = k;
					}
					++m_counts[i];
					m_tags[i] = k;
				}
			}
		}
		m_blockOuter[0] = 0;
		for(int k = 0; k < bodies; ++k)
		{
			m_blockOuter[k + 1] = m_blockOuter[k] + m_counts[k] + 1;
		}
		// Block kernels read the Hessian's body-pair blocks directly, and measured faster
		// than the scalar LLT even for thin chains, so every analyzed pattern uses them.
		// The scalar LLT remains the fallback for a nonpositive block pivot.
		m_useBlocks = true;
		const int blocks = m_blockOuter[bodies];
		m_blockRows.resize(blocks);
		m_blockColumns.resize(blocks);
		m_blocks.resize(blocks);
		m_work.resize(bodies);
		m_inverseDiagonal.resize(6 * bodies);
		for(int body = 0; body < bodies; ++body)
		{
			m_work[body].setZero();
		}
		m_inputCoordinates.resize(ap.nonZeros());
		const int* inputInner = ap.innerIndexPtr();
		for(int entry = 0; entry < ap.nonZeros(); ++entry)
		{
			const int row = inputInner[entry];
			m_inputCoordinates[entry] = (row / 6 << 3) | row % 6;
		}
		m_rowEntries.clear();
		reserveStorage(m_rowEntries, std::uint32_t(blocks - bodies));
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
				{
					continue;
				}
				int length = 0;
				for(; m_tags[i] != k; i = m_parent[i])
				{
					m_pattern[length++] = i;
					m_tags[i] = k;
				}
				while(length)
				{
					m_pattern[--top] = m_pattern[--length];
				}
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
		m_solveLevels.resize(bodies);
		int levelCount = 0;
		for(int body = 0; body < bodies; ++body)
		{
			int level = 0;
			for(int entry = m_rowOuter[body]; entry < m_rowOuter[body + 1]; ++entry)
			{
				level = std::max(level, m_solveLevels[m_blockColumns[m_rowEntries[entry]]] + 1);
			}
			m_solveLevels[body] = level;
			levelCount = std::max(levelCount, level + 1);
		}
		m_solveLevelOuter.assign(levelCount + 1, 0);
		for(int body = 0; body < bodies; ++body)
		{
			++m_solveLevelOuter[m_solveLevels[body] + 1];
		}
		for(int level = 0; level < levelCount; ++level)
		{
			m_solveLevelOuter[level + 1] += m_solveLevelOuter[level];
		}
		m_solveBodies.resize(bodies);
		m_counts.assign(m_solveLevelOuter.begin(), m_solveLevelOuter.end() - 1);
		for(int body = 0; body < bodies; ++body)
		{
			m_solveBodies[m_counts[m_solveLevels[body]]++] = body;
		}
		m_updateOuter.resize(bodies + 1);
		m_updateOuter[0] = 0;
		for(int body = 0; body < bodies; ++body)
		{
			const int count = m_blockOuter[body + 1] - m_blockOuter[body] - 1;
			m_updateOuter[body + 1] = m_updateOuter[body] + count * (count + 1) / 2;
		}
		// The right-looking factor has higher serial cost and extra symbolic
		// storage. Use it only when parallel work repays both overheads.
		if(m_updateOuter[bodies] >= MIN_PARALLEL_FACTOR_UPDATES && m_parallelExecutor != NULL && m_parallelExecutor->workerCapacity() > 1)
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
					{
						m_updateTargets[update++] = findBlock(targetColumn, m_blockRows[first + row]);
					}
				}
			}
			m_inputBlocks.resize(ap.nonZeros());
			const int columnCount = ap.outerSize();
			const int* inputOuter = ap.outerIndexPtr();
			for(int column = 0; column < columnCount; ++column)
			{
				const int end = inputOuter[column + 1];
				for(int entry = inputOuter[column]; entry < end; ++entry)
				{
					const int blockColumn = inputInner[entry] / 6;
					const int blockRow = column / 6;
					m_inputBlocks[entry] = blockColumn == blockRow ? m_blockOuter[blockRow] : findBlock(blockColumn, blockRow);
				}
			}
		}
		else
		{
			m_updateTargets.clear();
			m_inputBlocks.clear();
		}

		// Scalar storage is an export view for signed rank updates; its
		// sparsity pattern is invariant.
		const int size = 6 * bodies;
		m_matrix.resize(size, size);
		int entries = 0;
		for(int body = 0; body < bodies; ++body)
		{
			for(int axis = 0; axis < 6; ++axis)
			{
				m_matrix.outerIndexPtr()[6 * body + axis] = entries;
				entries += 6 * (m_blockOuter[body + 1] - m_blockOuter[body]) - axis;
			}
		}
		m_matrix.outerIndexPtr()[size] = entries;
		m_matrix.resizeNonZeros(entries);
		int* inner = m_matrix.innerIndexPtr();
		for(int body = 0; body < bodies; ++body)
		{
			for(int axis = 0; axis < 6; ++axis)
			{
				int entry = m_matrix.outerIndexPtr()[6 * body + axis];
				for(int row = axis; row < 6; ++row)
				{
					inner[entry++] = 6 * body + row;
				}
				for(int block = m_blockOuter[body] + 1; block < m_blockOuter[body + 1]; ++block)
				{
					for(int row = 0; row < 6; ++row)
					{
						inner[entry++] = 6 * m_blockRows[block] + row;
					}
				}
			}
		}
	}

	bool factorize(const SparseStorage& ap)
	{
		if(!m_useBlocks)
		{
			m_blocksCurrent = false;
			m_scalarCurrent = true;
			return StorageCholesky::factorize(ap);
		}
		if(m_parallelExecutor != NULL && !m_updateTargets.empty())
		{
			if(m_parallelWorkers == 0)
			{
				m_parallelWorkers = m_parallelExecutor->acquireWorkerCount();
			}
			if(m_parallelWorkers > 1)
			{
				return factorizeParallel(ap);
			}
		}
		return factorizeSerial(&ap, NULL) || scalarFallback(ap);
	}

	// The serial block factor can read the Hessian's body-pair blocks directly,
	// avoiding the scalar CSC export and permutation of every factorization.
	bool readsBlocks() const { return m_useBlocks && m_updateTargets.empty(); }

	// pairs are the Hessian's sorted (first, second) body pairs; permutation maps
	// a body's first coordinate to its factor coordinate.
	void prepareBlockInput(const std::vector<std::uint64_t>& pairs, const std::vector<int>& permutation)
	{
		if(!m_useBlocks)
		{
			return;
		}
		const int bodies = int(m_blockOuter.size()) - 1;
		const std::uint32_t pairCount = std::uint32_t(pairs.size());
		m_blockInputOuter.assign(bodies + 1, 0);
		for(std::uint32_t pair = 0; pair < pairCount; ++pair)
		{
			const int first = permutation[6 * int(pairs[pair] >> 32)] / 6;
			const int second = permutation[6 * int(std::uint32_t(pairs[pair]))] / 6;
			++m_blockInputOuter[std::max(first, second) + 1];
		}
		for(int body = 0; body < bodies; ++body)
		{
			m_blockInputOuter[body + 1] += m_blockInputOuter[body];
		}
		m_counts.assign(m_blockInputOuter.begin(), m_blockInputOuter.end() - 1);
		m_blockInputs.resize(pairCount);
		for(std::uint32_t pair = 0; pair < pairCount; ++pair)
		{
			const int first = permutation[6 * int(pairs[pair] >> 32)] / 6;
			const int second = permutation[6 * int(std::uint32_t(pairs[pair]))] / 6;
			// A pair block holds rows of its second body and columns of its first.
			// Factor column k reads row block k, so a second body factored first
			// supplies the transposed block.
			BlockInput& input = m_blockInputs[m_counts[std::max(first, second)]++];
			input.source = int(pair);
			input.target = std::min(first, second);
			input.transpose = first > second;
		}
	}

	// Factor directly from body-pair blocks. False leaves the caller to use the
	// scalar input path, including its failed-pivot fallback.
	bool factorizeBlocks(const Block* hessian)
	{
		return factorizeSerial(NULL, hessian);
	}

private:
	struct BlockInput
	{
		int source;
		int target;
		bool transpose;
	};

	// Scatter row block k of the upper scalar input into the lower work blocks.
	void scatterInputRow(const SparseStorage& ap, int k)
	{
		const int* inputOuter = ap.outerIndexPtr();
		const double* inputValues = ap.valuePtr();
		for(int axis = 0; axis < 6; ++axis)
		{
			const int column = 6 * k + axis;
			const int end = inputOuter[column + 1];
			for(int entry = inputOuter[column]; entry < end; ++entry)
			{
				const int coordinate = m_inputCoordinates[entry];
				m_work[coordinate >> 3].data()[6 * (coordinate & 7) + axis] = inputValues[entry];
			}
		}
	}

	// Copy row block k of the Hessian's body-pair blocks into the work blocks.
	void loadBlockRow(const Block* hessian, int k)
	{
		for(int index = m_blockInputOuter[k]; index < m_blockInputOuter[k + 1]; ++index)
		{
			const BlockInput& input = m_blockInputs[index];
			Block& work = m_work[input.target];
			const Block& source = hessian[input.source];
			if(input.transpose)
			{
				for(int column = 0; column < 6; ++column)
				{
					for(int row = 0; row < 6; ++row)
					{
						work(row, column) = source(column, row);
					}
				}
			}
			else
			{
				work = source;
			}
		}
	}

	// Left-looking block factor reading row block k from the Hessian blocks when
	// given, otherwise from the scalar input. A failed pivot clears the work
	// blocks and returns false.
	bool factorizeSerial(const SparseStorage* ap, const Block* hessian)
	{
		const int bodies = int(m_blockOuter.size()) - 1;
		for(int k = 0; k < bodies; ++k)
		{
			if(hessian)
			{
				loadBlockRow(hessian, k);
			}
			else
			{
				scatterInputRow(*ap, k);
			}
			Block& diagonal = m_work[k];
			for(int rowEntry = m_rowOuter[k]; rowEntry < m_rowOuter[k + 1]; ++rowEntry)
			{
				const int address = m_rowEntries[rowEntry];
				const int i = m_blockColumns[address];
				Block& value = m_work[i];
				const Block& lower = m_blocks[m_blockOuter[i]];
				// Solve value * L' = work using contiguous six-value columns.
				// These fixed blocks need neither packed GEMM panels nor a transpose.
				for(int column = 0; column < 6; ++column)
				{
					for(int inner = 0; inner < column; ++inner)
					{
						subtractScaled6(value.data() + 6 * column, value.data() + 6 * inner, lower(column, inner));
					}
					const double inverse = m_inverseDiagonal[6 * i + column];
					for(int row = 0; row < 6; ++row)
					{
						value(row, column) *= inverse;
					}
				}
				for(int previous = m_blockOuter[i] + 1; previous < address; ++previous)
				{
					Block& work = m_work[m_blockRows[previous]];
					const Block& factor = m_blocks[previous];
					subtractProduct6(work.data(), value.data(), factor.data());
				}
				subtractLowerOuterProduct6(diagonal.data(), value.data());
				m_blocks[address] = value;
				value.setZero();
			}
			Block lower;
			if(!cholesky6(diagonal, lower))
			{
				// Successful columns consume their work blocks. A failed pivot is
				// the only path that leaves pending values for the next solve.
				for(int body = 0; body < bodies; ++body)
				{
					m_work[body].setZero();
				}
				return false;
			}
			m_blocks[m_blockOuter[k]] = lower;
			for(int column = 0; column < 6; ++column)
			{
				m_inverseDiagonal[6 * k + column] = 1.0 / lower(column, column);
			}
			diagonal.setZero();
		}
		m_blocksCurrent = true;
		m_scalarCurrent = false;
		return true;
	}

	struct ParallelUpdate
	{
		BlockCholesky* factor;
		int body;
		int first;
		int end;
		int count;
	};

	struct ParallelSolve
	{
		BlockCholesky* factor;
		double* solution;
		int first;
	};

	void solveForwardBody(double* solution, int body) const
	{
		Vec6 current = loadVector<6>(solution + 6 * body);
		for(int entry = m_rowOuter[body]; entry < m_rowOuter[body + 1]; ++entry)
		{
			const int block = m_rowEntries[entry];
			const Vec6 solved = loadVector<6>(solution + 6 * m_blockColumns[block]);
			subtractMatrixVector6(current.data(), m_blocks[block].data(), solved.data());
		}
		const Block& diagonal = m_blocks[m_blockOuter[body]];
		for(int column = 0; column < 6; ++column)
		{
			current[column] /= diagonal(column, column);
			for(int row = column + 1; row < 6; ++row)
			{
				current[row] -= diagonal(row, column) * current[column];
			}
		}
		storeVector<6>(solution + 6 * body, current);
	}

	void solveBackwardBody(double* solution, int body) const
	{
		Vec6 current = loadVector<6>(solution + 6 * body);
		for(int block = m_blockOuter[body] + 1; block < m_blockOuter[body + 1]; ++block)
		{
			const Vec6 solved = loadVector<6>(solution + 6 * m_blockRows[block]);
			const Block& factor = m_blocks[block];
			for(int axis = 0; axis < 6; ++axis)
			{
				current[axis] -= dot6(factor.data() + 6 * axis, solved.data());
			}
		}
		const Block& diagonal = m_blocks[m_blockOuter[body]];
		for(int column = 5; column >= 0; --column)
		{
			for(int row = column + 1; row < 6; ++row)
			{
				current[column] -= diagonal(row, column) * current[row];
			}
			current[column] /= diagonal(column, column);
		}
		storeVector<6>(solution + 6 * body, current);
	}

	void solveBlocksSerial(double* solution) const
	{
		const int bodies = int(m_blockOuter.size()) - 1;
		for(int body = 0; body < bodies; ++body)
		{
			solveForwardBody(solution, body);
		}
		for(int body = bodies - 1; body >= 0; --body)
		{
			solveBackwardBody(solution, body);
		}
	}

	static void solveForwardParallel(void* context, int index)
	{
		ParallelSolve& solve = *static_cast<ParallelSolve*>(context);
		const int body = solve.factor->m_solveBodies[solve.first + index];
		solve.factor->solveForwardBody(solve.solution, body);
	}

	static void solveBackwardParallel(void* context, int index)
	{
		ParallelSolve& solve = *static_cast<ParallelSolve*>(context);
		const int body = solve.factor->m_solveBodies[solve.first + index];
		solve.factor->solveBackwardBody(solve.solution, body);
	}

	int findBlock(int column, int row) const
	{
		int begin = m_blockOuter[column], end = m_blockOuter[column + 1];
		while(begin < end)
		{
			const int middle = begin + (end - begin) / 2;
			if(m_blockRows[middle] < row)
			{
				begin = middle + 1;
			}
			else
			{
				end = middle;
			}
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
			subtractProduct6(target.data(), left.data(), right.data());
		}
	}

	static void updateTrailingColumnParallel(void* context, int column)
	{
		ParallelUpdate& update = *static_cast<ParallelUpdate*>(context);
		update.factor->updateTrailingColumn(update.body, update.first, update.end, update.count, column);
	}

	void exportBody(int body)
	{
		double* values = m_matrix.valuePtr();
		for(int axis = 0; axis < 6; ++axis)
		{
			int entry = m_matrix.outerIndexPtr()[6 * body + axis];
			const Block& diagonal = m_blocks[m_blockOuter[body]];
			for(int row = axis; row < 6; ++row)
			{
				values[entry++] = diagonal(row, axis);
			}
			for(int block = m_blockOuter[body] + 1; block < m_blockOuter[body + 1]; ++block)
			{
				for(int row = 0; row < 6; ++row)
				{
					values[entry++] = m_blocks[block](row, axis);
				}
			}
		}
	}

	static void exportBodyParallel(void* context, int body)
	{
		static_cast<BlockCholesky*>(context)->exportBody(body);
	}

	void ensureScalarFactor()
	{
		if(m_scalarCurrent)
		{
			return;
		}
		const int bodies = int(m_blockOuter.size()) - 1;
		if(m_parallelExecutor != NULL && m_parallelWorkers > 1)
		{
			m_parallelExecutor->parallelFor(bodies, exportBodyParallel, this);
		}
		else
		{
			for(int body = 0; body < bodies; ++body)
			{
				exportBody(body);
			}
		}
		m_scalarCurrent = true;
	}

	bool factorizeParallel(const SparseStorage& ap)
	{
		const int bodies = int(ap.cols()) / 6;
		const std::uint32_t blockCount = std::uint32_t(m_blocks.size());
		for(std::uint32_t block = 0; block < blockCount; ++block)
		{
			m_blocks[block].setZero();
		}
		const int columnCount = ap.outerSize();
		const int* inputOuter = ap.outerIndexPtr();
		const double* inputValues = ap.valuePtr();
		for(int column = 0; column < columnCount; ++column)
		{
			const int end = inputOuter[column + 1];
			for(int entry = inputOuter[column]; entry < end; ++entry)
			{
				m_blocks[m_inputBlocks[entry]](column % 6, m_inputCoordinates[entry] & 7) = inputValues[entry];
			}
		}
		for(int body = 0; body < bodies; ++body)
		{
			Block lower;
			if(!cholesky6(m_blocks[m_blockOuter[body]], lower))
			{
				m_parallelWorkers = 1;
				return scalarFallback(ap);
			}
			m_blocks[m_blockOuter[body]] = lower;
			double inverseDiagonal[6];
			for(int column = 0; column < 6; ++column)
			{
				inverseDiagonal[column] = 1.0 / lower(column, column);
				m_inverseDiagonal[6 * body + column] = inverseDiagonal[column];
			}
			const int first = m_blockOuter[body] + 1;
			const int end = m_blockOuter[body + 1];
			for(int address = first; address < end; ++address)
			{
				Block& value = m_blocks[address];
				for(int column = 0; column < 6; ++column)
				{
					for(int inner = 0; inner < column; ++inner)
					{
						subtractScaled6(value.data() + 6 * column, value.data() + 6 * inner, lower(column, inner));
					}
					for(int row = 0; row < 6; ++row)
					{
						value(row, column) *= inverseDiagonal[column];
					}
				}
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
			{
				for(int column = 0; column < count; ++column)
				{
					updateTrailingColumn(body, first, end, count, column);
				}
			}
		}
		m_blocksCurrent = true;
		m_scalarCurrent = false;
		return true;
	}

	bool scalarFallback(const SparseStorage& ap)
	{
		// The established scalar LLT remains available if different rounding
		// of dense block arithmetic encounters a nonpositive pivot.
		StorageCholesky::analyzePattern(ap);
		m_blocksCurrent = false;
		m_scalarCurrent = true;
		return StorageCholesky::factorize(ap);
	}

	bool m_useBlocks = false;
	bool m_blocksCurrent = false;
	bool m_scalarCurrent = false;
	ParallelExecutor* m_parallelExecutor = NULL;
	int m_parallelWorkers = 0;
	std::vector<int> m_parent, m_tags, m_counts, m_pattern;
	std::vector<int> m_blockOuter, m_blockRows, m_blockColumns, m_rowOuter, m_rowEntries, m_inputCoordinates;
	std::vector<double> m_inverseDiagonal;
	std::vector<int> m_solveLevels, m_solveLevelOuter, m_solveBodies;
	std::vector<int> m_updateOuter, m_updateTargets, m_inputBlocks, m_blockInputOuter;
	std::vector<BlockInput> m_blockInputs;
	std::vector<Block> m_blocks, m_work;
};
}
