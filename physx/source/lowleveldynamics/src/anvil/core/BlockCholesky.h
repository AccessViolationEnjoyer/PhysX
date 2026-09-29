// Six-coordinate supernodal LLT for the solver's complete rigid-body blocks.
// Scalar storage is exported only when signed rank updates invalidate the retained blocks.
namespace anvil
{
// A 6x6 factor block in single precision, each column padded to eight rows that stay zero.
// Newton directions tolerate the rounding of a single-precision factor: the factor of a
// slightly perturbed positive definite Hessian still gives a descent direction, and the exact
// line search and convergence test work on the double-precision problem. Halving the width
// doubles the SIMD lanes of the block updates that dominate the factorization.
class alignas(32) FloatBlock
{
public:
	float& operator()(int row, int column) { return m_values[8 * column + row]; }
	float operator()(int row, int column) const { return m_values[8 * column + row]; }
	float* data() { return m_values; }
	const float* data() const { return m_values; }
	void setZero() { std::fill(m_values, m_values + 48, 0.0f); }

private:
	float m_values[48];
};

class BlockCholesky : public StorageCholesky
{
	typedef FloatBlock Block;
	enum
	{
		MIN_PARALLEL_FACTOR_UPDATES = 60000,
		MIN_PARALLEL_PIVOT_UPDATES = 64,
		// A panel groups up to this many pivots, or pivots with this many block updates. Larger
		// panels synchronize less but factor more of their own columns serially.
		PANEL_PIVOTS = 16,
		PANEL_UPDATES = 8192,
		MIN_PARALLEL_SOLVE_UPDATES = 1000000
	};
public:
	bool usesBlocks() const { return m_useBlocks; }
	bool hasCurrentBlocks() const { return m_blocksCurrent; }
	void swapInverseDiagonal(std::vector<double>& inverseDiagonal) { m_inverseDiagonal.swap(inverseDiagonal); }
	// Whether the factor has enough block updates to use helpers, on any machine.
	bool parallelSized() const { return m_useBlocks && m_updateOuter.back() >= MIN_PARALLEL_FACTOR_UPDATES; }
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
		m_blockNonzero.assign(blocks, 1);
		m_work.resize(bodies);
		m_workNonzero.assign(bodies, 0);
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
			// Ascending pivots are a topological order too, and the one in which the parallel
			// factor updates each block, so both factors round alike on every machine.
			std::sort(m_pattern.begin() + top, m_pattern.begin() + bodies);
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
			preparePanels(bodies);
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
		if(parallel())
		{
			return factorizeParallel(&ap, NULL) || scalarFallback(ap);
		}
		return factorizeSerial(&ap, NULL) || scalarFallback(ap);
	}

	// The block factors read the Hessian's body-pair blocks directly, avoiding the
	// scalar CSC export and permutation of every factorization.
	bool readsBlocks() const { return m_useBlocks; }

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
			const int row = std::max(first, second);
			BlockInput& input = m_blockInputs[m_counts[row]++];
			input.source = int(pair);
			input.target = std::min(first, second);
			input.transpose = first > second;
			// The right-looking factor loads each block in place.
			input.address = m_updateTargets.empty() ? -1 : input.target == row ? m_blockOuter[row] : findBlock(input.target, row);
		}
	}

	// Factor directly from body-pair blocks. False leaves the caller to use the
	// scalar input path, including its failed-pivot fallback.
	// nonzero marks the blocks that received any product; others are zero.
	bool factorizeBlocks(const Mat6* hessian, const unsigned char* nonzero)
	{
		m_inputNonzero = nonzero;
		return parallel() ? factorizeParallel(NULL, hessian) : factorizeSerial(NULL, hessian);
	}

private:
	struct BlockInput
	{
		int source;
		int target;
		int address;
		bool transpose;
	};

	bool parallel()
	{
		if(m_parallelExecutor == NULL || m_updateTargets.empty())
		{
			return false;
		}
		if(m_parallelWorkers == 0)
		{
			m_parallelWorkers = m_parallelExecutor->acquireWorkerCount();
		}
		return m_parallelWorkers > 1;
	}

	// Rows of a pair block belong to its second body and columns to its first.
	static void loadBlock(Block& destination, const Mat6& source, bool transpose)
	{
		if(transpose)
		{
			for(int column = 0; column < 6; ++column)
			{
				for(int row = 0; row < 6; ++row)
				{
					destination(row, column) = float(source(column, row));
				}
			}
		}
		else
		{
			for(int column = 0; column < 6; ++column)
			{
				convertColumnFloat6(destination.data() + 8 * column, source.data() + 6 * column);
			}
		}
	}

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
				m_work[coordinate >> 3](axis, coordinate & 7) = float(inputValues[entry]);
				m_workNonzero[coordinate >> 3] = 1;
			}
		}
	}

	// Copy row block k of the Hessian's body-pair blocks into the work blocks.
	void loadBlockRow(const Mat6* hessian, int k)
	{
		for(int index = m_blockInputOuter[k]; index < m_blockInputOuter[k + 1]; ++index)
		{
			const BlockInput& input = m_blockInputs[index];
			Block& work = m_work[input.target];
			const Mat6& source = hessian[input.source];
			// Body pairs whose contacts carry no curvature, such as untouched neighbours,
			// leave zero blocks; the work block stays zero and the factor skips it.
			if(!m_inputNonzero[input.source])
			{
				continue;
			}
			m_workNonzero[input.target] = 1;
			loadBlock(work, source, input.transpose);
		}
	}

	// Left-looking block factor reading row block k from the Hessian blocks when
	// given, otherwise from the scalar input. A failed pivot clears the work
	// blocks and returns false.
	bool factorizeSerial(const SparseStorage* ap, const Mat6* hessian)
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
				// A block that received no input and no update is a zero factor block: it
				// contributes nothing to later blocks of this row or to the diagonal.
				if(!m_workNonzero[i])
				{
					m_blocks[address].setZero();
					m_blockNonzero[address] = 0;
					continue;
				}
				m_workNonzero[i] = 0;
				m_blockNonzero[address] = 1;
				Block& value = m_work[i];
				// Solve value * L' = work using contiguous columns.
				// These fixed blocks need neither packed GEMM panels nor a transpose.
				solveTransposedLowerFloat6(value.data(), m_blocks[m_blockOuter[i]].data(), &m_inverseDiagonal[6 * i]);
				for(int previous = m_blockOuter[i] + 1; previous < address; ++previous)
				{
					if(!m_blockNonzero[previous])
					{
						continue;
					}
					const int row = m_blockRows[previous];
					subtractProductFloat6(m_work[row].data(), value.data(), m_blocks[previous].data());
					m_workNonzero[row] = 1;
				}
				// The whole symmetric product costs no more than its lower half in these kernels.
				subtractProductFloat6(diagonal.data(), value.data(), value.data());
				m_blocks[address] = value;
				value.setZero();
			}
			if(!factorDiagonal(diagonal, m_blocks[m_blockOuter[k]], &m_inverseDiagonal[6 * k]))
			{
				// Successful columns consume their work blocks. A failed pivot is
				// the only path that leaves pending values for the next solve.
				for(int body = 0; body < bodies; ++body)
				{
					m_work[body].setZero();
					m_workNonzero[body] = 0;
				}
				return false;
			}
			m_workNonzero[k] = 0;
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
			subtractFloatMatrixVector6(current.data(), m_blocks[block].data(), solved.data());
		}
		// Pivots multiply by the reciprocals of the stored pivots rather than dividing.
		const Block& diagonal = m_blocks[m_blockOuter[body]];
		const double* inverse = &m_inverseDiagonal[6 * body];
		ANVIL_UNROLL
		for(int column = 0; column < 6; ++column)
		{
			current[column] *= inverse[column];
			ANVIL_UNROLL
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
				current[axis] -= dotFloat6(factor.data() + 8 * axis, solved.data());
			}
		}
		const Block& diagonal = m_blocks[m_blockOuter[body]];
		const double* inverse = &m_inverseDiagonal[6 * body];
		ANVIL_UNROLL
		for(int column = 5; column >= 0; --column)
		{
			ANVIL_UNROLL
			for(int row = column + 1; row < 6; ++row)
			{
				current[column] -= diagonal(row, column) * current[row];
			}
			current[column] *= inverse[column];
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
			subtractProductFloat6(target.data(), left.data(), right.data());
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

	// Consecutive pivots form panels. A panel factors its pivots in order and applies their
	// updates to the panel's own later columns serially; the updates to every later column are
	// then applied in one parallel region, one task per target column, instead of one region per
	// pivot. Each target block still receives its updates in pivot order, as in the serial factor.
	void preparePanels(int bodies)
	{
		m_panelOuter.clear();
		m_panelTaskOuter.clear();
		m_panelTaskItemOuter.clear();
		m_panelItems.clear();
		m_panelOuter.push_back(0);
		m_panelTaskOuter.push_back(0);
		m_panelTaskItemOuter.push_back(0);
		std::vector<int>& targetCounts = m_tags;
		std::fill(targetCounts.begin(), targetCounts.end(), 0);
		int begin = 0;
		while(begin < bodies)
		{
			int end = begin;
			int work = 0;
			while(end < bodies && end - begin < PANEL_PIVOTS && work < PANEL_UPDATES)
			{
				const int count = m_blockOuter[end + 1] - m_blockOuter[end] - 1;
				work += count * (count + 1) / 2;
				++end;
			}
			m_panelOuter.push_back(end);
			// Group the panel's updates of later columns by target column, pivots ascending.
			int taskCount = 0;
			for(int pivot = begin; pivot < end; ++pivot)
			{
				for(int address = m_blockOuter[pivot] + 1; address < m_blockOuter[pivot + 1]; ++address)
				{
					const int target = m_blockRows[address];
					if(target >= end && targetCounts[target]++ == 0)
					{
						m_pattern[taskCount++] = target;
					}
				}
			}
			std::sort(m_pattern.begin(), m_pattern.begin() + taskCount);
			const std::uint32_t itemBase = std::uint32_t(m_panelItems.size());
			int itemCursor = int(itemBase);
			for(int task = 0; task < taskCount; ++task)
			{
				const int target = m_pattern[task];
				itemCursor += targetCounts[target];
				m_panelTaskItemOuter.push_back(itemCursor);
				targetCounts[target] = itemCursor - targetCounts[target];
			}
			m_panelItems.resize(itemCursor);
			for(int pivot = begin; pivot < end; ++pivot)
			{
				const int first = m_blockOuter[pivot] + 1;
				for(int address = first; address < m_blockOuter[pivot + 1]; ++address)
				{
					const int target = m_blockRows[address];
					if(target >= end)
					{
						PanelItem& item = m_panelItems[targetCounts[target]++];
						item.pivot = pivot;
						item.local = address - first;
					}
				}
			}
			for(int task = 0; task < taskCount; ++task)
			{
				targetCounts[m_pattern[task]] = 0;
			}
			m_panelTaskOuter.push_back(m_panelTaskOuter.back() + taskCount);
			begin = end;
		}
	}

	struct PanelItem
	{
		int pivot;
		int local;
	};

	void updatePanelTarget(int task)
	{
		for(int item = m_panelTaskItemOuter[task]; item < m_panelTaskItemOuter[task + 1]; ++item)
		{
			const PanelItem& update = m_panelItems[item];
			const int first = m_blockOuter[update.pivot] + 1;
			const int end = m_blockOuter[update.pivot + 1];
			updateTrailingColumn(update.pivot, first, end, end - first, update.local);
		}
	}

	struct PanelUpdate
	{
		BlockCholesky* factor;
		int firstTask;
	};

	static void updatePanelTargetParallel(void* context, int index)
	{
		PanelUpdate& update = *static_cast<PanelUpdate*>(context);
		update.factor->updatePanelTarget(update.firstTask + index);
	}

	// Zero row block k of the factor and scatter input column block k into it. Row blocks are
	// disjoint, so bodies load in parallel.
	void loadInputRow(const SparseStorage& ap, int body)
	{
		m_blocks[m_blockOuter[body]].setZero();
		for(int entry = m_rowOuter[body]; entry < m_rowOuter[body + 1]; ++entry)
		{
			m_blocks[m_rowEntries[entry]].setZero();
		}
		const int* inputOuter = ap.outerIndexPtr();
		const double* inputValues = ap.valuePtr();
		for(int column = 6 * body; column < 6 * body + 6; ++column)
		{
			const int end = inputOuter[column + 1];
			for(int entry = inputOuter[column]; entry < end; ++entry)
			{
				m_blocks[m_inputBlocks[entry]](column % 6, m_inputCoordinates[entry] & 7) = float(inputValues[entry]);
			}
		}
	}

	// Zero row block k of the factor and copy the Hessian's pair blocks of row k into it.
	void loadHessianRow(const Mat6* hessian, int body)
	{
		m_blocks[m_blockOuter[body]].setZero();
		for(int entry = m_rowOuter[body]; entry < m_rowOuter[body + 1]; ++entry)
		{
			m_blocks[m_rowEntries[entry]].setZero();
		}
		for(int index = m_blockInputOuter[body]; index < m_blockInputOuter[body + 1]; ++index)
		{
			const BlockInput& input = m_blockInputs[index];
			if(m_inputNonzero[input.source])
			{
				loadBlock(m_blocks[input.address], hessian[input.source], input.transpose);
			}
		}
	}

	struct InputLoad
	{
		BlockCholesky* factor;
		const SparseStorage* input;
		const Mat6* hessian;
	};

	static void loadInputRowParallel(void* context, int body)
	{
		InputLoad& load = *static_cast<InputLoad*>(context);
		if(load.hessian)
		{
			load.factor->loadHessianRow(load.hessian, body);
		}
		else
		{
			load.factor->loadInputRow(*load.input, body);
		}
	}

	// Factor a diagonal block's lower triangle in double precision into its single-precision
	// factor and the reciprocals of the stored pivots; false for a nonpositive pivot.
	// Each pivot is rounded to single precision before its column is scaled, so the column,
	// the stored factor and the returned reciprocal all use the stored pivot.
	static bool factorDiagonal(const Block& input, Block& output, double* inverseDiagonal)
	{
		double lower[6][6];
		ANVIL_UNROLL
		for(int column = 0; column < 6; ++column)
		{
			double diagonal = input(column, column);
			ANVIL_UNROLL
			for(int inner = 0; inner < column; ++inner)
			{
				diagonal -= lower[inner][column] * lower[inner][column];
			}
			if(diagonal <= 0.0)
			{
				return false;
			}
			const float pivot = float(std::sqrt(diagonal));
			const double inverse = 1.0 / double(pivot);
			inverseDiagonal[column] = inverse;
			float* stored = output.data() + 8 * column;
			ANVIL_UNROLL
			for(int row = 0; row < column; ++row)
			{
				stored[row] = 0.0f;
			}
			stored[column] = pivot;
			ANVIL_UNROLL
			for(int row = column + 1; row < 6; ++row)
			{
				double value = input(row, column);
				ANVIL_UNROLL
				for(int inner = 0; inner < column; ++inner)
				{
					value -= lower[inner][row] * lower[inner][column];
				}
				lower[column][row] = value * inverse;
				stored[row] = float(lower[column][row]);
			}
			stored[6] = stored[7] = 0.0f;
		}
		return true;
	}

	// Factor one pivot's diagonal block and scale its column; false for a nonpositive pivot.
	bool factorPivot(int body)
	{
		Block& diagonal = m_blocks[m_blockOuter[body]];
		if(!factorDiagonal(diagonal, diagonal, &m_inverseDiagonal[6 * body]))
		{
			return false;
		}
		const int first = m_blockOuter[body] + 1;
		const int end = m_blockOuter[body + 1];
		for(int address = first; address < end; ++address)
		{
			solveTransposedLowerFloat6(m_blocks[address].data(), diagonal.data(), &m_inverseDiagonal[6 * body]);
		}
		return true;
	}

	// Reads the scalar input when given, otherwise the Hessian blocks. A failed pivot
	// returns false.
	bool factorizeParallel(const SparseStorage* ap, const Mat6* hessian)
	{
		const int bodies = int(m_blockOuter.size()) - 1;
		InputLoad load = { this, ap, hessian };
		m_parallelExecutor->parallelFor(bodies, loadInputRowParallel, &load);
		const int panels = int(m_panelOuter.size()) - 1;
		for(int panel = 0; panel < panels; ++panel)
		{
			const int begin = m_panelOuter[panel], end = m_panelOuter[panel + 1];
			for(int body = begin; body < end; ++body)
			{
				if(!factorPivot(body))
				{
					m_parallelWorkers = 1;
					return false;
				}
				// Complete the panel's own later columns before they are factored.
				const int first = m_blockOuter[body] + 1;
				const int last = m_blockOuter[body + 1];
				const int count = last - first;
				for(int column = 0; column < count && m_blockRows[first + column] < end; ++column)
				{
					updateTrailingColumn(body, first, last, count, column);
				}
			}
			const int firstTask = m_panelTaskOuter[panel];
			const int taskCount = m_panelTaskOuter[panel + 1] - firstTask;
			const int updates = m_updateOuter[end] - m_updateOuter[begin];
			if(updates >= MIN_PARALLEL_PIVOT_UPDATES && taskCount > 1)
			{
				PanelUpdate update = { this, firstTask };
				m_parallelExecutor->parallelFor(taskCount, updatePanelTargetParallel, &update);
			}
			else
			{
				for(int task = 0; task < taskCount; ++task)
				{
					updatePanelTarget(firstTask + task);
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
	std::vector<int> m_panelOuter, m_panelTaskOuter, m_panelTaskItemOuter;
	std::vector<PanelItem> m_panelItems;
	std::vector<Block> m_blocks, m_work;
	// Serial factor only: blocks known to be nonzero, and work blocks holding values.
	std::vector<unsigned char> m_blockNonzero, m_workNonzero;
	const unsigned char* m_inputNonzero = NULL;
};
}
