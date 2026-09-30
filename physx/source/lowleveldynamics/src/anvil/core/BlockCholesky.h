// Six-coordinate supernodal LLT for the solver's complete rigid-body blocks.
// Scalar storage is exported only when signed rank updates invalidate the retained blocks.
namespace anvil
{
// A 6x6 factor block in single precision, in the layout of the kernels' vector width.
// Newton directions tolerate the rounding of a single-precision factor: the factor of a
// slightly perturbed positive definite Hessian still gives a descent direction, and the exact
// line search and convergence test work on the double-precision problem. Halving the width
// doubles the SIMD lanes of the block updates that dominate the factorization.
class alignas(FLOAT_BLOCK_ALIGNMENT) FloatBlock
{
public:
	float& operator()(int row, int column) { return m_values[floatBlockIndex(row, column)]; }
	float operator()(int row, int column) const { return m_values[floatBlockIndex(row, column)]; }
	float* data() { return m_values; }
	const float* data() const { return m_values; }
	void setZero() { std::fill(m_values, m_values + FLOAT_BLOCK_SIZE, 0.0f); }
	// Clears a padded column's unused rows; packed blocks have none.
	void clearPadding(int column)
	{
#if defined(ANVIL_PACKED_FLOAT_BLOCKS)
		(void)column;
#else
		m_values[8 * column + 6] = m_values[8 * column + 7] = 0.0f;
#endif
	}

private:
	float m_values[FLOAT_BLOCK_SIZE];
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
		MIN_PARALLEL_SOLVE_UPDATES = 60000,
		// Chains of the elimination tree up to this depth are solved in stages; the subtrees
		// below them are one task each.
		SOLVE_STAGES = 4,
		// A stage's work is split into at most this many tasks, of at least this many blocks,
		// so helpers claim a few tasks of similar size rather than one for every body.
		SOLVE_TASKS = 16,
		SOLVE_TASK_BLOCKS = 256,
		// A factor with this many block updates costs many solves with passes over the rows, so
		// a solve takes steps on the factor it has before it refactors.
		MIN_RETAINED_STEP_UPDATES = 60000
	};
public:
	bool usesBlocks() const { return m_useBlocks; }
	bool hasCurrentBlocks() const { return m_blocksCurrent; }
	void swapInverseDiagonal(std::vector<double>& inverseDiagonal) { m_inverseDiagonal.swap(inverseDiagonal); }
	// Whether a refactorization costs enough to take steps on the retained factor instead.
	bool retainedStepSized() const { return m_useBlocks && m_blocksCurrent && m_updateOuter.back() >= MIN_RETAINED_STEP_UPDATES; }
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
		if(m_parallelExecutor == NULL || m_parallelWorkers < 2 || m_stageGroupOuter.empty())
		{
			solveBlocksSerial(solution);
			return;
		}
		ParallelSolve solve;
		solve.factor = this;
		solve.solution = solution;
		// Forward: the subtrees, then each depth's chains from the deepest. A chain's bodies
		// first gather the blocks of deeper chains, every body on its own, and then the chain
		// solves in order.
		solveTasks(solve, m_groupTaskOuter, SOLVE_STAGES, solveForwardGroups);
		for(int stage = SOLVE_STAGES - 1; stage >= 0; --stage)
		{
			solveTasks(solve, m_gatherTaskOuter, stage, gatherForwardBodies);
			solveTasks(solve, m_groupTaskOuter, stage, solveForwardGroups);
		}
		// Backward: chains from the top, whose columns hold their own chain's blocks and then
		// those of the chains above, and the subtrees last.
		for(int stage = 0; stage <= SOLVE_STAGES; ++stage)
		{
			solveTasks(solve, m_groupTaskOuter, stage, solveBackwardGroups);
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
		m_updateOuter.resize(bodies + 1);
		m_updateOuter[0] = 0;
		for(int body = 0; body < bodies; ++body)
		{
			const int count = m_blockOuter[body + 1] - m_blockOuter[body] - 1;
			m_updateOuter[body + 1] = m_updateOuter[body] + count * (count + 1) / 2;
		}
		prepareSolveStages(bodies);
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
				convertColumnFloat6(destination.data(), column, source.data() + 6 * column);
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

	// Schedule of a large factor's parallel solves. The elimination tree's chains are its
	// maximal paths of bodies with one child each: the separators of a nested dissection
	// order. A chain's depth counts the chains above it, and chains of one depth are independent.
	// The chains of the first SOLVE_STAGES depths are groups of their own; each subtree below
	// them is one group. A level schedule, one parallel region per set of independent bodies,
	// does not scale here: half of the forward solve is in the rows of the top chain, whose
	// bodies follow one another.
	// A row holds blocks of the body's descendants. Those outside its chain lie below the
	// chain's lowest body and so precede the chain's own, and m_solveRowSplit is the first of
	// the chain's own. Every solve therefore subtracts a row's blocks in the serial order, and
	// gives the serial result.
	void prepareSolveStages(int bodies)
	{
		m_stageGroupOuter.clear();
		if(m_updateOuter[bodies] < MIN_PARALLEL_SOLVE_UPDATES)
		{
			return;
		}
		std::vector<int>& children = m_tags;
		std::vector<int>& groups = m_solveGroups;
		std::vector<int>& stages = m_pattern;
		std::fill(children.begin(), children.end(), 0);
		for(int body = 0; body < bodies; ++body)
		{
			if(m_parent[body] >= 0)
			{
				++children[m_parent[body]];
			}
		}
		groups.resize(bodies);
		m_groupStages.clear();
		// Parents follow their children, so a descending pass sees a body's parent first.
		for(int body = bodies - 1; body >= 0; --body)
		{
			const int parent = m_parent[body];
			if(parent >= 0 && (stages[parent] == SOLVE_STAGES || children[parent] == 1))
			{
				stages[body] = stages[parent];
				groups[body] = groups[parent];
				continue;
			}
			stages[body] = parent >= 0 ? stages[parent] + 1 : 0;
			groups[body] = int(m_groupStages.size());
			m_groupStages.push_back(stages[body]);
		}
		// Number the groups by stage, and list each group's bodies and each chain stage's
		// bodies in ascending order.
		const int groupCount = int(m_groupStages.size());
		m_stageGroupOuter.assign(SOLVE_STAGES + 2, 0);
		m_stageBodyOuter.assign(SOLVE_STAGES + 1, 0);
		for(int group = 0; group < groupCount; ++group)
		{
			++m_stageGroupOuter[m_groupStages[group] + 1];
		}
		for(int stage = 0; stage <= SOLVE_STAGES; ++stage)
		{
			m_stageGroupOuter[stage + 1] += m_stageGroupOuter[stage];
		}
		m_counts.assign(m_stageGroupOuter.begin(), m_stageGroupOuter.end() - 1);
		// From here m_groupStages holds each group's number instead of its stage.
		for(int group = 0; group < groupCount; ++group)
		{
			m_groupStages[group] = m_counts[m_groupStages[group]]++;
		}
		m_groupOuter.assign(groupCount + 1, 0);
		for(int body = 0; body < bodies; ++body)
		{
			groups[body] = m_groupStages[groups[body]];
			++m_groupOuter[groups[body] + 1];
			if(stages[body] < SOLVE_STAGES)
			{
				++m_stageBodyOuter[stages[body] + 1];
			}
		}
		for(int group = 0; group < groupCount; ++group)
		{
			m_groupOuter[group + 1] += m_groupOuter[group];
		}
		for(int stage = 0; stage < SOLVE_STAGES; ++stage)
		{
			m_stageBodyOuter[stage + 1] += m_stageBodyOuter[stage];
		}
		m_groupBodies.resize(bodies);
		m_stageBodies.resize(m_stageBodyOuter[SOLVE_STAGES]);
		m_solveRowSplit.resize(bodies);
		m_counts.assign(m_groupOuter.begin(), m_groupOuter.end() - 1);
		int stageCursors[SOLVE_STAGES];
		for(int stage = 0; stage < SOLVE_STAGES; ++stage)
		{
			stageCursors[stage] = m_stageBodyOuter[stage];
		}
		// Blocks each group solves, forward and backward, and each chain body gathers.
		std::vector<int>& groupWork = m_groupWork;
		groupWork.assign(groupCount, 0);
		m_gatherWork.resize(m_stageBodies.size());
		for(int body = 0; body < bodies; ++body)
		{
			m_groupBodies[m_counts[groups[body]]++] = body;
			int split = m_rowOuter[body];
			if(stages[body] < SOLVE_STAGES)
			{
				m_stageBodies[stageCursors[stages[body]]++] = body;
				while(split < m_rowOuter[body + 1] && groups[m_blockColumns[m_rowEntries[split]]] != groups[body])
				{
					++split;
				}
				for(int entry = split; entry < m_rowOuter[body + 1]; ++entry)
				{
					// An order that breaks the rule above keeps the serial solve.
					if(groups[m_blockColumns[m_rowEntries[entry]]] != groups[body])
					{
						m_stageGroupOuter.clear();
						return;
					}
				}
			}
			m_solveRowSplit[body] = split;
			if(stages[body] < SOLVE_STAGES)
			{
				m_gatherWork[stageCursors[stages[body]] - 1] = split - m_rowOuter[body];
			}
			groupWork[groups[body]] += m_rowOuter[body + 1] - split + m_blockOuter[body + 1] - m_blockOuter[body];
		}
		m_gatherTasks.clear();
		m_groupTasks.clear();
		m_gatherTaskOuter.assign(SOLVE_STAGES + 1, 0);
		m_groupTaskOuter.assign(SOLVE_STAGES + 2, 0);
		for(int stage = 0; stage <= SOLVE_STAGES; ++stage)
		{
			if(stage < SOLVE_STAGES)
			{
				appendTasks(m_gatherTasks, m_gatherWork, m_stageBodyOuter[stage], m_stageBodyOuter[stage + 1]);
				m_gatherTaskOuter[stage + 1] = int(m_gatherTasks.size());
			}
			appendTasks(m_groupTasks, groupWork, m_stageGroupOuter[stage], m_stageGroupOuter[stage + 1]);
			m_groupTaskOuter[stage + 1] = int(m_groupTasks.size());
		}
		m_gatherTasks.push_back(m_stageBodyOuter[SOLVE_STAGES]);
		m_groupTasks.push_back(groupCount);
	}

	// Appends the first items of the tasks that split items [first, end) by their work.
	static void appendTasks(std::vector<int>& tasks, const std::vector<int>& work, int first, int end)
	{
		int total = 0;
		for(int item = first; item < end; ++item)
		{
			total += work[item];
		}
		const int target = std::max(int(SOLVE_TASK_BLOCKS), (total + SOLVE_TASKS - 1) / SOLVE_TASKS);
		int gathered = target;
		for(int item = first; item < end; ++item)
		{
			if(gathered >= target)
			{
				tasks.push_back(item);
				gathered = 0;
			}
			gathered += work[item];
		}
	}

	// Subtracts the products of a row's blocks [first, end) and their solved bodies.
	void gatherForwardBody(double* solution, int body, int first, int end) const
	{
		Vec6 current = loadVector<6>(solution + 6 * body);
		for(int entry = first; entry < end; ++entry)
		{
			const int block = m_rowEntries[entry];
			const Vec6 solved = loadVector<6>(solution + 6 * m_blockColumns[block]);
			subtractFloatMatrixVector6(current.data(), m_blocks[block].data(), solved.data());
		}
		storeVector<6>(solution + 6 * body, current);
	}

	// Solves a body whose row's blocks before first have been subtracted.
	void solveForwardBody(double* solution, int body, int first) const
	{
		Vec6 current = loadVector<6>(solution + 6 * body);
		for(int entry = first; entry < m_rowOuter[body + 1]; ++entry)
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
				current[axis] -= dotFloat6(factor.data(), axis, solved.data());
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
			solveForwardBody(solution, body, m_rowOuter[body]);
		}
		for(int body = bodies - 1; body >= 0; --body)
		{
			solveBackwardBody(solution, body);
		}
	}

	static void gatherForwardBodies(void* context, int index)
	{
		ParallelSolve& solve = *static_cast<ParallelSolve*>(context);
		const BlockCholesky& factor = *solve.factor;
		const int task = solve.first + index;
		for(int entry = factor.m_gatherTasks[task]; entry < factor.m_gatherTasks[task + 1]; ++entry)
		{
			const int body = factor.m_stageBodies[entry];
			factor.gatherForwardBody(solve.solution, body, factor.m_rowOuter[body], factor.m_solveRowSplit[body]);
		}
	}

	// The bodies of the task's groups, which follow one another in ascending order.
	static void solveForwardGroups(void* context, int index)
	{
		ParallelSolve& solve = *static_cast<ParallelSolve*>(context);
		const BlockCholesky& factor = *solve.factor;
		const int task = solve.first + index;
		const int end = factor.m_groupOuter[factor.m_groupTasks[task + 1]];
		for(int entry = factor.m_groupOuter[factor.m_groupTasks[task]]; entry < end; ++entry)
		{
			const int body = factor.m_groupBodies[entry];
			factor.solveForwardBody(solve.solution, body, factor.m_solveRowSplit[body]);
		}
	}

	static void solveBackwardGroups(void* context, int index)
	{
		ParallelSolve& solve = *static_cast<ParallelSolve*>(context);
		const BlockCholesky& factor = *solve.factor;
		const int task = solve.first + index;
		const int first = factor.m_groupOuter[factor.m_groupTasks[task]];
		for(int entry = factor.m_groupOuter[factor.m_groupTasks[task + 1]] - 1; entry >= first; --entry)
		{
			factor.solveBackwardBody(solve.solution, factor.m_groupBodies[entry]);
		}
	}

	// Runs a stage's tasks; a single task needs no helpers.
	void solveTasks(ParallelSolve& solve, const std::vector<int>& taskOuter, int stage, ParallelFunction function)
	{
		solve.first = taskOuter[stage];
		const int tasks = taskOuter[stage + 1] - solve.first;
		if(tasks > 1)
		{
			m_parallelExecutor->parallelFor(tasks, function, &solve);
		}
		else if(tasks == 1)
		{
			function(&solve, 0);
		}
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
				promoteColumnFloat6(values + entry, m_blocks[block].data(), axis);
				entry += 6;
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
			ANVIL_UNROLL
			for(int row = 0; row < column; ++row)
			{
				output(row, column) = 0.0f;
			}
			output(column, column) = pivot;
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
				output(row, column) = float(lower[column][row]);
			}
			output.clearPadding(column);
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
	// Staged solves: groups by stage, bodies by group and by chain stage, and each body's first
	// row entry of its own group. m_stageGroupOuter is empty for factors that solve serially.
	std::vector<int> m_stageGroupOuter, m_groupOuter, m_groupBodies, m_stageBodyOuter, m_stageBodies, m_solveRowSplit;
	std::vector<int> m_solveGroups, m_groupStages;
	// Tasks by stage and the first body or group of each task, with the total count last.
	std::vector<int> m_gatherTaskOuter, m_gatherTasks, m_groupTaskOuter, m_groupTasks, m_gatherWork, m_groupWork;
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
