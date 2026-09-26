#include "NewtonMetis.h"
typedef std::pair<int, int> BodyPair;

// Internal factor implementation. Included by NewtonSolver.cpp.
// Signed rank updates operate directly on the retained CSC factor.
class IncrementalCholesky
{
	enum
	{
		MIN_METIS_BODIES = 192
	};

	struct RankUpdate
	{
		int contact;
		Vec3 axis;
		bool add;
	};

	struct PatchUpdate
	{
		int contact;
		Vec6 vector[2];
		bool add;
	};

public:
	IncrementalCholesky() : m_size(0), m_profile(false), m_updateInverseCurrent(false), m_dense(false) {}
	const Curvature& currentWeights() const { return m_weights; }

	void beginSolve(bool profile, bool continuation, ParallelExecutor* parallelExecutor)
	{
		// Ordinary solves rebuild numerics. Explicit same-prepared-problem
		// continuation compares fresh curvature against this factor's m_weights.
		if(!continuation)
		{
			m_size = 0;
			m_updateInverseCurrent = false;
		}
		m_profile = profile;
		m_factor.setParallelExecutor(parallelExecutor);
	}

	// changedRows optionally lists every diagonal row whose weight differs from
	// m_weights; otherwise the complete diagonal is compared.
	bool factor(const Problem& problem, Curvature& weights, Result& result, const int* changedRows = NULL, int changedCount = 0)
	{
		reserveStorage(m_updates, std::uint32_t(problem.rowCount()));
		if(m_size == 0)
		{
			return refactor(problem, weights, result);
		}
		// A one-body Hessian is a single dense block; refactoring it costs no more than
		// updating it, so the factor is reused only for unchanged curvature.
		if(m_dense)
		{
			if(problem.isUnilateral() && weights.diagonal.size() == m_weights.diagonal.size() &&
				std::equal(weights.diagonal.data(), weights.diagonal.data() + weights.diagonal.size(), m_weights.diagonal.data()))
			{
				++result.reusedFactors;
				m_weights.swap(weights);
				return true;
			}
			return refactor(problem, weights, result);
		}
		// Bilateral curvature is constant within a solve. Its existing factor
		// can be reused without scanning weights or estimating update costs.
		if(problem.equalityRows == problem.rowCount())
		{
			++result.reusedFactors;
			return true;
		}
		const Clock::time_point start = profileStart(m_profile);
		m_updates.clear();
		m_patchUpdates.clear();
		reserveStorage(m_patchUpdates, 4u * std::uint32_t(problem.patches.size()));
		if(changedRows)
		{
			for(int i = 0; i < changedCount; ++i)
			{
				const int row = changedRows[i];
				appendDiagonalUpdate(problem, row, weights.diagonal[row] - m_weights.diagonal[row]);
			}
		}
		else
		{
			const int diagonalCount = weights.diagonal.size();
			for(int row = 0; row < diagonalCount; ++row)
			{
				const double change = weights.diagonal[row] - m_weights.diagonal[row];
				if(change != 0.0)
				{
					appendDiagonalUpdate(problem, row, change);
				}
			}
		}
		const int coupledCount = int(weights.coupled.size());
		for(int block = 0; block < coupledCount; ++block)
		{
			if(weights.coupled[block].equals(m_weights.coupled[block]))
			{
				continue;
			}
			const Mat3 change = weights.coupled[block] - m_weights.coupled[block];
			Vec3 eigenvalues;
			Mat3 eigenvectors;
			symmetricEigen(change, eigenvalues, eigenvectors);
			const double largest = eigenvalues.cwiseAbs().maxCoeff();
			for(int axis = 0; axis < 3; ++axis)
			{
				const double value = eigenvalues[axis];
				// Discard only numerical zero relative to this Hessian change.
				if(std::abs(value) <= 1.0e-12 * largest)
				{
					continue;
				}
				m_updates.emplace_back();
				RankUpdate& update = m_updates.back();
				update.contact = problem.coupledContacts[block];
				for(int row = 0; row < 3; ++row)
				{
					update.axis[row] = std::sqrt(std::abs(value)) * eigenvectors(row, axis);
				}
				update.add = value > 0.0;
			}
		}

		const std::uint32_t patchCount = std::uint32_t(weights.patches.size());
		for(std::uint32_t i = 0; i < patchCount; ++i)
		{
			const PatchCurvature& current = weights.patches[i];
			const PatchCurvature& previous = m_weights.patches[i];
			bool changed = current.normalCoefficient != previous.normalCoefficient || current.crossCoefficient != previous.crossCoefficient || current.tangentCoefficient != previous.tangentCoefficient;
			for(int end = 0; end < 2; ++end)
			{
				changed = changed || !current.normal[end].equals(previous.normal[end]) || !current.tangent[end].equals(previous.tangent[end]);
			}
			if(changed && (!appendPatchUpdates(current, problem.patches[i].firstContact, 1.0) || !appendPatchUpdates(previous, problem.patches[i].firstContact, -1.0)))
			{
				result.updateMs += profileElapsed(m_profile, start);
				++result.factorFallbacks;
				return refactor(problem, weights, result);
			}
		}

		// Compare structural arithmetic costs before applying any update.
		// This changes only how the same target Hessian is factorized. The
		// estimate is deterministic and depends on the sparse pattern and
		// current changed contacts, never wall time or a scene identifier.
		// Matrix assembly remains serial; only discount the numeric factor work.
		double refactorWork = m_factorWork + problem.rebuildWorkEstimate();
		const int parallelWorkers = m_factor.parallelWorkerCount();
		if(parallelWorkers > 1)
		{
			refactorWork -= m_factorWork * double(parallelWorkers - 1) / parallelWorkers;
		}
		double updateWork = 0.0;
		for(const RankUpdate& update : m_updates)
		{
			const CompactContact& contact = problem.contacts[update.contact];
			int first = m_size;
			for(int end = 0; end < 2; ++end)
			{
				if(contact.body[end] >= 0)
				{
					first = std::min(first, m_permutation[6 * contact.body[end]]);
				}
			}
			if(first < m_size)
			{
				updateWork += m_reachWork[first];
			}
			if(updateWork > refactorWork)
			{
				result.updateMs += profileElapsed(m_profile, start);
				return refactor(problem, weights, result);
			}
		}
		const std::uint32_t patchUpdateCount = std::uint32_t(m_patchUpdates.size());
		for(std::uint32_t i = 0; i < patchUpdateCount; ++i)
		{
			const CompactContact& contact = problem.contacts[m_patchUpdates[i].contact];
			int first = m_size;
			for(int end = 0; end < 2; ++end)
			{
				if(contact.body[end] >= 0)
				{
					first = std::min(first, m_permutation[6 * contact.body[end]]);
				}
			}
			if(first < m_size)
			{
				updateWork += m_reachWork[first];
			}
			if(updateWork > refactorWork)
			{
				result.updateMs += profileElapsed(m_profile, start);
				return refactor(problem, weights, result);
			}
		}
		if(m_updates.empty() && m_patchUpdates.empty())
		{
			++result.reusedFactors;
		}
		else
		{
			const bool blockInverseCurrent = m_factor.hasCurrentBlocks();
			m_factor.beginScalarUpdates();
			if(!m_updateInverseCurrent)
			{
				if(blockInverseCurrent)
				{
					m_updateInverseDiagonal.resize(m_size);
					m_factor.swapInverseDiagonal(m_updateInverseDiagonal);
				}
				else
				{
					const Sparse& lower = m_factor.m_matrix;
					const int* outer = lower.outerIndexPtr();
					const double* values = lower.valuePtr();
					m_updateInverseDiagonal.resize(m_size);
					for(int column = 0; column < m_size; ++column)
					{
						m_updateInverseDiagonal[column] = 1.0 / values[outer[column]];
					}
				}
				m_updateInverseCurrent = true;
			}
		}
		// Add positive changes before downdates. Every intermediate matrix
		// then remains positive definite whenever the target Hessian is SPD.
		for(int pass = 0; pass < 2; ++pass)
		{
			const std::uint32_t updateCount = std::uint32_t(m_updates.size());
			for(std::uint32_t i = 0; i < updateCount; ++i)
			{
				const RankUpdate& update = m_updates[i];
				if(update.add != (pass == 0))
				{
					continue;
				}
				const CompactContact& contact = problem.contacts[update.contact];
				const bool fullRank = updateSparse(problem, contact, update.axis, update.add);
				++result.rankUpdates;
				if(!fullRank)
				{
					result.updateMs += profileElapsed(m_profile, start);
					++result.factorFallbacks;
					return refactor(problem, weights, result);
				}
			}
			for(std::uint32_t i = 0; i < patchUpdateCount; ++i)
			{
				const PatchUpdate& update = m_patchUpdates[i];
				if(update.add != (pass == 0))
				{
					continue;
				}
				const bool fullRank = updatePair(problem.contacts[update.contact].body, update.vector, update.add);
				++result.rankUpdates;
				if(!fullRank)
				{
					result.updateMs += profileElapsed(m_profile, start);
					++result.factorFallbacks;
					return refactor(problem, weights, result);
				}
			}
		}
		m_weights.swap(weights);
		result.updateMs += profileElapsed(m_profile, start);
		return true;
	}

	// Numerically factor the Hessian of these weights without comparing them to the
	// retained factor. Interior-point steps change every weight.
	bool factorFresh(const Problem& problem, Curvature& weights, Result& result)
	{
		reserveStorage(m_updates, std::uint32_t(problem.rowCount()));
		return refactor(problem, weights, result);
	}

	void solveDirection(ConstVector gradient, MutableVector direction)
	{
		if(m_dense)
		{
			// Solve L L^T direction = -gradient with the dense one-body factor.
			double value[6];
			for(int row = 0; row < 6; ++row)
			{
				double sum = -gradient[row];
				for(int inner = 0; inner < row; ++inner)
				{
					sum -= m_denseLower[row * 6 + inner] * value[inner];
				}
				value[row] = sum * m_denseInverse[row];
			}
			for(int row = 5; row >= 0; --row)
			{
				double sum = value[row];
				for(int inner = row + 1; inner < 6; ++inner)
				{
					sum -= m_denseLower[inner * 6 + row] * value[inner];
				}
				direction[row] = value[row] = sum * m_denseInverse[row];
			}
			return;
		}
		// Solve H * direction = -gradient directly into the caller's buffer.
		std::vector<double>& solution = m_solution;
		for(int row = 0; row < m_size; ++row)
		{
			solution[m_permutation[row]] = -gradient[row];
		}
		if(m_factor.hasCurrentBlocks())
		{
			m_factor.solveBlocks(solution.data());
		}
		else
		{
			solveForwardBlocks(solution.data());
			solveBackwardPackets(solution.data());
		}
		for(int row = 0; row < m_size; ++row)
		{
			direction[row] = solution[m_permutation[row]];
		}
	}

private:

	bool appendPatchUpdates(const PatchCurvature& patch, int contact, double sign)
	{
		Matrix<2, 2> coefficients;
		coefficients(0, 0) = patch.normalCoefficient;
		coefficients(1, 0) = patch.crossCoefficient;
		coefficients(0, 1) = patch.crossCoefficient;
		coefficients(1, 1) = patch.tangentCoefficient;
		Vec2 eigenvalues;
		Matrix<2, 2> eigenvectors;
		symmetricEigen(coefficients, eigenvalues, eigenvectors);
		for(int axis = 0; axis < 2; ++axis)
		{
			const double value = sign * eigenvalues[axis];
			if(value == 0.0)
			{
				continue;
			}
			m_patchUpdates.emplace_back();
			PatchUpdate& update = m_patchUpdates.back();
			update.contact = contact;
			update.add = value > 0.0;
			const double scale = std::sqrt(std::abs(value));
			for(int end = 0; end < 2; ++end)
			{
				update.vector[end] = scale * (eigenvectors(0, axis) * patch.normal[end] +
											  eigenvectors(1, axis) * patch.tangent[end]);
			}
		}
		return true;
	}

	void solveForwardBlocks(double* solution) const
	{
		const Sparse& lower = m_factor.m_matrix;
		const int* outer = lower.outerIndexPtr();
		const int* inner = lower.innerIndexPtr();
		const double* values = lower.valuePtr();
		// The body ordering and full endpoint cliques preserve complete six-row
		// off-diagonal blocks in all six columns, including after rank updates.
		// Accumulate one destination body in registers across its six source
		// columns. Each scalar still receives subtractions in ascending-column
		// order, matching the established forward substitution arithmetic.
		for(int first = 0; first < m_size; first += 6)
		{
			Vec6 current = loadVector<6>(solution + first);
			for(int axis = 0; axis < 6; ++axis)
			{
				if(current[axis] == 0.0)
				{
					continue;
				}
				const int begin = outer[first + axis];
				current[axis] /= values[begin];
				for(int row = axis + 1; row < 6; ++row)
				{
					current[row] -= current[axis] * values[begin + row - axis];
				}
			}
			storeVector<6>(solution + first, current);
			const int offBegin = outer[first] + 6;
			const int offEnd = outer[first + 1];
			int offColumn[6];
			for(int axis = 0; axis < 6; ++axis)
			{
				offColumn[axis] = outer[first + axis] + 6 - axis - offBegin;
			}
			for(int entry = offBegin; entry < offEnd; entry += 6)
			{
				double* destination = solution + inner[entry];
#if defined(NEWTON_AVX2_FMA)
				__m256d firstFour = _mm256_loadu_pd(destination);
				__m128d lastTwo = _mm_loadu_pd(destination + 4);
				for(int axis = 0; axis < 6; ++axis)
				{
					if(current[axis] != 0.0)
					{
						const double* column = values + entry + offColumn[axis];
						const __m256d scale = _mm256_set1_pd(current[axis]);
						firstFour = _mm256_fnmadd_pd(scale, _mm256_loadu_pd(column), firstFour);
						lastTwo = _mm_fnmadd_pd(_mm256_castpd256_pd128(scale), _mm_loadu_pd(column + 4), lastTwo);
					}
				}
				_mm256_storeu_pd(destination, firstFour);
				_mm_storeu_pd(destination + 4, lastTwo);
#elif defined(NEWTON_SIMD128)
				simd::Double2 low = simd::load(destination), middle = simd::load(destination + 2), high = simd::load(destination + 4);
				for(int axis = 0; axis < 6; ++axis)
				{
					if(current[axis] != 0.0)
					{
						const double* column = values + entry + offColumn[axis];
						const simd::Double2 scale = simd::splat(current[axis]);
						low = simd::negativeMultiplyAdd(scale, simd::load(column), low);
						middle = simd::negativeMultiplyAdd(scale, simd::load(column + 2), middle);
						high = simd::negativeMultiplyAdd(scale, simd::load(column + 4), high);
					}
				}
				simd::store(destination, low);
				simd::store(destination + 2, middle);
				simd::store(destination + 4, high);
#else
				for(int axis = 0; axis < 6; ++axis)
				{
					if(current[axis] != 0.0)
					{
						const int address = entry + offColumn[axis];
						for(int row = 0; row < 6; ++row)
						{
							destination[row] -= current[axis] * values[address + row];
						}
					}
				}
#endif
			}
		}
	}

	void solveBackwardPackets(double* solution) const
	{
		const Sparse& lower = m_factor.m_matrix;
		const int* outer = lower.outerIndexPtr();
		const int* inner = lower.innerIndexPtr();
		const double* values = lower.valuePtr();
		for(int column = m_size - 1; column >= 0; --column)
		{
			const int begin = outer[column], end = outer[column + 1];
			double current = solution[column];
			int entry = begin + 1;
			const int offBegin = begin + 6 - column % 6;
			for(; entry < offBegin; ++entry)
			{
				current -= values[entry] * solution[inner[entry]];
			}
			for(; entry < end; entry += 6)
			{
				// Evaluate independent products in packets, then retain their
				// left-to-right subtraction order.
				current = subtractDot6(current, values + entry, solution + inner[entry]);
			}
			solution[column] = current / values[begin];
		}
	}

	void appendDiagonalUpdate(const Problem& problem, int row, double change)
	{
		m_updates.emplace_back();
		RankUpdate& update = m_updates.back();
		update.contact = problem.rowContact[row];
		const CompactContact& contact = problem.contacts[update.contact];
		const int axis = contact.rowCount() == 1 ? 2 : row - contact.row;
		update.axis = Vec3::Unit(axis) * std::sqrt(std::abs(change));
		update.add = change > 0.0;
	}

	// Exports the scalar CSC Hessian on first use within one refactorization.
	const Sparse& exportedHessian(const Problem& problem, const Sparse*& matrix)
	{
		if(!matrix)
		{
			matrix = &exportHessian(problem, m_hessian);
		}
		return *matrix;
	}

	bool refactor(const Problem& problem, Curvature& weights, Result& result)
	{
		const bool firstFactor = m_size == 0;
		bool changedPattern = false;
		const Clock::time_point assemblyStart = profileStart(m_profile);
		assembleHessianBlocks(problem, weights, m_hessian);
		// The scalar CSC Hessian is exported only for symbolic analysis and for
		// factor paths that do not read the body-pair blocks directly.
		const Sparse* matrix = NULL;
		const double assemblyMs = profileElapsed(m_profile, assemblyStart);
		result.matrixMs += assemblyMs;
		const Clock::time_point factorStart = profileStart(m_profile);
		// One body's Hessian is its lower 6x6 diagonal block. A failed dense pivot
		// falls through to the block factor and its scalar fallback.
		Mat6 denseLower;
		m_dense = problem.bodyCount() == 1 && cholesky6(m_hessian.blocks[problem.hessianDiagonalBlocks[0]], denseLower, m_denseInverse);
		if(m_dense)
		{
			for(int row = 0; row < 6; ++row)
			{
				for(int column = 0; column < row; ++column)
				{
					m_denseLower[row * 6 + column] = denseLower(row, column);
				}
			}
			m_size = 6;
			m_updateInverseCurrent = false;
			++result.factorizations;
			result.factorMs += profileElapsed(m_profile, factorStart);
			if(m_weights.diagonal.size() != weights.diagonal.size() || m_weights.coupled.size() != weights.coupled.size() || m_weights.patches.size() != weights.patches.size())
			{
				m_weights.resize(problem);
			}
			m_weights.swap(weights);
			return true;
		}
		if(m_size == 0)
		{
			const Clock::time_point symbolicStart = profileStart(m_profile);
			// Sorted body pairs determine the complete sparse pattern.
			if(m_pairs != problem.hessianPairs)
			{
				changedPattern = true;
				const Sparse& pattern = exportedHessian(problem, matrix);
				orderBodies(pattern);
				m_pairs = problem.hessianPairs;
				preparePermutation(pattern);
				m_factor.analyzePattern(m_permuted);
				m_factor.prepareBlockInput(m_pairs, m_permutation);
				++result.symbolicAnalyses;
			}
			result.symbolicMs += profileElapsed(m_profile, symbolicStart);
		}
		m_size = problem.bodyCount() * 6;
		m_vector.resize(m_size);
		m_solution.resize(m_size);
		if(!m_factor.readsBlocks() || !m_factor.factorizeBlocks(m_hessian.blocks.data()))
		{
			const Sparse& values = exportedHessian(problem, matrix);
			const int nonzeroCount = values.nonZeros();
			for(int entry = 0; entry < nonzeroCount; ++entry)
			{
				m_permuted.valuePtr()[entry] = values.valuePtr()[m_permutedSourceIndices[entry]];
			}
			if(!m_factor.factorize(m_permuted))
			{
				return false;
			}
		}
		m_updateInverseCurrent = false;
		if(firstFactor)
		{
			// Numeric-factor work is proportional to squared column lengths;
			// a rank update touches the elimination-tree reach once per column.
			// Cache those reach costs so selection is linear in changed contacts.
			if(changedPattern)
			{
				const Sparse& lower = m_factor.m_matrix;
				m_reachWork.resize(m_size);
				m_factorWork = double(exportedHessian(problem, matrix).nonZeros());
				for(int column = m_size - 1; column >= 0; --column)
				{
					const int begin = lower.outerIndexPtr()[column], end = lower.outerIndexPtr()[column + 1];
					const int count = end - begin - 1;
					const int parent = count ? lower.innerIndexPtr()[begin + 1] : m_size;
					m_reachWork[column] = 6.0 * count + 12.0 + (parent < m_size ? m_reachWork[parent] : 0.0);
					m_factorWork += double(count) * (count + 3) + 12.0;
				}
			}
		}
		++result.factorizations;
		result.factorMs += profileElapsed(m_profile, factorStart);
		if(m_weights.diagonal.size() != weights.diagonal.size() || m_weights.coupled.size() != weights.coupled.size() || m_weights.patches.size() != weights.patches.size())
		{
			m_weights.resize(problem);
		}
		m_weights.swap(weights);
		return true;
	}

	void orderBodies(const Sparse& matrix)
	{
		const int bodies = int(matrix.cols()) / 6;
		m_bodyEdges.clear();
		for(int body = 0; body < bodies; ++body)
		{
			int previous = -1;
			for(Sparse::InnerIterator entry(matrix, 6 * body); entry; ++entry)
			{
				const int row = int(entry.row()) / 6;
				if(row != body && row != previous)
				{
					m_bodyEdges.emplace_back(body, row);
					m_bodyEdges.emplace_back(row, body);
				}
				previous = row;
			}
		}
		std::sort(m_bodyEdges.begin(), m_bodyEdges.end());
		m_bodyOrder.resize(bodies);
		// Retained exact minimum-degree storage avoids METIS setup and heap traffic
		// on small changing islands. METIS pays for itself on larger graphs.
		if(bodies < MIN_METIS_BODIES)
		{
			orderSmallGraph(bodies);
		}
		else
		{
			m_metisOuter.resize(bodies + 1);
			const std::uint32_t edgeCount = std::uint32_t(m_bodyEdges.size());
			m_metisInner.resize(edgeCount);
			int edge = 0;
			for(int body = 0; body < bodies; ++body)
			{
				m_metisOuter[body] = idx_t(edge);
				while(edge < int(edgeCount) && m_bodyEdges[edge].first == body)
				{
					m_metisInner[edge] = idx_t(m_bodyEdges[edge].second);
					++edge;
				}
			}
			m_metisOuter[bodies] = idx_t(edge);
			m_metisInverse.resize(bodies);
			idx_t count = idx_t(bodies);
			idx_t options[METIS_NOPTIONS];
			Newton_METIS_SetDefaultOptions(options);
			const int result = Newton_METIS_NodeND(&count, m_metisOuter.data(), m_metisInner.data(), NULL, options, m_bodyOrder.data(), m_metisInverse.data());
			if(result != METIS_OK)
			{
				for(int body = 0; body < bodies; ++body)
				{
					m_bodyOrder[body] = body;
				}
			}
		}
		m_permutation.resize(matrix.cols());
		for(int body = 0; body < bodies; ++body)
		{
			for(int axis = 0; axis < 6; ++axis)
			{
				m_permutation[6 * m_bodyOrder[body] + axis] = 6 * body + axis;
			}
		}
	}

	void orderSmallGraph(int bodies)
	{
		const int words = (bodies + 63) / 64;
		m_smallAdjacency.assign(std::uint32_t(bodies) * std::uint32_t(words), 0);
		m_smallEliminated.assign(bodies, 0);
		const std::uint32_t edgeCount = std::uint32_t(m_bodyEdges.size());
		for(std::uint32_t edge = 0; edge < edgeCount; ++edge)
		{
			const BodyPair& pair = m_bodyEdges[edge];
			m_smallAdjacency[std::uint32_t(pair.first) * std::uint32_t(words) + std::uint32_t(pair.second / 64)] |=
				std::uint64_t(1) << (pair.second % 64);
		}
		for(int position = 0; position < bodies; ++position)
		{
			int selected = -1, minimumDegree = bodies;
			for(int body = 0; body < bodies; ++body)
			{
				if(m_smallEliminated[body])
				{
					continue;
				}
				int degree = 0;
				for(int word = 0; word < words; ++word)
				{
					std::uint64_t bits = m_smallAdjacency[std::uint32_t(body) * std::uint32_t(words) + std::uint32_t(word)];
					while(bits)
					{
						bits &= bits - 1;
						++degree;
					}
				}
				if(degree < minimumDegree)
				{
					selected = body;
					minimumDegree = degree;
				}
			}
			m_bodyOrder[position] = selected;
			m_smallEliminated[selected] = 1;
			for(int body = 0; body < bodies; ++body)
			{
				if(m_smallEliminated[body] || !(m_smallAdjacency[std::uint32_t(selected) * std::uint32_t(words) + std::uint32_t(body / 64)] & (std::uint64_t(1) << (body % 64))))
				{
					continue;
				}
				std::uint64_t* adjacency = m_smallAdjacency.data() + std::uint32_t(body) * std::uint32_t(words);
				const std::uint64_t* fill = m_smallAdjacency.data() + std::uint32_t(selected) * std::uint32_t(words);
				for(int word = 0; word < words; ++word)
				{
					adjacency[word] |= fill[word];
				}
				adjacency[selected / 64] &= ~(std::uint64_t(1) << (selected % 64));
				adjacency[body / 64] &= ~(std::uint64_t(1) << (body % 64));
			}
		}
	}

	void preparePermutation(const Sparse& matrix)
	{
		// Bucket the permuted upper triangle by row, then scatter rows in order
		// into CSC columns. Both passes are linear and produce sorted row indices
		// without sorting every scalar entry of each complete body block.
		const int size = int(matrix.cols());
		m_permutationRowOffsets.resize(size + 1);
		m_permutationCursors.assign(size, 0);
		const int* matrixOuter = matrix.outerIndexPtr();
		const int* matrixInner = matrix.innerIndexPtr();
		const int nonzeroCount = matrix.nonZeros();
		reserveStorage(m_permutedEntries, std::uint32_t(nonzeroCount));
		m_permutedEntries.resize(nonzeroCount);
		reserveStorage(m_permutedSourceIndices, std::uint32_t(nonzeroCount));
		m_permutedSourceIndices.resize(nonzeroCount);
		for(int column = 0; column < size; ++column)
		{
			const int end = matrixOuter[column + 1];
			for(int entry = matrixOuter[column]; entry < end; ++entry)
			{
				++m_permutationCursors[std::min(m_permutation[column], m_permutation[matrixInner[entry]])];
			}
		}
		int offset = 0;
		for(int row = 0; row < size; ++row)
		{
			m_permutationRowOffsets[row] = offset;
			const int count = m_permutationCursors[row];
			m_permutationCursors[row] = offset;
			offset += count;
		}
		m_permutationRowOffsets[size] = offset;
		for(int column = 0; column < size; ++column)
		{
			const int end = matrixOuter[column + 1];
			for(int entry = matrixOuter[column]; entry < end; ++entry)
			{
				const int a = m_permutation[column];
				const int b = m_permutation[matrixInner[entry]];
				PermutedEntry& mapped = m_permutedEntries[m_permutationCursors[std::min(a, b)]++];
				mapped.column = std::max(a, b);
				mapped.sourceIndex = entry;
			}
		}
		// Reuse the scatter cursors for columns after the row buckets are filled.
		std::fill(m_permutationCursors.begin(), m_permutationCursors.end(), 0);
		const std::uint32_t entryCount = std::uint32_t(m_permutedEntries.size());
		for(std::uint32_t entry = 0; entry < entryCount; ++entry)
		{
			++m_permutationCursors[m_permutedEntries[entry].column];
		}
		m_permuted.resize(size, size);
		m_permuted.resizeNonZeros(matrix.nonZeros());
		offset = 0;
		for(int column = 0; column < size; ++column)
		{
			m_permuted.outerIndexPtr()[column] = offset;
			const int count = m_permutationCursors[column];
			m_permutationCursors[column] = offset;
			offset += count;
		}
		m_permuted.outerIndexPtr()[size] = offset;
		for(int row = 0; row < size; ++row)
		{
			for(int entry = m_permutationRowOffsets[row]; entry < m_permutationRowOffsets[row + 1]; ++entry)
			{
				const PermutedEntry& mapped = m_permutedEntries[entry];
				const int destination = m_permutationCursors[mapped.column]++;
				m_permuted.innerIndexPtr()[destination] = row;
				m_permutedSourceIndices[destination] = mapped.sourceIndex;
			}
		}
	}

	bool updateSparse(const Problem& problem, const CompactContact& contact, const Vec3& axis, bool add)
	{
		Vec6 vector[2];
		for(int end = 0; end < 2; ++end)
		{
			if(contact.body[end] >= 0)
			{
				if(contact.rowCount() == 3)
				{
					vector[end] = problem.contactJacobian(contact, end).transpose() * axis;
				}
				else
				{
					vector[end] = contact.jacobian[end] * axis[2];
				}
			}
		}
		return updatePair(contact.body, vector, add);
	}

	template<bool Add>
	bool updatePairSigned(const int* body, const Vec6* vector)
	{
		// Preserve the established signed-update kernel's pivot floor.
		static constexpr double minimumPivotSquare = 1.0e-15;
		// The allocated Hessian contains the complete clique of each contact's
		// endpoint coordinates, including inactive rows. Consequently its rank
		// update reaches a single ancestor path of the Cholesky elimination tree.
		// The first off-diagonal row in a factor column is its parent. A retained
		// dense work vector avoids rebuilding/merging sparse index lists at every
		// pivot, while the factor's sparse columns bound all numerical work.
		// A successful update consumes the single elimination-tree path below and
		// clears every visited entry, so the retained vector is already zero here.
		// Avoid clearing the entire factor-sized vector for every contact update.
		int first = m_size;
		for(int end = 0; end < 2; ++end)
		{
			if(body[end] >= 0)
			{
				const Vec6& row = vector[end];
				const int base = m_permutation[6 * body[end]];
				for(int component = 0; component < 6; ++component)
				{
					if(row[component] != 0.0)
					{
						const int column = base + component;
						m_vector[column] = row[component];
						first = std::min(first, column);
					}
				}
			}
		}
		Sparse& lower = m_factor.m_matrix;
		const int* outer = lower.outerIndexPtr();
		const int* inner = lower.innerIndexPtr();
		double* values = lower.valuePtr();
		for(int column = first; column < m_size;)
		{
			const int begin = outer[column], end = outer[column + 1];
			const int parent = end > begin + 1 ? inner[begin + 1] : m_size;
			const double x = m_vector[column];
			if(x != 0.0)
			{
				const double diagonal = values[begin];
				const double square = diagonal * diagonal + (Add ? x * x : -x * x);
				// A nonpositive or undersized pivot requires a full refactorization.
				if(!(square >= minimumPivotSquare))
				{
					// A failed downdate leaves values on the unconsumed suffix of the
					// path. Restore the zero invariant before the full-factor fallback.
					std::fill(m_vector.begin(), m_vector.end(), 0.0);
					return false;
				}
				const double r = std::sqrt(square);
				const double inverseDiagonal = m_updateInverseDiagonal[column];
				const double inverseR = 1.0 / r;
				const double c = r * inverseDiagonal;
				const double s = x * inverseDiagonal;
				const double inverseC = diagonal * inverseR;
				const double signedSC = (Add ? x : -x) * inverseR;
				values[begin] = r;
				m_updateInverseDiagonal[column] = inverseR;
				// Full body blocks make six consecutive factor entries address six
				// consecutive work values. Expose those packets without gathers.
				const int remainder = 5 - column % 6;
				int entry = begin + 1;
				for(int stop = entry + remainder; entry < stop; ++entry)
				{
					const int row = inner[entry];
					const double next = inverseC * values[entry] + signedSC * m_vector[row];
					values[entry] = next;
					m_vector[row] = c * m_vector[row] - s * next;
				}
				for(; entry < end; entry += 6)
				{
					double* factor = values + entry;
					double* work = m_vector.data() + inner[entry];
					updateCholesky6(factor, work, inverseC, signedSC, c, s);
				}
				m_vector[column] = 0.0;
			}
			column = parent;
		}
		return true;
	}

	bool updatePair(const int* body, const Vec6* vector, bool add)
	{
		return add ? updatePairSigned<true>(body, vector) : updatePairSigned<false>(body, vector);
	}

	int m_size;
	bool m_profile;
	BlockCholesky m_factor;
	std::vector<BodyPair> m_bodyEdges;
	std::vector<int> m_bodyOrder, m_permutation;
	std::vector<idx_t> m_metisOuter, m_metisInner, m_metisInverse;
	std::vector<uint64_t> m_smallAdjacency;
	std::vector<unsigned char> m_smallEliminated;
	HessianStorage m_hessian;
	SparseStorage m_permuted;
	struct PermutedEntry
	{
		int column;
		int sourceIndex;
	};
	std::vector<PermutedEntry> m_permutedEntries;
	std::vector<int> m_permutedSourceIndices, m_permutationRowOffsets, m_permutationCursors;
	std::vector<std::uint64_t> m_pairs;
	std::vector<double> m_vector, m_solution, m_reachWork, m_updateInverseDiagonal;
	double m_factorWork = 0.0;
	// One-body problems keep a dense factor instead of the block factor: its strictly
	// lower entries, row-major, and reciprocal pivots. Plain arrays keep the class free
	// of over-aligned members, which C++14 heap allocation does not honour.
	bool m_dense;
	double m_denseLower[36];
	double m_denseInverse[6];
	bool m_updateInverseCurrent;
	Curvature m_weights;
	std::vector<RankUpdate> m_updates;
	std::vector<PatchUpdate> m_patchUpdates;
};
