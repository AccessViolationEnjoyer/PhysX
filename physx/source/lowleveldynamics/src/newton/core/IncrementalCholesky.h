// Internal factor implementation. Included by NewtonSolver.cpp.
// Signed rank updates operate directly on the retained Eigen-format factor.
class IncrementalCholesky
{
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
	IncrementalCholesky() : m_size(0), m_profile(false) {}

	void beginSolve(bool profile, bool continuation, ParallelExecutor* parallelExecutor)
	{
		// Ordinary solves rebuild numerics. Explicit same-prepared-problem
		// continuation compares fresh curvature against this factor's m_weights.
		if(!continuation)
			m_size = 0;
		m_profile = profile;
		m_factor.setParallelExecutor(parallelExecutor);
	}

	bool factor(const Problem& problem, const Curvature& weights, Result& result)
	{
		reserveStorage(m_updates, size_t(problem.rowCount()));
		if(m_size == 0)
		{
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
		reserveStorage(m_patchUpdates, 4 * problem.patches.size());
		for(int row = 0; row < weights.diagonal.size(); ++row)
		{
			const double change = weights.diagonal[row] - m_weights.diagonal[row];
			if(change == 0.0)
				continue;
			RankUpdate update;
			update.contact = problem.rowContact[row];
			const CompactContact& contact = problem.contacts[update.contact];
			const int axis = contact.rowCount() == 1 ? 2 : row - contact.row;
			update.axis = Vec3::Unit(axis) * std::sqrt(std::abs(change));
			update.add = change > 0.0;
			m_updates.push_back(update);
		}
		for(int block = 0; block < int(weights.coupled.size()); ++block)
		{
			if((weights.coupled[block].array() == m_weights.coupled[block].array()).all())
				continue;
			const Mat3 change = weights.coupled[block] - m_weights.coupled[block];
			Eigen::SelfAdjointEigenSolver<Mat3> eigen(change);
			const double largest = eigen.eigenvalues().cwiseAbs().maxCoeff();
			for(int axis = 0; axis < 3; ++axis)
			{
				const double value = eigen.eigenvalues()[axis];
				// Discard only numerical zero relative to this Hessian change.
				if(std::abs(value) <= 1.0e-12 * largest)
					continue;
				RankUpdate update;
				update.contact = problem.coupledContacts[block];
				update.axis = std::sqrt(std::abs(value)) * eigen.eigenvectors().col(axis);
				update.add = value > 0.0;
				m_updates.push_back(update);
			}
		}

		for(size_t i = 0; i < weights.patches.size(); ++i)
		{
			const PatchCurvature& current = weights.patches[i];
			const PatchCurvature& previous = m_weights.patches[i];
			bool changed = current.normalCoefficient != previous.normalCoefficient ||
				current.crossCoefficient != previous.crossCoefficient || current.tangentCoefficient != previous.tangentCoefficient;
			for(int end = 0; end < 2; ++end)
				changed = changed || (current.normal[end].array() != previous.normal[end].array()).any() ||
					(current.tangent[end].array() != previous.tangent[end].array()).any();
			if(changed && (!appendPatchUpdates(current, problem.patches[i].firstContact, 1.0) ||
				!appendPatchUpdates(previous, problem.patches[i].firstContact, -1.0)))
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
		double refactorWork = m_refactorWork;
		const int parallelWorkers = m_factor.parallelWorkerCount();
		if(parallelWorkers > 1)
			refactorWork -= m_factorWork * double(parallelWorkers - 1) / parallelWorkers;
		double updateWork = 0.0;
		for(const RankUpdate& update : m_updates)
		{
			const CompactContact& contact = problem.contacts[update.contact];
			int first = m_size;
			for(int end = 0; end < 2; ++end)
				if(contact.body[end] >= 0)
					first = std::min(first, m_permutation[6 * contact.body[end]]);
			if(first < m_size)
				updateWork += m_reachWork[first];
			if(updateWork > refactorWork)
			{
				result.updateMs += profileElapsed(m_profile, start);
				return refactor(problem, weights, result);
			}
		}
		for(size_t i = 0; i < m_patchUpdates.size(); ++i)
		{
			const CompactContact& contact = problem.contacts[m_patchUpdates[i].contact];
			int first = m_size;
			for(int end = 0; end < 2; ++end)
				if(contact.body[end] >= 0)
					first = std::min(first, m_permutation[6 * contact.body[end]]);
			if(first < m_size)
				updateWork += m_reachWork[first];
			if(updateWork > refactorWork)
			{
				result.updateMs += profileElapsed(m_profile, start);
				return refactor(problem, weights, result);
			}
		}
		if(m_updates.empty() && m_patchUpdates.empty())
			++result.reusedFactors;
		// Add positive changes before downdates. Every intermediate matrix
		// then remains positive definite whenever the target Hessian is SPD.
		for(int pass = 0; pass < 2; ++pass)
		{
			for(size_t i = 0; i < m_updates.size(); ++i)
			{
				const RankUpdate& update = m_updates[i];
				if(update.add != (pass == 0))
					continue;
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
			for(size_t i = 0; i < m_patchUpdates.size(); ++i)
			{
				const PatchUpdate& update = m_patchUpdates[i];
				if(update.add != (pass == 0))
					continue;
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
		m_weights = weights;
		result.updateMs += profileElapsed(m_profile, start);
		return true;
	}

	void solveDirection(ConstVector gradient, MutableVector direction)
	{
		// Solve H * direction = -gradient directly into the caller's buffer.
		m_solution.resize(m_size);
		MutableVector solution = Eigen::Map<Vector>(m_solution.data(), m_size);
		for(int row = 0; row < m_size; ++row)
			solution[m_permutation[row]] = -gradient[row];
		solveForwardBlocks(solution.data());
		solveBackwardPackets(solution.data());
		for(int row = 0; row < m_size; ++row)
			direction[row] = solution[m_permutation[row]];
	}

private:

	bool appendPatchUpdates(const PatchCurvature& patch, int contact, double sign)
	{
		Eigen::Matrix2d coefficients;
		coefficients << patch.normalCoefficient, patch.crossCoefficient,
			patch.crossCoefficient, patch.tangentCoefficient;
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(coefficients);
		if(eigen.info() != Eigen::Success || !eigen.eigenvalues().allFinite())
			return false;
		for(int axis = 0; axis < 2; ++axis)
		{
			const double value = sign * eigen.eigenvalues()[axis];
			if(value == 0.0)
				continue;
			PatchUpdate update;
			update.contact = contact;
			update.add = value > 0.0;
			const double scale = std::sqrt(std::abs(value));
			for(int end = 0; end < 2; ++end)
				update.vector[end] = scale * (eigen.eigenvectors()(0, axis) * patch.normal[end] +
					eigen.eigenvectors()(1, axis) * patch.tangent[end]);
			m_patchUpdates.push_back(update);
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
			Vec6 current = Eigen::Map<const Vec6>(solution + first);
			for(int axis = 0; axis < 6; ++axis)
			{
				if(current[axis] == 0.0)
					continue;
				const int begin = outer[first + axis];
				current[axis] /= values[begin];
				for(int row = axis + 1; row < 6; ++row)
					current[row] -= current[axis] * values[begin + row - axis];
			}
			Eigen::Map<Vec6>(solution + first) = current;
			const int offBegin = outer[first] + 6;
			const int offEnd = outer[first + 1];
			for(int entry = offBegin; entry < offEnd; entry += 6)
			{
				const int offset = entry - offBegin;
				Eigen::Map<Vec6> destination(solution + inner[entry]);
				Vec6 accumulated = destination;
				for(int axis = 0; axis < 6; ++axis)
					if(current[axis] != 0.0)
					{
						const int address = outer[first + axis] + 6 - axis + offset;
						const Eigen::Map<const Vec6> coefficients(values + address);
						accumulated.noalias() -= current[axis] * coefficients;
					}
				destination = accumulated;
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
				current -= values[entry] * solution[inner[entry]];
			for(; entry < end; entry += 6)
			{
				// Evaluate independent products in packets, then retain the same
				// left-to-right subtraction order instead of regrouping a dot sum.
				const Eigen::Map<const Vec6> coefficients(values + entry);
				const Eigen::Map<const Vec6> solved(solution + inner[entry]);
				const Vec6 products = coefficients.cwiseProduct(solved);
				for(int axis = 0; axis < 6; ++axis)
					current -= products[axis];
			}
			solution[column] = current / values[begin];
		}
	}

	bool refactor(const Problem& problem, const Curvature& weights, Result& result)
	{
		const bool firstFactor = m_size == 0;
		bool changedPattern = false;
		const Clock::time_point assemblyStart = profileStart(m_profile);
		const Sparse& matrix = makeHessian(problem, weights, m_hessian);
		const double assemblyMs = profileElapsed(m_profile, assemblyStart);
		result.matrixMs += assemblyMs;
		const Clock::time_point factorStart = profileStart(m_profile);
		if(m_size == 0)
		{
			const Clock::time_point symbolicStart = profileStart(m_profile);
			const bool samePattern = m_outer.size() == size_t(matrix.outerSize() + 1) &&
				m_inner.size() == size_t(matrix.nonZeros()) &&
				std::equal(m_outer.begin(), m_outer.end(), matrix.outerIndexPtr()) &&
				std::equal(m_inner.begin(), m_inner.end(), matrix.innerIndexPtr());
			if(!samePattern)
			{
				changedPattern = true;
				orderBodies(matrix);
				m_outer.assign(matrix.outerIndexPtr(), matrix.outerIndexPtr() + matrix.outerSize() + 1);
				m_inner.assign(matrix.innerIndexPtr(), matrix.innerIndexPtr() + matrix.nonZeros());
				preparePermutation(matrix);
				m_factor.analyzePattern(m_permuted, false);
				++result.symbolicAnalyses;
			}
			result.symbolicMs += profileElapsed(m_profile, symbolicStart);
		}
		m_size = int(matrix.rows());
		for(int entry = 0; entry < matrix.nonZeros(); ++entry)
			m_permuted.valuePtr()[entry] = matrix.valuePtr()[m_permutedSourceIndices[entry]];
		if(!m_factor.factorize<false>(m_permuted))
			return false;
		if(firstFactor)
		{
			// Numeric-factor work is proportional to squared column lengths;
			// a rank update touches the elimination-tree reach once per column.
			// Cache those reach costs so selection is linear in changed contacts.
			if(changedPattern)
			{
				const Sparse& lower = m_factor.m_matrix;
				m_reachWork.resize(m_size);
				m_factorWork = double(matrix.nonZeros());
				for(int column = m_size - 1; column >= 0; --column)
				{
					const int begin = lower.outerIndexPtr()[column], end = lower.outerIndexPtr()[column + 1];
					const int count = end - begin - 1;
					const int parent = count ? lower.innerIndexPtr()[begin + 1] : m_size;
					m_reachWork[column] = 6.0 * count + 12.0 + (parent < m_size ? m_reachWork[parent] : 0.0);
					m_factorWork += double(count) * (count + 3) + 12.0;
				}
			}
			m_refactorWork = m_factorWork;
			// Include the local contact outer products in the rebuild estimate.
			// Counting potential entries conservatively includes inactive rows.
			// Fixed bilateral curvature never invokes the update/refactor choice.
			if(problem.equalityRows != problem.rowCount())
				for(const CompactContact& contact : problem.contacts)
				{
					const int first = contact.rowCount() == 1 ? 2 : 0;
					for(int axis = first; axis < 3; ++axis)
					{
						int count = 0;
						for(int end = 0; end < 2; ++end)
							if(contact.body[end] >= 0)
								for(int component = 0; component < 6; ++component)
									count += problem.contactEntry(contact, end, axis, component) != 0.0;
						m_refactorWork += double(count) * (count + 3) * (contact.block > 0 ? 3.0 : 1.0);
					}
				}
		}
		++result.factorizations;
		result.factorMs += profileElapsed(m_profile, factorStart);
		m_weights = weights;
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
				if(row != previous)
				{
					m_bodyEdges.push_back(BodyPair(body, row));
					if(row != body)
						m_bodyEdges.push_back(BodyPair(row, body));
					previous = row;
				}
			}
		}
		std::sort(m_bodyEdges.begin(), m_bodyEdges.end());
		m_bodyGraph.resize(bodies, bodies);
		m_bodyGraph.resizeNonZeros(m_bodyEdges.size());
		int edge = 0;
		for(int body = 0; body < bodies; ++body)
		{
			m_bodyGraph.outerIndexPtr()[body] = edge;
			while(edge < int(m_bodyEdges.size()) && m_bodyEdges[edge].first == body)
			{
				m_bodyGraph.innerIndexPtr()[edge] = m_bodyEdges[edge].second;
				++edge;
			}
		}
		m_bodyGraph.outerIndexPtr()[bodies] = edge;
		Eigen::internal::orderSparse(m_bodyGraph, m_bodyOrder, m_orderingWorkspace);
		m_permutation.resize(matrix.cols());
		for(int body = 0; body < bodies; ++body)
			for(int axis = 0; axis < 6; ++axis)
				m_permutation[6 * m_bodyOrder[body] + axis] = 6 * body + axis;
	}

	void preparePermutation(const Sparse& matrix)
	{
		// Bucket the permuted upper triangle by row, then scatter rows in order
		// into CSC columns. Both passes are linear and produce sorted row indices
		// without sorting every scalar entry of each complete body block.
		const int size = int(matrix.cols());
		m_permutationRowOffsets.resize(size + 1);
		m_permutationCursors.assign(size, 0);
		reserveStorage(m_permutedEntries, size_t(matrix.nonZeros()));
		m_permutedEntries.resize(matrix.nonZeros());
		reserveStorage(m_permutedSourceIndices, size_t(matrix.nonZeros()));
		m_permutedSourceIndices.resize(matrix.nonZeros());
		for(int column = 0; column < size; ++column)
			for(int entry = matrix.outerIndexPtr()[column]; entry < matrix.outerIndexPtr()[column + 1]; ++entry)
				++m_permutationCursors[std::min(m_permutation[column], m_permutation[matrix.innerIndexPtr()[entry]])];
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
			for(int entry = matrix.outerIndexPtr()[column]; entry < matrix.outerIndexPtr()[column + 1]; ++entry)
			{
				const int a = m_permutation[column];
				const int b = m_permutation[matrix.innerIndexPtr()[entry]];
				PermutedEntry& mapped = m_permutedEntries[m_permutationCursors[std::min(a, b)]++];
				mapped.column = std::max(a, b);
				mapped.sourceIndex = entry;
			}
		// Reuse the scatter cursors for columns after the row buckets are filled.
		std::fill(m_permutationCursors.begin(), m_permutationCursors.end(), 0);
		for(size_t entry = 0; entry < m_permutedEntries.size(); ++entry)
			++m_permutationCursors[m_permutedEntries[entry].column];
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
			for(int entry = m_permutationRowOffsets[row]; entry < m_permutationRowOffsets[row + 1]; ++entry)
			{
				const PermutedEntry& mapped = m_permutedEntries[entry];
				const int destination = m_permutationCursors[mapped.column]++;
				m_permuted.innerIndexPtr()[destination] = row;
				m_permutedSourceIndices[destination] = mapped.sourceIndex;
			}
	}

	bool updateSparse(const Problem& problem, const CompactContact& contact, const Vec3& axis, bool add)
	{
		Vec6 vector[2];
		for(int end = 0; end < 2; ++end)
			if(contact.body[end] >= 0)
			{
				if(contact.rowCount() == 3)
					vector[end] = problem.contactJacobian(contact, end).transpose() * axis;
				else
					vector[end] = contact.jacobian[end] * axis[2];
			}
		return updatePair(contact.body, vector, add);
	}

	bool updatePair(const int* body, const Vec6* vector, bool add)
	{
		// Preserve the established signed-update kernel's pivot floor.
		static constexpr double minimumPivotSquare = 1.0e-15;
		// The allocated Hessian contains the complete clique of each contact's
		// endpoint coordinates, including inactive rows. Consequently its rank
		// update reaches a single ancestor path of the Cholesky elimination tree.
		// The first off-diagonal row in a factor column is its parent. A retained
		// dense work vector avoids rebuilding/merging sparse index lists at every
		// pivot, while the factor's sparse columns bound all numerical work.
		m_vector.resize(m_size);
		std::fill(m_vector.begin(), m_vector.end(), 0.0);
		int first = m_size;
		for(int end = 0; end < 2; ++end)
			if(body[end] >= 0)
			{
				const Vec6& row = vector[end];
				for(int component = 0; component < 6; ++component)
					if(row[component] != 0.0)
					{
						const int column = m_permutation[6 * body[end] + component];
						m_vector[column] = row[component];
						first = std::min(first, column);
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
				const double square = diagonal * diagonal + (add ? x * x : -x * x);
				// Reject lost rank and nonfinite arithmetic; the caller immediately
				// rebuilds the target Hessian with the full factorization path.
				if(!(square >= minimumPivotSquare) || !std::isfinite(square))
					return false;
				const double r = std::sqrt(square);
				const double c = r / diagonal, s = x / diagonal;
				const double inverseC = 1.0 / c, signedSC = (add ? s : -s) / c;
				values[begin] = r;
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
					Eigen::Map<Eigen::Array<double, 6, 1>> factor(values + entry);
					Eigen::Map<Eigen::Array<double, 6, 1>> work(m_vector.data() + inner[entry]);
					factor = inverseC * factor + signedSC * work;
					work = c * work - s * factor;
				}
				m_vector[column] = 0.0;
			}
			column = parent;
		}
		return true;
	}

	int m_size;
	bool m_profile;
	BlockCholesky m_factor;
	SparseStorage m_bodyGraph;
	std::vector<BodyPair> m_bodyEdges;
	std::vector<int> m_bodyOrder, m_orderingWorkspace, m_permutation;
	HessianStorage m_hessian;
	SparseStorage m_permuted;
	struct PermutedEntry
	{
		int column;
		int sourceIndex;
	};
	std::vector<PermutedEntry> m_permutedEntries;
	std::vector<int> m_permutedSourceIndices, m_permutationRowOffsets, m_permutationCursors;
	std::vector<int> m_outer, m_inner;
	std::vector<double> m_vector, m_solution, m_reachWork;
	double m_refactorWork = 0.0, m_factorWork = 0.0;
	Curvature m_weights;
	std::vector<RankUpdate> m_updates;
	std::vector<PatchUpdate> m_patchUpdates;
};
