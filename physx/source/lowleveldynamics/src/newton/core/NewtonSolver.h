#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

#include "NewtonParallel.h"
#include "NewtonStorage.h"
#include "NewtonMath.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <limits>
#include <vector>

namespace newton
{
struct SolveStatus
{
	enum Enum
	{
		eSUCCESS,
		eITERATION_LIMIT,
		eINVALID_INPUT,
		eFACTORIZATION_FAILED,
		eNUMERICAL_FAILURE
	};
};
typedef VectorStorage Vector;
typedef const VectorStorage& ConstVector;
typedef VectorStorage& MutableVector;
typedef Matrix<2, 1> Vec2;
typedef Matrix<3, 1> Vec3;
typedef Matrix<3, 3> Mat3;
typedef Matrix<3, 6> Jacobian;
typedef Matrix<6, 1> Vec6;
typedef Matrix<2, 6> Mat26;
typedef Matrix<6, 6> Mat6;
typedef SparseStorage Sparse;
typedef std::chrono::steady_clock Clock;
static constexpr double MAX_IMPULSE = (std::numeric_limits<double>::max)();

// J is mass scaled: each column is divided by sqrt(mass or inertia).
// Contact rows are tangent 0, tangent 1, normal. Bilateral blocks hold three
// equality rows. Scalar nonnegative edges pack the normal row into one solver row.
struct Contact
{
	int rowCount() const { return bilateral || friction != 0.0 ? 3 : 1; }
	bool bilateral = false; // Three equality rows for joint/spring benchmarks.
	int body[2]; // Distinct dynamic bodies, or -1 for the fixed world.
	Jacobian jacobian[2]; // Rows are tangent 0, tangent 1, normal; mass-scaled columns.
	Vec3 freeVelocity;
	Vec3 regularization;
	double friction;
	double maxNormalImpulse = MAX_IMPULSE;
};

// The normal row is the complete scalar contact. Three-row contacts retain
// their two additional rows separately, so scalar records contain no padding
// Jacobians or duplicate solve representation.
struct CompactContact
{
	int row = 0;
	// Ordinary scalar rows need no sidecar. Small negative tags identify bilateral
	// blocks; the reserved lower range identifies optional scalar impulse bounds.
	enum { SCALAR_BOUNDS_TAG = -0x40000000 };
	int block = 0; // 0: nonnegative scalar; >0: friction block; <0: bilateral or bounds.
	int body[2];
	Vec6 jacobian[2];
	double freeVelocity;
	double regularization;
	bool hasScalarBounds() const { return block <= SCALAR_BOUNDS_TAG; }
	int rowCount() const { return block && !hasScalarBounds() ? 3 : 1; }
};

struct ScalarBounds
{
	double lower;
	double upper;
};

struct ContactBlock
{
	Mat26 tangentJacobian[2];
	Vec2 freeVelocity;
	Vec2 regularization;
	double friction;
	double maxNormalImpulse = MAX_IMPULSE;
	int coupled = -1;
};

// Native patch rows are contiguous scalar contacts: normals, then up to four
// tangent rows. Every row has the same ordered body pair. Normal bounds are
// [0,cap], tangent scalar bounds are unbounded; friction couples the group.
struct Patch
{
	int firstContact;
	int normalCount;
	int tangentCount;
	double friction; // May change between solves without rebuilding the Jacobian.
};

struct ScalarContactRun
{
	int first;
	int end;
};


struct Problem
{
	std::string name;
	double timestep;
	std::vector<double> inverseMass;
	std::vector<CompactContact> contacts;
	// Exactly one uniquely owned block per three-row contact; no shared or orphan blocks.
	std::vector<ContactBlock> contactBlocks;
	std::vector<ScalarBounds> scalarBounds;
	std::vector<Patch> patches;
	bool hasFiniteBounds = false;
	bool prepared = false;
	bool compactJacobian = false;
	// Core-assigned identity of prepared equations; bounds edits also invalidate continuation.
	std::uint64_t preparationGeneration = 0;
	SparseStorage jacobian;
	std::vector<int> columnCursors;
	std::vector<int> rowContact;
	std::vector<int> coupledContacts;
	std::vector<ScalarContactRun> scalarContactRuns;
	std::vector<std::uint64_t> hessianPairs;
	std::vector<int> hessianPairLookup;
	std::vector<int> hessianContactBlocks;
	std::vector<int> hessianDiagonalBlocks;
	int equalityRows = 0;
	// Structural arithmetic of a fresh factorization's contact outer products, counting
	// potential entries of inactive rows too. It is computed when the update or refactor
	// choice first needs it; solves that never update skip it.
	mutable double contactRebuildWork = -1.0;
	double rebuildWorkEstimate() const
	{
		if(contactRebuildWork < 0.0)
		{
			double work = 0.0;
			for(const CompactContact& contact : contacts)
			{
				work += rebuildWork(contact);
			}
			contactRebuildWork = work;
		}
		return contactRebuildWork;
	}
	bool isUnilateral() const { return equalityRows == 0 && coupledContacts.empty() && scalarBounds.empty() && patches.empty(); }
	VectorStorage freeVelocity;
	VectorStorage regularization;
	VectorStorage freeBodyVelocity;
	VectorStorage massDiagonal; // Physical mass/inertia diagonal for acceleration-space convergence tests.
	VectorStorage inverseMassDiagonal;
	int bodyCount() const { return int(inverseMass.size()); }
	int rowCount() const { return int(freeVelocity.size()); }
	void clearContacts()
	{
		contacts.clear();
		contactBlocks.clear();
		scalarBounds.clear();
		patches.clear();
		prepared = false;
		compactJacobian = false;
	}
	// Returns the new contact index; [0,MAX_IMPULSE] uses the ordinary scalar record.
	NEWTON_FORCE_INLINE int addScalarContact(const CompactContact& input, double lowerImpulse, double upperImpulse)
	{
		const CompactContact contact = input;
		int block = 0;
		if(lowerImpulse != 0.0 || upperImpulse != MAX_IMPULSE)
		{
			const ScalarBounds limits = { lowerImpulse, upperImpulse };
			block = CompactContact::SCALAR_BOUNDS_TAG - int(scalarBounds.size());
			reserveStorage(scalarBounds, std::uint32_t(scalarBounds.size()) + 1);
			scalarBounds.push_back(limits);
		}
		reserveStorage(contacts, std::uint32_t(contacts.size()) + 1);
		const int index = int(contacts.size());
		contacts.push_back(contact);
		contacts.back().row = 0;
		contacts.back().block = block;
		prepared = false;
		return index;
	}
	// Native preparation can construct hot scalar rows directly in retained
	// storage, avoiding a second 128-byte contact copy per emitted row.
	NEWTON_FORCE_INLINE CompactContact& beginScalarContact()
	{
		reserveStorage(contacts, std::uint32_t(contacts.size()) + 1);
		contacts.emplace_back();
		CompactContact& contact = contacts.back();
		contact.row = 0;
		contact.block = 0;
		prepared = false;
		return contact;
	}
	NEWTON_FORCE_INLINE void cancelScalarContact()
	{
		contacts.pop_back();
	}
	NEWTON_FORCE_INLINE void finishScalarContact(double lowerImpulse, double upperImpulse)
	{
		if(lowerImpulse != 0.0 || upperImpulse != MAX_IMPULSE)
		{
			const ScalarBounds limits = { lowerImpulse, upperImpulse };
			contacts.back().block = CompactContact::SCALAR_BOUNDS_TAG - int(scalarBounds.size());
			reserveStorage(scalarBounds, std::uint32_t(scalarBounds.size()) + 1);
			scalarBounds.push_back(limits);
		}
	}
	// Change an existing bounded scalar sidecar without rebuilding J or allocating.
	// Grouped patch rows keep their group contract; only ungrouped rows use this API.
	void setScalarBounds(int contactIndex, double lowerImpulse, double upperImpulse) noexcept;
	// Groups must be appended in increasing, nonoverlapping contact order.
	int addPatch(int firstContact, int normalCount, int tangentCount, double friction);
	const ScalarBounds& bounds(const CompactContact& contact) const
	{
		return scalarBounds[std::uint32_t(CompactContact::SCALAR_BOUNDS_TAG - contact.block)];
	}
	// Compatibility builder for full three-row inputs. Native scalar producers
	// can populate CompactContact records directly, without an intermediate Contact.
	void addContact(const Contact& input);
	const ContactBlock& block(const CompactContact& contact) const
	{
		return contactBlocks[std::uint32_t(contact.block > 0 ? contact.block - 1 : -contact.block - 1)];
	}
	ContactBlock& block(const CompactContact& contact)
	{
		return contactBlocks[std::uint32_t(contact.block > 0 ? contact.block - 1 : -contact.block - 1)];
	}
	int coupledIndex(const CompactContact& contact) const { return contact.block > 0 ? block(contact).coupled : -1; }
	Vec6 contactRow(const CompactContact& contact, int end, int axis) const
	{
		if(axis == 2)
		{
			return contact.jacobian[end];
		}
		return block(contact).tangentJacobian[end].row(axis).transpose();
	}
	double rebuildWork(const CompactContact& contact) const
	{
		if(contact.rowCount() == 1)
		{
			const int count = (contact.body[0] >= 0 ? nonzeroCount6(contact.jacobian[0].data()) : 0) +
				(contact.body[1] >= 0 ? nonzeroCount6(contact.jacobian[1].data()) : 0);
			return double(count) * (count + 3);
		}
		double work = 0.0;
		for(int axis = 0; axis < 3; ++axis)
		{
			int count = 0;
			for(int end = 0; end < 2; ++end)
			{
				if(contact.body[end] >= 0)
				{
					for(int column = 0; column < 6; ++column)
					{
						count += contactEntry(contact, end, axis, column) != 0.0;
					}
				}
			}
			work += double(count) * (count + 3) * (contact.block > 0 ? 3.0 : 1.0);
		}
		return work;
	}
	double contactEntry(const CompactContact& contact, int end, int axis, int column) const
	{
		return axis == 2 ? contact.jacobian[end][column] : block(contact).tangentJacobian[end](axis, column);
	}
	Jacobian contactJacobian(const CompactContact& contact, int end) const
	{
		Jacobian result;
		for(int column = 0; column < 6; ++column)
		{
			result(0, column) = block(contact).tangentJacobian[end](0, column);
			result(1, column) = block(contact).tangentJacobian[end](1, column);
			result(2, column) = contact.jacobian[end][column];
		}
		return result;
	}
};

struct SolverStatistics
{
	int iterations;
	int factorizations;
	int symbolicAnalyses = 0;
	int lineSearchEvaluations = 0;
	int rankUpdates = 0;
	int reusedFactors = 0;
	int factorFallbacks = 0;
	int interiorPointSteps = 0;
	double matrixMs = 0.0;
	double evaluationMs = 0.0;
	double lineSearchMs = 0.0;
	double conversionMs = 0.0;
	double factorMs = 0.0;
	double symbolicMs = 0.0;
	double updateMs = 0.0;
	double backsolveMs = 0.0;
	double factorError = 0.0;
	double scaledGradient = 0.0;
	double scaledImprovement = 0.0;
	int stopReason = 0; // 0: iteration limit, 1: gradient, 2: improvement, 5: precision, 6: line search.
	double gradientResidual = 0.0;
	double elapsedMs;
	SolverStatistics() : iterations(0), factorizations(0), elapsedMs(0.0) {}
};

struct Result : SolverStatistics
{
	SolveStatus::Enum status = SolveStatus::eINVALID_INPUT;
	VectorStorage impulse;
	VectorStorage primal; // Mass-scaled body velocity increment used for warm starting.
};

inline double elapsed(Clock::time_point start)
{
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Settings
{
	int iterations = 100;
	ParallelExecutor* parallelExecutor = NULL;
	double tolerance = 1.0e-8;
	double lineTolerance = 0.01;
	// Unilateral solves continue with interior-point steps once Newton has run at least
	// interiorPointSwitch iterations with three consecutive short line-search steps, or
	// four times that many iterations (negative disables). They return to Newton once
	// the scaled gradient is below interiorPointExit. Each step counts as an iteration.
	int interiorPointSwitch = 20;
	double interiorPointExit = 1.0e-3;
	// Rows whose interior-point weight times compliance is below this are omitted from
	// the factored Hessian only; interior-point step equations use every exact weight.
	double interiorPointDrop = 1.0e-7;
	// Continuations of an interior-point solve warm start with multipliers shifted by
	// this fraction of their mean (negative restarts with Newton).
	double interiorPointWarmShift = 1.0e-1;
	bool checkFactor = false;
	bool profile = false; // Detailed phase timers.
	// Measures Result::elapsedMs. Clock reads leave WebAssembly, so callers that do not
	// read the total can disable it.
	bool timing = true;
};

// Pack scalar rows and construct J from the contact blocks. Called during constraint preparation.
void prepareProblem(Problem& problem) noexcept;

// Producers can count entries while emitting rows: zero 6 * bodyCount() counters,
// then count the stored nonzero coefficients for each dynamic endpoint. Scalar
// contacts contribute their normal row; blocks contribute all three rows.
// addContact does not count. This call consumes columnCursors into CSC cursors;
// reset and recount before each preparation.
void prepareProblemFromColumnCounts(Problem& problem) noexcept;
// Native scalar-unilateral systems can solve directly from compact contact rows
// without materializing the duplicate CSC Jacobian. Other shapes fall back.
void prepareCompactProblemFromColumnCounts(Problem& problem) noexcept;

// Optional diagnostic; kept outside simulation solve timing.
double computeResidual(const Problem& problem, ConstVector impulse);

struct WorkspaceData;

// One synchronous solve may use a workspace at a time. The caller owns its
// lifetime and may retain/move it between jobs; storage grows only with capacity.
class Workspace
{
public:
	Workspace() noexcept;
	~Workspace();
	Workspace(Workspace&& other) noexcept;
	Workspace& operator=(Workspace&& other) noexcept;
	Workspace(const Workspace&) = delete;
	Workspace& operator=(const Workspace&) = delete;
private:
	WorkspaceData* m_data;
	friend SolveStatus::Enum solveNewton(const Problem&, const Settings&, Result&, Workspace&, const Result*) noexcept;
	friend SolveStatus::Enum continueNewton(const Problem&, const Settings&, Result&, Workspace&, const Result*) noexcept;
};

// Minimize 0.5*|v|^2 + sum phi(Jv + freeVelocity), where
// phi(s) = max_lambda(-s*lambda - 0.5*lambda'R*lambda) over each row/block's bounds.
// The scalar lambda is clamp(-s/R, lower, upper). Positive compliance is required.
// The caller retains workspace and result storage; previous may alias result.
// Failures return status without escaping through an engine task. Iteration-limit
// results remain available for callers that explicitly accept bounded iteration work.
SolveStatus::Enum solveNewton(const Problem& problem, const Settings& settings, Result& result, Workspace& workspace, const Result* previous = NULL) noexcept;

// Continue target-only solves of the same prepared problem without discarding the
// numerical factor. Only freeVelocity and patch friction may change directly;
// changing J, R, body mapping, row kinds or caps requires preparation (or the
// scalar-bound setter). Preparation, bounds edits, another Problem, and failed
// solves invalidate reuse automatically. An ordinary solve always starts a new
// numerical factor, and must begin each new island/timestep in native callers.
// Existing curvature updates/refactor fallback still produce the exact new H.
SolveStatus::Enum continueNewton(const Problem& problem, const Settings& settings, Result& result, Workspace& workspace, const Result* previous = NULL) noexcept;
}
#endif
