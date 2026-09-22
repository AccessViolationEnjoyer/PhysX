#include "foundation/PxFoundation.h"
#include "foundation/PxPhysicsVersion.h"
#include "extensions/PxDefaultAllocator.h"
#include "extensions/PxDefaultErrorCallback.h"
#include "DyConstraintPrep.h"
#include "CmSpatialVector.h"
#include "newton/DyNewtonConstraintPrep.h"
#include "newton/core/NewtonSolver.h"
#include "solver/PxSolverDefs.h"
#include <cmath>
#include <cstdio>

using namespace physx;
using namespace physx::Dy;

struct JointFixture
{
	Px1DConstraint rows[MAX_CONSTRAINT_ROWS];
	PxConstraintInvMassScale scales;
	PxVec3 offset;
	PxU32 count;
	PxU32 calls;
	bool extended;

	JointFixture() : scales(1.0f, 1.0f, 1.0f, 1.0f), offset(0.0f), count(1), calls(0), extended(false)
	{
		setupConstraintRows(rows, MAX_CONSTRAINT_ROWS);
		rows[0].linear0 = PxVec3(1.0f, 0.0f, 0.0f);
		rows[0].linear1 = rows[0].linear0;
	}
};

static PxU32 prepareRows(Px1DConstraint* rows, PxVec3p& offset, PxU32 maxRows,
	PxConstraintInvMassScale& scales, const void* data, const PxTransform&, const PxTransform&,
	bool extended, PxVec3p&, PxVec3p&)
{
	JointFixture& fixture = *const_cast<JointFixture*>(static_cast<const JointFixture*>(data));
	PX_ASSERT(maxRows >= fixture.count);
	PX_UNUSED(maxRows);
	PxMemCopy(rows, fixture.rows, fixture.count * sizeof(Px1DConstraint));
	offset = fixture.offset;
	scales = fixture.scales;
	fixture.extended = extended;
	++fixture.calls;
	return fixture.count;
}

static int failures = 0;

static void check(bool condition, const char* message)
{
	if(!condition)
	{
		printf("FAIL %s\n", message);
		++failures;
	}
}

static bool close(double actual, double expected)
{
	return std::abs(actual - expected) <= 1.0e-7 * std::max(1.0, std::abs(expected));
}

static void initializeProblem(newton::Problem& problem)
{
	problem.clearContacts();
	problem.timestep = 0.01;
	problem.inverseMass.assign(2, 1.0);
	problem.massDiagonal.resize(12);
	problem.massDiagonal.setOnes();
	problem.freeBodyVelocity.setZero(12);
}

static Constraint makeConstraint(JointFixture& fixture)
{
	Constraint constraint;
	PxMemZero(&constraint, sizeof(constraint));
	constraint.solverPrep = prepareRows;
	constraint.constantBlock = &fixture;
	constraint.linBreakForce = constraint.angBreakForce = PX_MAX_F32;
	return constraint;
}

static PxSolverBodyData makeBody(PxReal inverseMass, PxReal velocity)
{
	PxSolverBodyData body;
	PxMemZero(&body, sizeof(body));
	body.invMass = inverseMass;
	body.linearVelocity = PxVec3(velocity, 0.0f, 0.0f);
	body.sqrtInvInertia = PxMat33(PxVec3(1.0f, 0.0f, 0.0f), PxVec3(0.0f, 2.0f, 0.0f), PxVec3(0.0f, 0.0f, 3.0f));
	body.body2World = PxTransform(PxIdentity);
	return body;
}

static NewtonJointSettings makeSettings(const PxSolverBodyData& body0, const PxSolverBodyData& body1,
	Cm::SpatialVector* initialVelocities)
{
	initialVelocities[0] = Cm::SpatialVector(body0.linearVelocity, body0.angularVelocity);
	initialVelocities[1] = Cm::SpatialVector(body1.linearVelocity, body1.angularVelocity);
	NewtonJointSettings settings;
	settings.timestep = 0.01f;
	settings.regularization = 1.0e-4f;
	settings.initialVelocities = initialVelocities;
	return settings;
}

static double hardJointFreeVelocity(const Px1DConstraint& row, const NewtonJointSettings& settings)
{
	const double impedance = 1.0 / (1.0 + double(settings.regularization));
	const double timeConstant = std::max(0.02, 2.0 * double(settings.timestep));
	const double damping = 2.0 / (impedance * timeConstant);
	const double stiffness = 1.0 / (impedance * impedance * timeConstant * timeConstant);
	return double(settings.timestep) *
		(damping * -double(row.velocityTarget) + stiffness * impedance * double(row.geometricError));
}

static double solveImpulse(newton::Problem& problem)
{
	newton::prepareProblem(problem);
	newton::Result result;
	newton::Workspace workspace;
	newton::Settings settings;
	settings.tolerance = 1.0e-12;
	check(newton::solveNewton(problem, settings, result, workspace) == newton::SolveStatus::eSUCCESS, "solve scalar joint problem");
	return result.impulse[0];
}

static void testSprings()
{
	const PxSolverBodyData body0 = makeBody(0.25f, 2.0f);
	const PxSolverBodyData body1 = makeBody(2.0f, -1.0f);
	Cm::SpatialVector initialVelocities[2];
	const NewtonJointSettings settings = makeSettings(body0, body1, initialVelocities);
	for(PxU32 acceleration = 0; acceleration < 2; ++acceleration)
	{
		JointFixture fixture;
		Px1DConstraint& row = fixture.rows[0];
		row.flags = Px1DConstraintFlag::eSPRING | (acceleration ? Px1DConstraintFlag::eACCELERATION_SPRING : 0);
		row.mods.spring.stiffness = 1200.0f;
		row.mods.spring.damping = 35.0f;
		row.geometricError = 0.2f;
		row.velocityTarget = 0.6f;
		Constraint constraint = makeConstraint(fixture);
		NewtonJointRows output;
		newton::Problem problem;
		initializeProblem(problem);
		prepareNewtonJoint(constraint, body0, body1, 0, 1, NULL, settings, problem, output);
		const double dt = settings.timestep;
		const double a = dt * (row.mods.spring.damping + dt * row.mods.spring.stiffness);
		const double b = dt * (double(row.mods.spring.damping) * row.velocityTarget - double(row.mods.spring.stiffness) * row.geometricError);
		const double response = 2.25;
		const double expected = (b - a * 3.0) / (acceleration ? response * (1.0 + a) : 1.0 + a * response);
		check(close(solveImpulse(problem), expected), "implicit spring matches native scalar law");
		check(fixture.calls == 1, "spring callback called once");
	}
}

static void testBoundsAndReferences()
{
	const PxSolverBodyData body0 = makeBody(1.0f, 0.0f);
	const PxSolverBodyData body1 = makeBody(0.0f, 0.0f);
	Cm::SpatialVector initialVelocities[2];
	const NewtonJointSettings settings = makeSettings(body0, body1, initialVelocities);
	JointFixture fixture;
	fixture.rows[0].velocityTarget = 100.0f;
	fixture.rows[0].minImpulse = -1.0f;
	fixture.rows[0].maxImpulse = 2.0f;
	fixture.rows[0].flags = Px1DConstraintFlag::eHAS_DRIVE_LIMIT;
	Constraint constraint = makeConstraint(fixture);
	constraint.flags = PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES | PxConstraintFlag::eENABLE_EXTENDED_LIMITS;
	newton::Problem problem;
	initializeProblem(problem);
	NewtonJointRows output;
	prepareNewtonJoint(constraint, body0, body1, 0, -1, NULL, settings, problem, output);
	check(close(solveImpulse(problem), 2.0 * settings.timestep), "drive force limit converted to impulse");
	check(fixture.extended, "extended limit flag forwarded");

	initializeProblem(problem);
	output.clear();
	fixture.rows[0].flags = 0;
	fixture.rows[0].velocityTarget = 0.0f;
	fixture.rows[0].geometricError = 0.25f;
	fixture.rows[0].minImpulse = -PX_MAX_F32;
	fixture.rows[0].maxImpulse = PX_MAX_F32;
	prepareNewtonJoint(constraint, body0, body1, 0, -1, NULL, settings, problem, output);
	const double expectedHardImpulse = -hardJointFreeVelocity(fixture.rows[0], settings) /
		(1.0 + settings.regularization);
	check(close(solveImpulse(problem), expectedHardImpulse), "hard joint uses the MuJoCo reference acceleration");
	check(fixture.calls == 2, "each preparation invokes the callback once");

	initializeProblem(problem);
	output.clear();
	fixture.rows[0].flags = Px1DConstraintFlag::eRESTITUTION;
	fixture.rows[0].mods.bounce.restitution = 0.5f;
	fixture.rows[0].mods.bounce.velocityThreshold = 0.1f;
	fixture.rows[0].geometricError = -0.1f;
	const PxSolverBodyData falling = makeBody(1.0f, -2.0f);
	prepareNewtonJoint(constraint, falling, body1, 0, -1, NULL, settings, problem, output);
	check(close(solveImpulse(problem), 3.0 / (1.0 + settings.regularization)), "restitution replaces error and target");
}

static void testWhiteningAndWriteback()
{
	PxSolverBodyData body0 = makeBody(0.25f, 0.0f);
	const PxMat33 rotation(PxQuat(0.7f, PxVec3(0.0f, 1.0f, 0.0f)));
	body0.sqrtInvInertia = rotation * body0.sqrtInvInertia * rotation.getTranspose();
	const PxSolverBodyData body1 = makeBody(0.0f, 0.0f);
	Cm::SpatialVector initialVelocities[2];
	const NewtonJointSettings settings = makeSettings(body0, body1, initialVelocities);
	JointFixture fixture;
	fixture.rows[0].angular0 = PxVec3(0.0f, 0.0f, 2.0f);
	fixture.rows[0].flags = Px1DConstraintFlag::eOUTPUT_FORCE;
	fixture.offset = PxVec3(0.0f, 1.0f, 0.0f);
	Constraint constraint = makeConstraint(fixture);
	constraint.angBreakForce = 2.0f;
	ConstraintWriteback writeback;
	writeback.initialize();
	newton::Problem problem;
	initializeProblem(problem);
	NewtonJointRows output;
	prepareNewtonJoint(constraint, body0, body1, 0, -1, &writeback, settings, problem, output);
	const PxVec3 angular = body0.sqrtInvInertia * fixture.rows[0].angular0;
	check(close(problem.contacts[0].jacobian[0][0], 0.5), "linear inverse mass whitening");
	for(PxU32 axis = 0; axis < 3; ++axis)
		check(close(problem.contacts[0].jacobian[0][axis + 3], angular[axis]), "full world inertia whitening");
	newton::prepareProblem(problem);
	newton::Result result;
	result.impulse.resize(1);
	result.impulse.setConstant(0.5);
	writebackNewtonJoints(output, problem, result);
	check(close(writeback.linearImpulse.x, 0.5) && close(writeback.angularImpulse.z, 1.5), "unscaled force reporting and anchor moment");
	check(writeback.broken != 0, "angular break impulse threshold");
}

static void testSlerpDrives()
{
	PxSolverBodyData body0 = makeBody(1.0f, 0.0f);
	const PxMat33 rotation(PxQuat(0.7f, PxVec3(0.0f, 1.0f, 0.0f)));
	body0.sqrtInvInertia = rotation * body0.sqrtInvInertia * rotation.getTranspose();
	body0.angularVelocity = PxVec3(0.7f, -0.2f, 0.5f);
	const PxSolverBodyData body1 = makeBody(0.0f, 0.0f);
	Cm::SpatialVector initialVelocities[2];
	const NewtonJointSettings settings = makeSettings(body0, body1, initialVelocities);
	const PxVec3 principalVelocity = rotation.transformTranspose(body0.angularVelocity);
	const PxVec3 inverseInertia(1.0f, 4.0f, 9.0f);
	for(PxU32 acceleration = 0; acceleration < 2; ++acceleration)
	{
		for(PxU32 capped = 0; capped < 2; ++capped)
		{
			JointFixture fixture;
			fixture.count = 4;
			for(PxU32 axis = 0; axis < 3; ++axis)
			{
				Px1DConstraint& row = fixture.rows[axis];
				row.linear0 = row.linear1 = PxVec3(0.0f);
				row.angular0[axis] = 1.0f;
				row.angular1[axis] = 1.0f;
				row.flags = Px1DConstraintFlag::eSPRING | Px1DConstraintFlag::eHAS_DRIVE_LIMIT;
				if(acceleration)
					row.flags |= Px1DConstraintFlag::eACCELERATION_SPRING;
				row.mods.spring.damping = 35.0f;
				row.solveHint = PxConstraintSolveHint::eSLERP_SPRING;
				row.minImpulse = capped ? -3.0f : -PX_MAX_F32;
				row.maxImpulse = capped ? 3.0f : PX_MAX_F32;
			}
			fixture.rows[3].linear0 = PxVec3(0.0f, 1.0f, 0.0f);
			fixture.rows[3].geometricError = 0.25f;
			fixture.rows[3].solveHint = PxConstraintSolveHint::eEQUALITY;
			Constraint constraint = makeConstraint(fixture);
			constraint.flags = PxConstraintFlag::eIMPROVED_SLERP | PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES;
			NewtonJointRows output;
			newton::Problem problem;
			initializeProblem(problem);
			prepareNewtonJoint(constraint, body0, body1, 0, -1, NULL, settings, problem, output);
			check(problem.contacts.size() == 4 && problem.contacts[3].jacobian[0][1] == 1.0 &&
				close(problem.contacts[3].freeVelocity, hardJointFreeVelocity(fixture.rows[3], settings)),
				"SLERP preprocessing leaves other rows unchanged");
			newton::prepareProblem(problem);
			newton::Result result;
			newton::Workspace workspace;
			newton::Settings solveSettings;
			solveSettings.tolerance = 1.0e-12;
			check(newton::solveNewton(problem, solveSettings, result, workspace) == newton::SolveStatus::eSUCCESS, "solve SLERP problem");

			PxVec3 expectedPrincipal = principalVelocity;
			const double a = double(settings.timestep) * 35.0;
			for(PxU32 axis = 0; axis < 3; ++axis)
			{
				const double response = inverseInertia[axis];
				double impulse = -a * principalVelocity[axis] /
					(acceleration ? response * (1.0 + a) : 1.0 + response * a);
				if(capped)
					impulse = PxClamp(impulse, -3.0 * settings.timestep, 3.0 * settings.timestep);
				expectedPrincipal[axis] += PxReal(response * impulse);
			}
			const PxVec3 expected = rotation * expectedPrincipal;
			const PxVec3 correction(PxReal(result.primal[3]), PxReal(result.primal[4]), PxReal(result.primal[5]));
			const PxVec3 actual = body0.angularVelocity + body0.sqrtInvInertia * correction;
			check((actual - expected).magnitude() < 3.0e-6f, "SLERP damping and caps match analytic principal-axis impulses");
			check(fixture.calls == 1, "SLERP callback invoked once");

			constraint.flags |= PxConstraintFlag::eDISABLE_PREPROCESSING;
			initializeProblem(problem);
			output.clear();
			prepareNewtonJoint(constraint, body0, body1, 0, -1, NULL, settings, problem, output);
			for(PxU32 axis = 0; axis < 3; ++axis)
				check(close(problem.contacts[0].jacobian[0][axis + 3], body0.sqrtInvInertia.column0[axis]),
					"disabled preprocessing preserves callback axes");
		}
	}
}

static void testEmptyAndUnsupported()
{
	const PxSolverBodyData body0 = makeBody(1.0f, 0.0f);
	const PxSolverBodyData body1 = makeBody(0.0f, 0.0f);
	Cm::SpatialVector initialVelocities[2];
	const NewtonJointSettings settings = makeSettings(body0, body1, initialVelocities);
	JointFixture fixture;
	fixture.rows[0].flags = Px1DConstraintFlag::eSPRING;
	Constraint constraint = makeConstraint(fixture);
	newton::Problem problem;
	initializeProblem(problem);
	NewtonJointRows output;
	prepareNewtonJoint(constraint, body0, body1, 0, -1, NULL, settings, problem, output);
	check(problem.contacts.empty(), "zero spring emits no row");
	fixture.rows[0].minImpulse = 0.5f;
	fixture.rows[0].maxImpulse = 2.0f;
	prepareNewtonJoint(constraint, body0, body1, 0, -1, NULL, settings, problem, output);
	check(close(solveImpulse(problem), 0.5), "zero spring obeys explicit nonzero bound");
	const PxU32 calls = fixture.calls;
	constraint.flags = PxConstraintFlag::eDISABLE_CONSTRAINT;
	prepareNewtonJoint(constraint, body0, body1, 0, -1, NULL, settings, problem, output);
	check(fixture.calls == calls, "disabled callback skipped");
	constraint.flags = PxConstraintFlag::eBROKEN;
	prepareNewtonJoint(constraint, body0, body1, 0, -1, NULL, settings, problem, output);
	check(fixture.calls == calls, "broken callback skipped");
}

int main()
{
	PxDefaultAllocator allocator;
	PxDefaultErrorCallback errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	testSprings();
	testBoundsAndReferences();
	testWhiteningAndWriteback();
	testSlerpDrives();
	testEmptyAndUnsupported();
	foundation->release();
	printf("Newton joint adapter: %d failures\n", failures);
	return failures ? 1 : 0;
}
