// Nonlinear sphere-chain replay: spherical anchors and orientation springs.
// Included by NewtonBenchmark.cpp.
static Vec3 cableVector(const Eigen::Vector3d& value)
{
	return Vec3(value[0], value[1], value[2]);
}

static Eigen::Vector3d cableVector(const Vec3& value)
{
	return Eigen::Vector3d(value[0], value[1], value[2]);
}

static void multiplyCableJacobian(const Problem& problem, const Vector& vector, Vector& product, bool transpose)
{
	const Sparse& jacobian = problem.jacobian;
	const int* outer = jacobian.outerIndexPtr();
	const int* inner = jacobian.innerIndexPtr();
	const double* values = jacobian.valuePtr();
	product.setZero(transpose ? jacobian.cols() : jacobian.rows());
	for(int column = 0; column < jacobian.cols(); ++column)
		for(int entry = outer[column]; entry < outer[column + 1]; ++entry)
			if(transpose)
				product[column] += values[entry] * vector[inner[entry]];
			else
				product[inner[entry]] += values[entry] * vector[column];
}

static void makeCableProblem(Problem& problem, const std::vector<Vec3>& positions, const std::vector<Eigen::Quaterniond>& rotations,
	const Vector& velocity, double dt, double anchorRegularization, bool gravity)
{
	const int links = int(positions.size());
	const double mass = 0.1, length = 0.05, inverseRootMass = std::sqrt(1.0 / mass), inverseRootInertia = 200.0;
	const double stiffness = 100.0 * 180.0 / double(3.1415927410125732421875f), damping = 20.0 * 180.0 / double(3.1415927410125732421875f);
	problem.clearContacts();
	problem.name = "cable";
	problem.timestep = dt;
	problem.inverseMass.assign(links, 1.0 / mass);
	setMassDiagonal(problem, 1.0 / (mass * inverseRootInertia * inverseRootInertia));
	problem.freeBodyVelocity = velocity;
	problem.freeBodyVelocity[6 * (links - 1) + 1] -= dt * inverseRootMass; // Constant 1 N tip force.
	if(gravity)
		for(int i = 0; i < links; ++i)
			problem.freeBodyVelocity[6 * i + 1] -= 9.81 * dt / inverseRootMass;
	for(int i = 0; i < links; ++i)
	{
		const Vec3 base = i ? positions[i - 1] : Vec3::Zero();
		const Eigen::Quaterniond rotation = i ? rotations[i - 1] : Eigen::Quaterniond::Identity();
		const Vec3 arm = cableVector(rotation * Eigen::Vector3d(length, 0.0, 0.0));
		const Vec3 anchorError = positions[i] - base - arm;
		Eigen::Quaterniond relative = rotation.conjugate() * rotations[i];
		if(relative.w() < 0.0)
			relative.coeffs() *= -1.0;
		const double sine = relative.vec().norm();
		const Vec3 angleError = cableVector(rotation * (relative.vec() *
			(sine > 1.0e-12 ? 2.0 * std::atan2(sine, relative.w()) / sine : 2.0)));
		for(int angular = 0; angular < 2; ++angular)
		{
			Contact c;
			c.bilateral = true;
			c.body[0] = i;
			c.body[1] = i - 1;
			c.friction = 0.0;
			c.jacobian[0].setZero();
			c.jacobian[1].setZero();
			for(int axis = 0; axis < 3; ++axis)
			{
				if(angular)
				{
					c.jacobian[0](axis, axis + 3) = inverseRootInertia;
					if(i)
						c.jacobian[1](axis, axis + 3) = -inverseRootInertia;
					c.regularization[axis] = 1.0 / (dt * (dt * stiffness + damping));
					c.freeVelocity[axis] = stiffness / (dt * stiffness + damping) * angleError[axis];
				}
				else
				{
					c.jacobian[0](axis, axis) = inverseRootMass;
					if(i)
					{
						c.jacobian[1](axis, axis) = -inverseRootMass;
						const Vec3 angularJacobian = Vec3::Unit(axis).cross(arm) * inverseRootInertia;
						for(int column = 0; column < 3; ++column)
							c.jacobian[1](axis, column + 3) = double(float(angularJacobian[column]));
					}
					// Positive anchor compliance keeps the primal Hessian finite.
					// Measure drift to check its effect on the cable geometry.
					c.regularization[axis] = anchorRegularization *
						(c.jacobian[0].row(axis).squaredNorm() + c.jacobian[1].row(axis).squaredNorm());
					c.freeVelocity[axis] = 0.8 * anchorError[axis] / dt;
				}
			}
			c.freeVelocity += c.jacobian[0] * loadVector<6>(problem.freeBodyVelocity.data() + 6 * i);
			if(i)
				c.freeVelocity += c.jacobian[1] * loadVector<6>(problem.freeBodyVelocity.data() + 6 * (i - 1));
			problem.addContact(c);
		}
	}
	prepareProblem(problem);
}

static int benchmarkCable(int links, int steps, const char* path, const Settings& settings)
{
	const int threads = 1;
	const double dt = 0.01, anchorRegularization = 1.0e-9;
	const bool warm = true, gravity = false;
	if(links < 1 || steps < 2)
		throw std::runtime_error("Invalid cable benchmark parameters");

	omp_set_dynamic(0);
	std::vector<Vec3> positions(links);
	std::vector<Eigen::Quaterniond> rotations(links, Eigen::Quaterniond::Identity());
	for(int i = 0; i < links; ++i)
		positions[i] = Vec3(0.05 * (i + 1), 0.0, 0.0);
	Vector velocity = Vector::Zero(links * 6);
	std::ofstream trajectory;
	if(path)
		trajectory.open(path);
	trajectory << "step,time,tip_deflection,anchor_error,equation_residual,step_ms,solve_ms,iterations,factors,rank_updates,reused_factors\n";
	Result previous, result;
	Problem problem;
	Vector response(links * 6);
	std::vector<double> times, solveTimes;
	double maxAnchorError = 0.0, peakDeflection = 0.0, maximumResidual = 0.0, finalResidual = 0.0;
	int maxIterations = 0;
	for(int step = 0; step < steps; ++step)
	{
		const Clock::time_point start = Clock::now();
		makeCableProblem(problem, positions, rotations, velocity, dt, anchorRegularization, gravity);
		solveNewton(problem, settings, result, warm && step ? &previous : NULL);

		multiplyCableJacobian(problem, result.impulse, response, true);
		for(int row = 0; row < velocity.size(); ++row)
			velocity[row] = problem.freeBodyVelocity[row] + result.primal[row];
		for(int i = 0; i < links; ++i)
		{
			positions[i] += dt * std::sqrt(10.0) * velocity.segment<3>(6 * i);
			const Vec3 angular = 200.0 * dt * velocity.segment<3>(6 * i + 3);
			const double angle = angular.norm();
			if(angle > 0.0)
				rotations[i] = (Eigen::Quaterniond(Eigen::AngleAxisd(angle, cableVector(angular / angle))) * rotations[i]).normalized();
		}
		const double stepMs = elapsed(start);
		if(step >= std::min(60, steps / 2))
		{
			times.push_back(stepMs);
			solveTimes.push_back(result.elapsedMs);
		}
		Vector equation;
		multiplyCableJacobian(problem, result.primal, equation, false);
		finalResidual = 0.0;
		for(int row = 0; row < result.primal.size(); ++row)
			finalResidual = std::max(finalResidual, std::abs(result.primal[row] - response[row]));
		for(int row = 0; row < equation.size(); ++row)
			finalResidual = std::max(finalResidual, std::abs(equation[row] + problem.freeVelocity[row] +
				problem.regularization[row] * result.impulse[row]));
		maximumResidual = std::max(maximumResidual, finalResidual);
		double anchorError = 0.0;
		for(int i = 0; i < links; ++i)
		{
			const Vec3 anchor = i ? positions[i - 1] + cableVector(rotations[i - 1] * Eigen::Vector3d(0.05, 0.0, 0.0)) : Vec3(0.05, 0.0, 0.0);
			anchorError = std::max(anchorError, (positions[i] - anchor).norm());
		}
		maxAnchorError = std::max(maxAnchorError, anchorError);
		peakDeflection = std::max(peakDeflection, -positions.back()[1]);
		maxIterations = std::max(maxIterations, result.iterations);
		if(positions.back().norm() > 1000.0)
		{
			std::printf("CABLE_DIVERGED,step=%d,time=%.9g,tip=%.12g,anchor_error=%.12g,residual=%.12g,solve_ms=%.6f\n",
			step, (step + 1) * dt, -positions.back()[1], anchorError, finalResidual, result.elapsedMs);
			throw std::runtime_error("Cable simulation diverged");
		}
		trajectory << std::setprecision(12) << step << ',' << (step + 1) * dt << ',' << -positions.back()[1] << ','
			<< anchorError << ',' << finalResidual << ',' << stepMs << ',' << result.elapsedMs << ',' << result.iterations << ','
			<< result.factorizations << ',' << result.rankUpdates << ',' << result.reusedFactors << '\n';
		previous = result;
	}
	std::sort(times.begin(), times.end());
	std::sort(solveTimes.begin(), solveTimes.end());
	const double stiffness = 100.0 * 180.0 / double(3.1415927410125732421875f);
	const double expected = 0.05 * 0.05 * (links - 1.0) * links * (2.0 * links - 1.0) / (6.0 * stiffness);
	std::printf("CABLE_RESULT,links=%d,steps=%d,threads=%d,dt=%.9g,warm=%d,anchor_regularization=%.9g,tip=%.12g,beam_reference=%.12g,peak_tip=%.12g,anchor_error=%.12g,final_residual=%.12g,max_residual=%.12g,median_step_ms=%.6f,median_solve_ms=%.6f,max_iterations=%d\n",
		links, steps, threads, dt, int(warm), anchorRegularization, -positions.back()[1], expected, peakDeflection,
		maxAnchorError, finalResidual, maximumResidual, times[times.size() / 2], solveTimes[solveTimes.size() / 2], maxIterations);
	return 0;
}
