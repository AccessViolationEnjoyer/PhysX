# Private numerical core shared by native PhysX and standalone validation builds.
# Dependencies and precise arithmetic options must not propagate to PGS/TGS.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(NEWTON_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/eigen")
set(NEWTON_STORAGE_KERNELS "${CMAKE_CURRENT_BINARY_DIR}/EigenStorageKernels.h")
add_custom_command(OUTPUT "${NEWTON_STORAGE_KERNELS}"
	COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/PrepareStorageKernels.py"
		"${NEWTON_EIGEN_DIR}" "${NEWTON_STORAGE_KERNELS}"
	DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/PrepareStorageKernels.py"
		"${NEWTON_EIGEN_DIR}/Eigen/src/OrderingMethods/Amd.h"
		"${NEWTON_EIGEN_DIR}/Eigen/src/SparseCholesky/SimplicialCholesky_impl.h"
	VERBATIM)
add_library(PhysXNewtonCore OBJECT
	NewtonSolver.cpp NewtonSolver.h NewtonParallel.h NewtonStorage.h BlockCholesky.h IncrementalCholesky.h
	"${NEWTON_STORAGE_KERNELS}")
target_include_directories(PhysXNewtonCore PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(PhysXNewtonCore SYSTEM PRIVATE "${NEWTON_EIGEN_DIR}")
# Match Eigen inline definitions across the core and native adapter on all platforms.
target_compile_definitions(PhysXNewtonCore PRIVATE EIGEN_DONT_PARALLELIZE $<$<NOT:$<CONFIG:debug>>:NDEBUG>)
set_target_properties(PhysXNewtonCore PROPERTIES
	CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON)
if(MSVC)
	target_compile_options(PhysXNewtonCore PRIVATE /fp:precise /EHsc)
else()
	target_compile_options(PhysXNewtonCore PRIVATE -fno-fast-math -ffp-contract=off -fexceptions)
endif()
