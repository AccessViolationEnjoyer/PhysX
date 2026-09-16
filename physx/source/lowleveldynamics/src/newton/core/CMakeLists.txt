# Private numerical core shared by native PhysX and standalone validation builds.
# Dependencies and precise arithmetic options must not propagate to PGS/TGS.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(NEWTON_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/eigen")
set(NEWTON_GKLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/gklib")
set(NEWTON_METIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/metis")
set(NEWTON_GKLIB_SOURCES
	${NEWTON_GKLIB_DIR}/src/blas.c
	${NEWTON_GKLIB_DIR}/src/error.c
	${NEWTON_GKLIB_DIR}/src/mcore.c
	${NEWTON_GKLIB_DIR}/src/memory.c
	${NEWTON_GKLIB_DIR}/src/random.c
	${NEWTON_GKLIB_DIR}/src/sort.c
	${NEWTON_GKLIB_DIR}/src/timers.c
)
set(NEWTON_METIS_SOURCES
	${NEWTON_METIS_DIR}/libmetis/auxapi.c
	${NEWTON_METIS_DIR}/libmetis/balance.c
	${NEWTON_METIS_DIR}/libmetis/bucketsort.c
	${NEWTON_METIS_DIR}/libmetis/checkgraph.c
	${NEWTON_METIS_DIR}/libmetis/coarsen.c
	${NEWTON_METIS_DIR}/libmetis/compress.c
	${NEWTON_METIS_DIR}/libmetis/contig.c
	${NEWTON_METIS_DIR}/libmetis/debug.c
	${NEWTON_METIS_DIR}/libmetis/fortran.c
	${NEWTON_METIS_DIR}/libmetis/fm.c
	${NEWTON_METIS_DIR}/libmetis/gklib.c
	${NEWTON_METIS_DIR}/libmetis/graph.c
	${NEWTON_METIS_DIR}/libmetis/initpart.c
	${NEWTON_METIS_DIR}/libmetis/kwayfm.c
	${NEWTON_METIS_DIR}/libmetis/kwayrefine.c
	${NEWTON_METIS_DIR}/libmetis/mcutil.c
	${NEWTON_METIS_DIR}/libmetis/minconn.c
	${NEWTON_METIS_DIR}/libmetis/mincover.c
	${NEWTON_METIS_DIR}/libmetis/mmd.c
	${NEWTON_METIS_DIR}/libmetis/ometis.c
	${NEWTON_METIS_DIR}/libmetis/options.c
	${NEWTON_METIS_DIR}/libmetis/pmetis.c
	${NEWTON_METIS_DIR}/libmetis/refine.c
	${NEWTON_METIS_DIR}/libmetis/separator.c
	${NEWTON_METIS_DIR}/libmetis/sfm.c
	${NEWTON_METIS_DIR}/libmetis/srefine.c
	${NEWTON_METIS_DIR}/libmetis/stat.c
	${NEWTON_METIS_DIR}/libmetis/timing.c
	${NEWTON_METIS_DIR}/libmetis/util.c
	${NEWTON_METIS_DIR}/libmetis/wspace.c
)
set(NEWTON_STORAGE_KERNELS "${CMAKE_CURRENT_BINARY_DIR}/EigenStorageKernels.h")
add_custom_command(OUTPUT "${NEWTON_STORAGE_KERNELS}"
	COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/PrepareStorageKernels.py"
		"${NEWTON_EIGEN_DIR}" "${NEWTON_STORAGE_KERNELS}"
	DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/PrepareStorageKernels.py"
		"${NEWTON_EIGEN_DIR}/Eigen/src/OrderingMethods/Amd.h"
		"${NEWTON_EIGEN_DIR}/Eigen/src/SparseCholesky/SimplicialCholesky_impl.h"
	VERBATIM)
add_library(PhysXNewtonCore OBJECT
	NewtonSolver.cpp NewtonSolver.h NewtonParallel.h NewtonStorage.h NewtonMetis.h BlockCholesky.h IncrementalCholesky.h
	${NEWTON_GKLIB_SOURCES} ${NEWTON_METIS_SOURCES} "${NEWTON_STORAGE_KERNELS}")
target_include_directories(PhysXNewtonCore PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
	"${NEWTON_GKLIB_DIR}/include" "${NEWTON_METIS_DIR}/include" "${NEWTON_METIS_DIR}/libmetis")
target_include_directories(PhysXNewtonCore SYSTEM PRIVATE "${NEWTON_EIGEN_DIR}")
set_source_files_properties(${NEWTON_GKLIB_SOURCES} PROPERTIES
	COMPILE_DEFINITIONS "NDEBUG;NDEBUG2")
set_source_files_properties(${NEWTON_METIS_SOURCES} PROPERTIES
	COMPILE_DEFINITIONS "METIS_NodeND=Newton_METIS_NodeND;METIS_SetDefaultOptions=Newton_METIS_SetDefaultOptions")
if(MSVC)
	set_property(SOURCE ${NEWTON_GKLIB_SOURCES} ${NEWTON_METIS_SOURCES} APPEND PROPERTY
		COMPILE_DEFINITIONS "__thread=__declspec(thread)")
	set_source_files_properties(${NEWTON_GKLIB_SOURCES} ${NEWTON_METIS_SOURCES} PROPERTIES COMPILE_OPTIONS /W0)
else()
	set_source_files_properties(${NEWTON_GKLIB_SOURCES} ${NEWTON_METIS_SOURCES} PROPERTIES COMPILE_OPTIONS -w)
endif()
# Match Eigen inline definitions across the core and native adapter on all platforms.
target_compile_definitions(PhysXNewtonCore PRIVATE EIGEN_DONT_PARALLELIZE $<$<NOT:$<CONFIG:debug>>:NDEBUG>)
set_target_properties(PhysXNewtonCore PROPERTIES
	CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON)
if(MSVC)
	target_compile_options(PhysXNewtonCore PRIVATE /fp:precise /fp:contract $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
else()
	target_compile_options(PhysXNewtonCore PRIVATE -fno-fast-math -ffp-contract=on $<$<COMPILE_LANGUAGE:CXX>:-fexceptions>)
endif()
