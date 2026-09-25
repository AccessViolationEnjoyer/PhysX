# Private numerical core shared by native PhysX and standalone validation builds.
# Dependencies and precise arithmetic options must not propagate to PGS/TGS.
option(PX_NEWTON_USE_AVX2 "Use AVX2 and FMA in the Newton numerical core" OFF)
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
add_library(PhysXNewtonCore OBJECT
	NewtonSolver.cpp NewtonSolver.h NewtonMath.h NewtonParallel.h NewtonStorage.h NewtonMetis.h
	StorageCholesky.h BlockCholesky.h IncrementalCholesky.h
	${NEWTON_GKLIB_SOURCES} ${NEWTON_METIS_SOURCES})
target_include_directories(PhysXNewtonCore PRIVATE
	"${NEWTON_GKLIB_DIR}/include" "${NEWTON_METIS_DIR}/include" "${NEWTON_METIS_DIR}/libmetis")
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
target_compile_definitions(PhysXNewtonCore PRIVATE $<$<NOT:$<CONFIG:debug>>:NDEBUG>)
set_target_properties(PhysXNewtonCore PROPERTIES
	CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON)
if(MSVC)
	target_compile_options(PhysXNewtonCore PRIVATE /fp:precise /fp:contract $<$<COMPILE_LANGUAGE:CXX>:/EHs-c->)
else()
	target_compile_options(PhysXNewtonCore PRIVATE -fno-fast-math -ffp-contract=on $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)
endif()
if(EMSCRIPTEN)
	# WebAssembly uses 128-bit SIMD kernels. Relaxed SIMD fuses multiply-adds where the
	# host supports it, which changes rounding between hosts, so it is opt-in.
	option(PX_NEWTON_USE_WASM_RELAXED_SIMD "Use WebAssembly relaxed SIMD in the Newton numerical core" OFF)
	target_compile_options(PhysXNewtonCore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-msimd128>)
	target_sources(PhysXNewtonCore PRIVATE NewtonGklibStubs.c)
	if(PX_NEWTON_USE_WASM_RELAXED_SIMD)
		target_compile_options(PhysXNewtonCore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mrelaxed-simd>)
	endif()
elseif(PX_NEWTON_USE_AVX2)
	if(MSVC)
		target_compile_options(PhysXNewtonCore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|amd64|x86_64|i.86")
		target_compile_options(PhysXNewtonCore PRIVATE
			$<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
	endif()
endif()
