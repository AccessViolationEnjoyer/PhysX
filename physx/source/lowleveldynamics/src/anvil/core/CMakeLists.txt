# Private numerical core shared by native PhysX and standalone validation builds.
# Dependencies and precise arithmetic options must not propagate to PGS/TGS.
option(PX_ANVIL_USE_AVX2 "Use AVX2 and FMA in the Anvil solver core" OFF)
set(ANVIL_GKLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/gklib")
set(ANVIL_METIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/metis")
set(ANVIL_GKLIB_SOURCES
	${ANVIL_GKLIB_DIR}/src/blas.c
	${ANVIL_GKLIB_DIR}/src/error.c
	${ANVIL_GKLIB_DIR}/src/mcore.c
	${ANVIL_GKLIB_DIR}/src/memory.c
	${ANVIL_GKLIB_DIR}/src/random.c
	${ANVIL_GKLIB_DIR}/src/sort.c
	${ANVIL_GKLIB_DIR}/src/timers.c
)
set(ANVIL_METIS_SOURCES
	${ANVIL_METIS_DIR}/libmetis/auxapi.c
	${ANVIL_METIS_DIR}/libmetis/balance.c
	${ANVIL_METIS_DIR}/libmetis/bucketsort.c
	${ANVIL_METIS_DIR}/libmetis/checkgraph.c
	${ANVIL_METIS_DIR}/libmetis/coarsen.c
	${ANVIL_METIS_DIR}/libmetis/compress.c
	${ANVIL_METIS_DIR}/libmetis/contig.c
	${ANVIL_METIS_DIR}/libmetis/debug.c
	${ANVIL_METIS_DIR}/libmetis/fortran.c
	${ANVIL_METIS_DIR}/libmetis/fm.c
	${ANVIL_METIS_DIR}/libmetis/gklib.c
	${ANVIL_METIS_DIR}/libmetis/graph.c
	${ANVIL_METIS_DIR}/libmetis/initpart.c
	${ANVIL_METIS_DIR}/libmetis/kwayfm.c
	${ANVIL_METIS_DIR}/libmetis/kwayrefine.c
	${ANVIL_METIS_DIR}/libmetis/mcutil.c
	${ANVIL_METIS_DIR}/libmetis/minconn.c
	${ANVIL_METIS_DIR}/libmetis/mincover.c
	${ANVIL_METIS_DIR}/libmetis/mmd.c
	${ANVIL_METIS_DIR}/libmetis/ometis.c
	${ANVIL_METIS_DIR}/libmetis/options.c
	${ANVIL_METIS_DIR}/libmetis/pmetis.c
	${ANVIL_METIS_DIR}/libmetis/refine.c
	${ANVIL_METIS_DIR}/libmetis/separator.c
	${ANVIL_METIS_DIR}/libmetis/sfm.c
	${ANVIL_METIS_DIR}/libmetis/srefine.c
	${ANVIL_METIS_DIR}/libmetis/stat.c
	${ANVIL_METIS_DIR}/libmetis/timing.c
	${ANVIL_METIS_DIR}/libmetis/util.c
	${ANVIL_METIS_DIR}/libmetis/wspace.c
)
add_library(PhysXAnvilCore OBJECT
	AnvilSolver.cpp AnvilSolver.h AnvilMath.h AnvilParallel.h AnvilStorage.h AnvilMetis.h
	StorageCholesky.h BlockCholesky.h IncrementalCholesky.h
	${ANVIL_GKLIB_SOURCES} ${ANVIL_METIS_SOURCES})
target_include_directories(PhysXAnvilCore PRIVATE
	"${ANVIL_GKLIB_DIR}/include" "${ANVIL_METIS_DIR}/include" "${ANVIL_METIS_DIR}/libmetis")
set_source_files_properties(${ANVIL_GKLIB_SOURCES} PROPERTIES
	COMPILE_DEFINITIONS "NDEBUG;NDEBUG2")
set_source_files_properties(${ANVIL_METIS_SOURCES} PROPERTIES
	COMPILE_DEFINITIONS "METIS_NodeND=Anvil_METIS_NodeND;METIS_SetDefaultOptions=Anvil_METIS_SetDefaultOptions")
if(MSVC)
	set_property(SOURCE ${ANVIL_GKLIB_SOURCES} ${ANVIL_METIS_SOURCES} APPEND PROPERTY
		COMPILE_DEFINITIONS "__thread=__declspec(thread)")
	set_source_files_properties(${ANVIL_GKLIB_SOURCES} ${ANVIL_METIS_SOURCES} PROPERTIES COMPILE_OPTIONS /W0)
else()
	set_source_files_properties(${ANVIL_GKLIB_SOURCES} ${ANVIL_METIS_SOURCES} PROPERTIES COMPILE_OPTIONS -w)
endif()
target_compile_definitions(PhysXAnvilCore PRIVATE $<$<NOT:$<CONFIG:debug>>:NDEBUG>)
set_target_properties(PhysXAnvilCore PROPERTIES
	CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON)
if(MSVC)
	target_compile_options(PhysXAnvilCore PRIVATE /fp:precise /fp:contract $<$<COMPILE_LANGUAGE:CXX>:/EHs-c->)
else()
	target_compile_options(PhysXAnvilCore PRIVATE -fno-fast-math -ffp-contract=on $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)
endif()
if(EMSCRIPTEN)
	# WebAssembly uses 128-bit SIMD kernels. Relaxed SIMD fuses multiply-adds where the
	# host supports it, which changes rounding between hosts, so it is opt-in.
	option(PX_ANVIL_USE_WASM_RELAXED_SIMD "Use WebAssembly relaxed SIMD in the Anvil solver core" OFF)
	target_compile_options(PhysXAnvilCore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-msimd128>)
	target_sources(PhysXAnvilCore PRIVATE AnvilGklibStubs.c)
	if(PX_ANVIL_USE_WASM_RELAXED_SIMD)
		target_compile_options(PhysXAnvilCore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mrelaxed-simd>)
	endif()
elseif(PX_ANVIL_USE_AVX2)
	if(MSVC)
		target_compile_options(PhysXAnvilCore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|amd64|x86_64|i.86")
		target_compile_options(PhysXAnvilCore PRIVATE
			$<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
	endif()
endif()
