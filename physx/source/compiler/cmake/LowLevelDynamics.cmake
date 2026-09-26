## SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: Apache-2.0

#
# Build LowLevelDynamics common
#

SET(PHYSX_SOURCE_DIR ${PHYSX_ROOT_DIR}/source)
SET(LL_SOURCE_DIR ${PHYSX_SOURCE_DIR}/lowleveldynamics/src)

# Include here after the directories are defined so that the platform specific file can use the variables.
include(${PHYSX_ROOT_DIR}/${PROJECT_CMAKE_FILES_DIR}/${TARGET_BUILD_PLATFORM}/LowLevelDynamics.cmake)


SET(LLDYNAMICS_BASE_DIR ${PHYSX_ROOT_DIR}/source/lowleveldynamics)

SET(LLDYNAMICS_INCLUDES
	${LLDYNAMICS_BASE_DIR}/include/DyArticulationCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyVArticulation.h
	${LLDYNAMICS_BASE_DIR}/include/DyArticulationTendon.h
	${LLDYNAMICS_BASE_DIR}/include/DyArticulationMimicJointCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableBodyCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableSurface.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableSurfaceCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableVolume.h
	${LLDYNAMICS_BASE_DIR}/include/DyDeformableVolumeCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyFeatherstoneArticulation.h
	${LLDYNAMICS_BASE_DIR}/include/DyFeatherstoneArticulationJointData.h
	${LLDYNAMICS_BASE_DIR}/include/DyFeatherstoneArticulationUtils.h
	${LLDYNAMICS_BASE_DIR}/include/DyConstraint.h
	${LLDYNAMICS_BASE_DIR}/include/DyConstraintWriteBack.h
	${LLDYNAMICS_BASE_DIR}/include/DyContext.h
	${LLDYNAMICS_BASE_DIR}/include/DySleepingConfigulation.h
	${LLDYNAMICS_BASE_DIR}/include/DyThresholdTable.h
	${LLDYNAMICS_BASE_DIR}/include/DyArticulationJointCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyParticleSystemCore.h
	${LLDYNAMICS_BASE_DIR}/include/DyParticleSystem.h
	${LLDYNAMICS_BASE_DIR}/include/DyIslandManager.h
)
SOURCE_GROUP("include" FILES ${LLDYNAMICS_INCLUDES})

SET(LLDYNAMICS_SHARED
	${LLDYNAMICS_BASE_DIR}/shared/DyCpuGpuArticulation.h
	${LLDYNAMICS_BASE_DIR}/shared/DyCpuGpu1dConstraint.h
    ${LLDYNAMICS_BASE_DIR}/shared/DyCpuGpuBiasCoefficient.h
)
SOURCE_GROUP("shared" FILES ${LLDYNAMICS_SHARED})

SET(LLDYNAMICS_SOURCE		
	${LLDYNAMICS_BASE_DIR}/src/DyAllocator.h
	${LLDYNAMICS_BASE_DIR}/src/DyAllocator.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyArticulationContactPrep.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyArticulationMimicJoint.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyFeatherstoneArticulation.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyFeatherstoneForwardDynamic.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyFeatherstoneInverseDynamic.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintPartition.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintSetup.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintSetupBlock.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyContactPrep.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyContactPrep4.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyDynamicsBase.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyDynamics.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyFrictionCorrelation.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyRigidBodyToSolverBody.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraints.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraintsBlock.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverControl.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraint1DStep.h
	${LLDYNAMICS_BASE_DIR}/src/DyThreadContext.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyThresholdTable.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyTGSDynamics.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyTGSContactPrep.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyTGSContactPrepBlock.cpp
	${LLDYNAMICS_BASE_DIR}/src/DyArticulationContactPrep.h
	${LLDYNAMICS_BASE_DIR}/src/DyBodyCoreIntegrator.h
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintPartition.h
	${LLDYNAMICS_BASE_DIR}/src/DyConstraintPrep.h
	${LLDYNAMICS_BASE_DIR}/src/DyContactPrep.h
	${LLDYNAMICS_BASE_DIR}/src/DyContactPrepShared.h
	${LLDYNAMICS_BASE_DIR}/src/DyContactReduction.h
	${LLDYNAMICS_BASE_DIR}/src/DyCorrelationBuffer.h
	${LLDYNAMICS_BASE_DIR}/src/DyDynamicsBase.h
	${LLDYNAMICS_BASE_DIR}/src/DyDynamics.h
	${LLDYNAMICS_BASE_DIR}/src/DyFrictionPatch.h
	${LLDYNAMICS_BASE_DIR}/src/DyFrictionPatchStreamPair.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverBody.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraint1D.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraint1D4.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraintDesc.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraintsShared.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverConstraintTypes.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverContact.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverContact4.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverContext.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverControl.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverCore.h
	${LLDYNAMICS_BASE_DIR}/src/DySolverCore.cpp
	${LLDYNAMICS_BASE_DIR}/src/DySolverExt.h
	${LLDYNAMICS_BASE_DIR}/src/DyThreadContext.h
	${LLDYNAMICS_BASE_DIR}/src/DyTGSDynamics.h
    ${LLDYNAMICS_BASE_DIR}/src/DyTGSContactPrep.h
    ${LLDYNAMICS_BASE_DIR}/src/DyTGS.h
    ${LLDYNAMICS_BASE_DIR}/src/DyPGS.h
	${LLDYNAMICS_BASE_DIR}/src/DySleep.h
	${LLDYNAMICS_BASE_DIR}/src/DySleep.cpp
)
SOURCE_GROUP("src" FILES ${LLDYNAMICS_SOURCE})

ADD_LIBRARY(LowLevelDynamics ${LOWLEVELDYNAMICS_LIBTYPE}
	${LLDYNAMICS_INCLUDES}
	${LLDYNAMICS_SHARED}
	${LLDYNAMICS_SOURCE}
)

GET_TARGET_PROPERTY(PHYSXFOUNDATION_INCLUDES PhysXFoundation INTERFACE_INCLUDE_DIRECTORIES)

TARGET_INCLUDE_DIRECTORIES(LowLevelDynamics 
	PRIVATE ${LOWLEVELDYNAMICS_PLATFORM_INCLUDES}

	PRIVATE ${PHYSXFOUNDATION_INCLUDES}

	PRIVATE ${PHYSX_ROOT_DIR}/include

	PRIVATE ${PHYSX_SOURCE_DIR}/common/src
	
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src/contact
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/include
	
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/api/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/common/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/common/include/pipeline
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/common/include/utils
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/software/include

	PRIVATE ${PHYSX_SOURCE_DIR}/lowleveldynamics/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowleveldynamics/shared
	PRIVATE ${PHYSX_SOURCE_DIR}/lowleveldynamics/src
	
	PRIVATE ${PHYSX_SOURCE_DIR}/physxgpu/include
)

# Use generator expressions to set config specific preprocessor definitions
TARGET_COMPILE_DEFINITIONS(LowLevelDynamics 

	# Common to all configurations
	PRIVATE ${LOWLEVELDYNAMICS_COMPILE_DEFS}
)

SET_TARGET_PROPERTIES(LowLevelDynamics PROPERTIES 
    ARCHIVE_OUTPUT_NAME_DEBUG "LowLevelDynamics_static"
    ARCHIVE_OUTPUT_NAME_CHECKED "LowLevelDynamics_static"
    ARCHIVE_OUTPUT_NAME_PROFILE "LowLevelDynamics_static"
    ARCHIVE_OUTPUT_NAME_RELEASE "LowLevelDynamics_static"
)

IF(LLDYNAMICS_COMPILE_PDB_NAME_DEBUG)
	SET_TARGET_PROPERTIES(LowLevelDynamics PROPERTIES 
		COMPILE_PDB_NAME_DEBUG "${LLDYNAMICS_COMPILE_PDB_NAME_DEBUG}"
		COMPILE_PDB_NAME_CHECKED "${LLDYNAMICS_COMPILE_PDB_NAME_CHECKED}"
		COMPILE_PDB_NAME_PROFILE "${LLDYNAMICS_COMPILE_PDB_NAME_PROFILE}"
		COMPILE_PDB_NAME_RELEASE "${LLDYNAMICS_COMPILE_PDB_NAME_RELEASE}"
	)
ENDIF()

IF(PX_EXPORT_LOWLEVEL_PDB)
	SET_TARGET_PROPERTIES(LowLevelDynamics PROPERTIES 
		COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG "${PHYSX_ROOT_DIR}/${PX_ROOT_LIB_DIR}/debug/"
		COMPILE_PDB_OUTPUT_DIRECTORY_CHECKED "${PHYSX_ROOT_DIR}/${PX_ROOT_LIB_DIR}/checked/"
		COMPILE_PDB_OUTPUT_DIRECTORY_PROFILE "${PHYSX_ROOT_DIR}/${PX_ROOT_LIB_DIR}/profile/"
		COMPILE_PDB_OUTPUT_DIRECTORY_RELEASE "${PHYSX_ROOT_DIR}/${PX_ROOT_LIB_DIR}/release/"
	)
ENDIF()

IF(PX_GENERATE_SOURCE_DISTRO)
	LIST(APPEND SOURCE_DISTRO_FILE_LIST ${LLDYNAMICS_INCLUDES})
	LIST(APPEND SOURCE_DISTRO_FILE_LIST ${LLDYNAMICS_SHARED})
	LIST(APPEND SOURCE_DISTRO_FILE_LIST ${LLDYNAMICS_SOURCE})	
ENDIF()

# enable -fPIC so we can link static libs with the editor
SET_TARGET_PROPERTIES(LowLevelDynamics PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# Anvil is compiled separately so its exceptions and precise FP do not alter PGS/TGS.
SET(ANVIL_SOURCE_DIR ${LLDYNAMICS_BASE_DIR}/src/anvil)
ADD_SUBDIRECTORY(${ANVIL_SOURCE_DIR}/core ${CMAKE_CURRENT_BINARY_DIR}/anvil-core)
SET(ANVIL_NATIVE_SOURCE
	${ANVIL_SOURCE_DIR}/DyAnvilSolver.h
	${ANVIL_SOURCE_DIR}/DyAnvilSolver.cpp
	${ANVIL_SOURCE_DIR}/DyAnvilConstraintPrep.h
	${ANVIL_SOURCE_DIR}/DyAnvilConstraintPrep.cpp
	${ANVIL_SOURCE_DIR}/DyAnvilContactPrep.h
	${ANVIL_SOURCE_DIR}/DyAnvilContactPrep.cpp
)
ADD_LIBRARY(PhysXAnvil OBJECT ${ANVIL_NATIVE_SOURCE})
GET_TARGET_PROPERTY(ANVIL_NATIVE_INCLUDES LowLevelDynamics INCLUDE_DIRECTORIES)
TARGET_INCLUDE_DIRECTORIES(PhysXAnvil PRIVATE ${ANVIL_NATIVE_INCLUDES})
TARGET_COMPILE_DEFINITIONS(PhysXAnvil PRIVATE ${LOWLEVELDYNAMICS_COMPILE_DEFS})
SET_TARGET_PROPERTIES(PhysXAnvil PROPERTIES
	CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
	POSITION_INDEPENDENT_CODE ON)
IF(MSVC)
	TARGET_COMPILE_OPTIONS(PhysXAnvil PRIVATE /fp:precise /EHsc)
ELSEIF(EMSCRIPTEN)
	# Emscripten's JavaScript exception handling routes every potentially throwing call
	# through a JavaScript trampoline. The adapter catches nothing, so disable it there.
	TARGET_COMPILE_OPTIONS(PhysXAnvil PRIVATE -fno-fast-math -ffp-contract=off -fno-exceptions)
ELSE()
	TARGET_COMPILE_OPTIONS(PhysXAnvil PRIVATE -fno-fast-math -ffp-contract=off -fexceptions)
ENDIF()

IF(PX_GENERATE_SOURCE_DISTRO)
	FILE(GLOB_RECURSE ANVIL_CORE_DISTRO_FILES ${ANVIL_SOURCE_DIR}/core/* ${ANVIL_SOURCE_DIR}/vendor/gklib/* ${ANVIL_SOURCE_DIR}/vendor/metis/*)
	LIST(APPEND SOURCE_DISTRO_FILE_LIST ${ANVIL_NATIVE_SOURCE} ${ANVIL_CORE_DISTRO_FILES})
ENDIF()
