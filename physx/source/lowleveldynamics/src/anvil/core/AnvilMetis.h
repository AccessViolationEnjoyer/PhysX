#ifndef ANVIL_METIS_H
#define ANVIL_METIS_H

#include <cstdint>
#define _GKLIB_H_
#define METIS_NodeND Anvil_METIS_NodeND
#define METIS_SetDefaultOptions Anvil_METIS_SetDefaultOptions
#include "../vendor/metis/include/metis.h"
#undef METIS_SetDefaultOptions
#undef METIS_NodeND
#undef _GKLIB_H_

#endif
