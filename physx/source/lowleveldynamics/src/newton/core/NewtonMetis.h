#ifndef NEWTON_METIS_H
#define NEWTON_METIS_H

#include <cstdint>
#define _GKLIB_H_
#define METIS_NodeND Newton_METIS_NodeND
#define METIS_SetDefaultOptions Newton_METIS_SetDefaultOptions
#include "../vendor/metis/include/metis.h"
#undef METIS_SetDefaultOptions
#undef METIS_NodeND
#undef _GKLIB_H_

#endif
