# GKlib licensing

GKlib is distributed under the Apache License, Version 2.0; see `LICENSE.txt`.
The retained source subset also contains files derived from third parties under
the license notices embedded in those files:

- `include/gk_mksort.h` and `include/gkregex.h`: LGPL-2.1-or-later.
- `src/random.c`: BSD-3-Clause Mersenne Twister code when `USE_GKRAND` is enabled.

The Newton ordering build does not define `USE_GKRAND`, but these notices remain
part of the distributed source.
