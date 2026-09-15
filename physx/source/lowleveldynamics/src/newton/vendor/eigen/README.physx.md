# Eigen dependency for the private Newton solver

Pinned version: Eigen 3.4.0.
Upstream source: https://gitlab.com/libeigen/eigen/-/tree/3.4.0

This is the unmodified `Eigen/` header tree and upstream license files from the
verified benchmark dependency. Tests, examples, documentation and unsupported
modules are omitted. `SHA256SUMS` records each retained upstream file.

The Newton build uses these headers privately. `core/PrepareStorageKernels.py`
adapts the pinned ordering and Cholesky interfaces to reusable storage without
changing their arithmetic, and writes its generated header into the build tree.
Neither the PhysX SDK nor its installed public headers expose Eigen.

Keep the upstream copyright and license notices. See `COPYING.README` and
`COPYING.MPL2`; individual files retain their applicable license notices.
