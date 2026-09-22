# Eigen dependency for standalone Newton validation

Pinned version: Eigen 3.4.0.
Upstream source: https://gitlab.com/libeigen/eigen/-/tree/3.4.0

This is the unmodified `Eigen/` header tree and upstream license files from the
verified benchmark dependency. Tests, examples, documentation and unsupported
modules are omitted. `SHA256SUMS` records each retained upstream file.

Some standalone tests use these headers as an independent numerical oracle.
The production Newton core and native PhysX integration do not include, link or
package Eigen.

Keep the upstream copyright and license notices. See `COPYING.README` and
`COPYING.MPL2`; individual files retain their applicable license notices.
