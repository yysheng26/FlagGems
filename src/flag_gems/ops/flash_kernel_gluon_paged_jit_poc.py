"""Paged gluon kernel handle for libtriton_jit C++ POC.

Re-exports the raw GluonJITFunction under its compile name so libtriton_jit can
load metadata (fn.__name__) and static signature from the same symbol.
"""
from flag_gems.ops.flash_kernel_gluon import flash_paged_fwd_gluon_kernel

flash_paged_fwd_gluon_kernel = flash_paged_fwd_gluon_kernel.fn