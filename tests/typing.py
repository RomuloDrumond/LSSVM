"""Shared type aliases for the test suite."""

from typing import TYPE_CHECKING, Any

from lssvm.LSSVC import LSSVC

if TYPE_CHECKING:
    from lssvm.LSSVC_GPU import LSSVC_GPU

    AnyLSSVC = LSSVC | LSSVC_GPU
    AnyLSSVCClass = type[LSSVC] | type[LSSVC_GPU]
else:
    # At runtime, we can't be sure LSSVC_GPU is available, so we use Any.
    try:
        from lssvm.LSSVC_GPU import LSSVC_GPU

        AnyLSSVC = LSSVC | LSSVC_GPU
        AnyLSSVCClass = type[LSSVC] | type[LSSVC_GPU]
    except ImportError:
        AnyLSSVC = Any
        AnyLSSVCClass = Any
