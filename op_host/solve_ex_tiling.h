
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SolveExTilingData)
TILING_DATA_FIELD_DEF(int64_t, m);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(bool, left);
TILING_DATA_FIELD_DEF(bool, check_errors);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SolveEx, SolveExTilingData)
}
