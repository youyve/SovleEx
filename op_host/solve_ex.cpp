
#include "solve_ex_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto A_shape   = context->GetInputShape(0)->GetStorageShape(); // [..., m, n]
    auto B_shape   = context->GetInputShape(1)->GetStorageShape(); // [..., n, k]
    int64_t dnA = A_shape.GetDimNum();
    int64_t dnB = B_shape.GetDimNum();

    int64_t m    = A_shape.GetDim(dnA - 2);
    int64_t n    = A_shape.GetDim(dnA - 1);
    int64_t k;
    if (dnB == dnA) {
        k = B_shape.GetDim(dnB - 1);
    } else {
        k = 1;
    }
    int64_t batch  = 1;
    for (int64_t i = 0; i < dnA - 2; ++i) {
        batch *= A_shape.GetDim(i);
    }

    auto attrs     = context->GetAttrs();
    bool left      = *attrs->GetAttrPointer<bool>(0); // "left"
    bool chk_err   = *attrs->GetAttrPointer<bool>(1); // "check_errors"

    SolveExTilingData tiling;
    tiling.set_m(m);
    tiling.set_n(n);
    tiling.set_k(k);
    tiling.set_batch(batch);
    tiling.set_left(left);
    tiling.set_check_errors(chk_err);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t k_rhs  = k == 0 ? 1 : k;
    // 需要存 batch*n*n + batch*n*k_rhs 个 float
    size_t floats = batch * n * n + 3 * batch * n * k_rhs;
    // 每个 float 4 字节
    size_t workspace_size = floats * sizeof(float);
    // 把它写回到 framework
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspace_size + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* B_shape = context->GetInputShape(1);
    *context->GetOutputShape(0)    = *B_shape;              // result
    context->GetOutputShape(1)->SetDim(0, 1);               // info scalar
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SolveEx : public OpDef {
public:
    explicit SolveEx(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("info")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("left").AttrType(OPTIONAL).Bool(true);
        this->Attr("check_errors").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(SolveEx);
}
