#include "kernel_operator.h"
using namespace AscendC;

/* ---------- 标量绝对值 ---------- */
template<typename T>
__aicore__ inline T sabs(T x) { return x < static_cast<T>(0) ? -x : x; }

/* ---------- 双精度模拟辅助函数 ---------- */
template<typename T>
__aicore__ inline void split_fp(T x, T &xh, T &xl) {
    const T SPLIT = static_cast<T>(4097.0f);
    T tmp = SPLIT * x;
    xh = tmp - (tmp - x);
    xl = x - xh;
}

template<typename T>
__aicore__ inline void twoProd(T a, T b, T &ph, T &pl) {
    ph = a * b;
    T ahi, alo, bhi, blo;
    split_fp(a, ahi, alo);
    split_fp(b, bhi, blo);
    pl = ((ahi * bhi - ph) + ahi * blo + alo * bhi) + alo * blo;
}

template<typename T>
__aicore__ inline void twoSum(T a, T b, T &sh, T &sl) {
    sh = a + b;
    T z = sh - a;
    sl = (a - (sh - z)) + (b - z);
}

/* ==================================================================== */
template<typename T, typename I32>
class KernelSolveEx {
public:
    __aicore__ inline KernelSolveEx() {}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR R, GM_ADDR info,
                                int64_t /*m*/, int64_t n, int64_t k,
                                int64_t batch, bool /*left*/, bool /*chk*/)
    {
        n_   = (int)n;
        rhs_ = (int)(k == 0 ? 1 : k);
        bat_ = (int)batch;
        printf("[KERNEL] n=%d rhs=%d batch=%d\n", n_, rhs_, bat_);

        int64_t AN = batch * n * n;
        int64_t BN = batch * n * rhs_;
        A_gm   .SetGlobalBuffer((__gm__ T  *)A, AN);
        B_gm   .SetGlobalBuffer((__gm__ T  *)B, BN);
        R_gm   .SetGlobalBuffer((__gm__ T  *)R, BN);
        info_gm.SetGlobalBuffer((__gm__ I32*)info, 1);
    }

    __aicore__ inline void Process() {
        int err = 0;
        for(int b = 0; b < bat_ && err == 0; ++b) {
            err = (rhs_ == 1 ? SolveVec(b) : SolveMat(b));
        }
        info_gm.SetValue(0, err);
    }

private:
    /* ---------------- rhs == 1：向量解 ---------------- */
    __aicore__ inline int SolveVec(int b) {
        const int n = n_;
        const int64_t Aoff = int64_t(b)*n*n;
        const int64_t Boff = int64_t(b)*n;
        const T eps = static_cast<T>(1e-12);

        // 行均衡  
        T rowmax[256];
        for(int i=0;i<n;++i){
            T m=0;
            int64_t ba = Aoff + int64_t(i)*n;
            for(int j=0;j<n;++j){
                T v = sabs(A_gm.GetValue(ba+j));
                if(v>m) m=v;
            }
            rowmax[i] = (m<eps ? static_cast<T>(1) : m);
        }
        for(int i=0;i<n;++i){
            T inv = static_cast<T>(1)/rowmax[i];
            int64_t ba = Aoff + int64_t(i)*n;
            for(int j=0;j<n;++j)
                A_gm.SetValue(ba+j, A_gm.GetValue(ba+j)*inv);
            B_gm.SetValue(Boff+i, B_gm.GetValue(Boff+i)*inv);
        }

        // LU 分解 + 主元缩放 + 消元（补偿）
        for(int col=0; col<n; ++col) {
            int piv = col;
            T maxa = sabs(A_gm.GetValue(Aoff+col*n+col));
            for(int r=col+1;r<n;++r){
                T v = sabs(A_gm.GetValue(Aoff+r*n+col));
                if(v>maxa){ maxa=v; piv=r; }
            }
            if(maxa<eps) return col+1;
            if(piv!=col){
                SwapRow(A_gm,Aoff,col,piv,n);
                SwapRow(B_gm,Boff,col,piv,1);
            }
            // 缩放主元行
            {
                T pv = A_gm.GetValue(Aoff+col*n+col);
                T pinv = static_cast<T>(1)/pv;
                A_gm.SetValue(Aoff+col*n+col, static_cast<T>(1));
                for(int c=col+1;c<n;++c)
                    A_gm.SetValue(Aoff+col*n+c,
                        A_gm.GetValue(Aoff+col*n+c)*pinv);
                B_gm.SetValue(Boff+col,
                    B_gm.GetValue(Boff+col)*pinv);
            }
            // 消元（补偿）
            for(int r=col+1;r<n;++r){
                T L = A_gm.GetValue(Aoff+r*n+col);
                // 更新 A
                for(int c=col+1;c<n;++c){
                    T a_rc = A_gm.GetValue(Aoff+r*n+c);
                    T up   = A_gm.GetValue(Aoff+col*n+c);
                    T ph,pl,sh,sl;
                    twoProd(L, up, ph, pl);
                    twoSum(a_rc, -ph, sh, sl);
                    A_gm.SetValue(Aoff+r*n+c, sh + (sl - pl));
                }
                // 更新 B
                {
                    T br = B_gm.GetValue(Boff+r);
                    T bc = B_gm.GetValue(Boff+col);
                    T ph,pl,sh,sl;
                    twoProd(L, bc, ph, pl);
                    twoSum(br, -ph, sh, sl);
                    B_gm.SetValue(Boff+r, sh + (sl - pl));
                }
            }
        }

        // 回代（补偿累加）
        for(int rr = n-1; rr>=0; --rr){
            T sumh = B_gm.GetValue(Boff+rr), suml = 0;
            for(int j=rr+1;j<n;++j){
                T u = A_gm.GetValue(Aoff+rr*n+j);
                T x = B_gm.GetValue(Boff+j);
                T ph,pl,sh,sl;
                twoProd(u, x, ph, pl);
                twoSum(sumh, -ph, sh, sl);
                suml += (sl - pl);
                twoSum(sh, suml, sumh, suml);
            }
            B_gm.SetValue(Boff+rr, sumh);
        }

        // 写回 R
        for(int i=0;i<n;++i)
            R_gm.SetValue(Boff+i, B_gm.GetValue(Boff+i));
        return 0;
    }

    /* ---------------- rhs > 1：矩阵解 ---------------- */
    __aicore__ inline int SolveMat(int b) {
        const int n = n_;
        const int64_t Aoff = int64_t(b)*n*n;
        const int64_t Boff = int64_t(b)*n*rhs_;
        const T eps = static_cast<T>(1e-12);

        // 行均衡
        T rowmax[256];
        for(int i=0;i<n;++i){
            T m=0; int64_t ba=Aoff+int64_t(i)*n;
            for(int j=0;j<n;++j){
                T v=sabs(A_gm.GetValue(ba+j));
                if(v>m) m=v;
            }
            rowmax[i] = (m<eps ? static_cast<T>(1) : m);
        }
        for(int i=0;i<n;++i){
            T inv=static_cast<T>(1)/rowmax[i];
            int64_t ba=Aoff+int64_t(i)*n;
            for(int j=0;j<n;++j)
                A_gm.SetValue(ba+j,
                    A_gm.GetValue(ba+j)*inv);
            int64_t bb=Boff+int64_t(i)*rhs_;
            for(int c=0;c<rhs_;++c)
                B_gm.SetValue(bb+c,
                    B_gm.GetValue(bb+c)*inv);
        }

        // LU + scale + eliminate（补偿）
        for(int col=0;col<n;++col){
            int piv=col;
            T maxa=sabs(A_gm.GetValue(Aoff+col*n+col));
            for(int r=col+1;r<n;++r){
                T v=sabs(A_gm.GetValue(Aoff+r*n+col));
                if(v>maxa){ maxa=v; piv=r; }
            }
            if(maxa<eps) return col+1;
            if(piv!=col){
                SwapRow(A_gm,Aoff,col,piv,n);
                SwapRow(B_gm,Boff,col,piv,rhs_);
            }
            // 缩放主元行
            {
                T pv=A_gm.GetValue(Aoff+col*n+col);
                T pinv=static_cast<T>(1)/pv;
                A_gm.SetValue(Aoff+col*n+col, static_cast<T>(1));
                for(int c=col+1;c<n;++c)
                    A_gm.SetValue(Aoff+col*n+c,
                        A_gm.GetValue(Aoff+col*n+c)*pinv);
                int64_t pb=Boff+int64_t(col)*rhs_;
                for(int c=0;c<rhs_;++c)
                    B_gm.SetValue(pb+c,
                        B_gm.GetValue(pb+c)*pinv);
            }
            // 消元
            for(int r=col+1;r<n;++r){
                T L=A_gm.GetValue(Aoff+r*n+col);
                // 更新 A
                for(int c=col+1;c<n;++c){
                    T a_rc=A_gm.GetValue(Aoff+r*n+c);
                    T up  =A_gm.GetValue(Aoff+col*n+c);
                    T ph,pl,sh,sl;
                    twoProd(L, up, ph, pl);
                    twoSum(a_rc, -ph, sh, sl);
                    A_gm.SetValue(Aoff+r*n+c, sh + (sl - pl));
                }
                // 更新 B
                int64_t rb=Boff+int64_t(r)*rhs_;
                int64_t pb=Boff+int64_t(col)*rhs_;
                for(int c=0;c<rhs_;++c){
                    T br=B_gm.GetValue(rb+c);
                    T bc=B_gm.GetValue(pb+c);
                    T ph,pl,sh,sl;
                    twoProd(L, bc, ph, pl);
                    twoSum(br, -ph, sh, sl);
                    B_gm.SetValue(rb+c, sh + (sl - pl));
                }
            }
        }

        // 回代
        for(int r=n-1;r>=0;--r){
            for(int c=0;c<rhs_;++c){
                T sumh=B_gm.GetValue(Boff+int64_t(r)*rhs_+c), suml=0;
                for(int j=r+1;j<n;++j){
                    T u=A_gm.GetValue(Aoff+r*n+j);
                    T x=B_gm.GetValue(Boff+int64_t(j)*rhs_+c);
                    T ph,pl,sh,sl;
                    twoProd(u, x, ph, pl);
                    twoSum(sumh, -ph, sh, sl);
                    suml += (sl - pl);
                    twoSum(sh, suml, sumh, suml);
                }
                B_gm.SetValue(Boff+int64_t(r)*rhs_+c, sumh);
            }
        }

        // 拷回结果
        for(int idx=0;idx<n*rhs_;++idx)
            R_gm.SetValue(Boff+idx, B_gm.GetValue(Boff+idx));
        return 0;
    }

    /* ---------- 行交换 ---------- */
    __aicore__ inline void SwapRow(GlobalTensor<T>& M,
                                   int64_t base,
                                   int r1,int r2,int width)
    {
        for(int j=0;j<width;++j){
            int64_t i1 = base + int64_t(r1)*width + j;
            int64_t i2 = base + int64_t(r2)*width + j;
            T tmp = M.GetValue(i1);
            M.SetValue(i1, M.GetValue(i2));
            M.SetValue(i2, tmp);
        }
    }

    GlobalTensor<T>   A_gm, B_gm, R_gm;
    GlobalTensor<I32> info_gm;
    int n_{0}, rhs_{1}, bat_{1};
};

/* ==================================================================== */
extern "C" __global__ __aicore__
void solve_ex(GM_ADDR a, GM_ADDR b, GM_ADDR r, GM_ADDR info,
              GM_ADDR /*workspace*/, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    KernelSolveEx<float,int32_t> op;
    op.Init(a, b, r, info,
            t.m, t.n, t.k,
            t.batch, t.left, t.check_errors);
    op.Process();
}
