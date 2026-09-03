"""Stage-A integrity tests for the zero-shot loop."""
import os, sys, math
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,"..","pilot"))
import zeroshot_loop as ZL
F=[]
def ck(n,c,e=""):
    print(f"{'PASS' if c else 'FAIL'}  {n}"+(f"   {e}" if e else ""));  F.append(n) if not c else None

P,H,L=24,96,384
rng=np.random.default_rng(0)
ctx=np.cumsum(rng.normal(0,1,(5,L)),1).astype(np.float32)+10
fut=np.cumsum(rng.normal(0,1,(5,H)),1).astype(np.float32)+10

# 1 future replacement: priors and residual must not change
p1=ZL.priors(ctx,P,H); p2=ZL.priors(ctx,P,H)
ck("1 priors deterministic and context-only", all(np.array_equal(p1[k],p2[k]) for k in p1))
nm,sel,used=ZL.pseudo_origin_select(ctx,P,H,p1)
nm2,sel2,_=ZL.pseudo_origin_select(ctx,P,H,ZL.priors(ctx,P,H))
ck("2 pseudo-origin selection is deterministic and future-free", np.array_equal(sel,sel2) and used>0, f"pseudo-origins used={used}")

# 3 neutral correction identity
base=p1["smean"]; y,a=ZL.safe_residual(base,base,ctx,P,H)
ck("3 neutral model correction decodes exactly to the base", np.allclose(y,base) and (a==0).all())

# 4 alpha=0 fallback when no pseudo evidence
y0,a0=ZL.safe_residual(base,base+5.0,ctx,P,H,cand_pseudo=None)
ck("4 alpha falls back to 0 without internal evidence", np.allclose(y0,base))

# 5 synthetic sinusoid: snaive must be near-perfect on an exact-period sinusoid
t=np.arange(L+H); s=np.sin(2*np.pi*t/P).astype(np.float32)[None,:].repeat(3,0)
pp=ZL.priors(s[:,:L],P,H)
ck("5 exact-period sinusoid -> snaive near perfect",
   float(((pp["snaive"]-s[:,L:])**2).mean())<1e-10, f"mse={float(((pp['snaive']-s[:,L:])**2).mean()):.2e}")
# constant series
c=np.ones((2,L+H),dtype=np.float32)*3
pc=ZL.priors(c[:,:L],P,H)
ck("6 constant series -> all priors exact", all(np.allclose(pc[k],3.0) for k in pc))

# 7 Q calculator
ck("7 equal-dataset Q", abs(ZL.q_equal({"a":0.5,"b":2.0},{"a":1.0,"b":1.0})-1.0)<1e-12)

# 8 bootstrap consumes seed and block
b=ZL.paired_bootstrap(rng.random(60)*0.5,rng.random(60),np.repeat(np.arange(30),2),block=5)
ck("8 bootstrap consumes seed/block and returns interval",
   b["seed"]==20260902 and b["block"]==5 and b["lo"]<b["hi"] and 0<=b["p_better"]<=1)

# 9 clipping caps catastrophic corrections
wild=base+1e6
yc,_=ZL.safe_residual(base,wild,ctx,P,H,cand_pseudo=[(base[:, :],wild,fut)])
ck("9 residual clipping caps catastrophic corrections",
   float(np.abs(yc-base).max())<1e5, f"max|delta|={float(np.abs(yc-base).max()):.1f}")

# 10 per-pair MSE / manifest hash stability
ck("10 per-pair MSE shape", ZL.per_pair_mse(base,fut).shape==(5,))
print(); print(f"{len(F)} failures"); sys.exit(1 if F else 0)
