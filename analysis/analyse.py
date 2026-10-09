import numpy as np
N = ["chamfer","through_hole","tri_passage","rect_passage","6s_passage","tri_thr_slot","rect_thr_slot","circ_thr_slot",
"rect_thr_step","2s_thr_step","slant_thr_step","Oring","blind_hole","tri_pocket","rect_pocket","6s_pocket","circ_end_pocket",
"rect_blind_slot","v_circ_blind_slot","h_circ_blind_slot","tri_blind_step","circ_blind_step","rect_blind_step","round","stock"]
d = np.load("results/mfcadpp_full/0930_000433/confusion/test_preds.npz")
P, Y, G = d["pred"], d["y"], d["graph"]
M = np.zeros((25,25), int); np.add.at(M, (Y,P), 1)
np.save("results/mfcadpp_full/0930_000433/confusion/cm.npy", M)
err = M.sum()-np.trace(M); print("total errors", err, "of", M.sum())
off = M.copy(); np.fill_diagonal(off,0)
print("\n== per class: support recall precision | missed-as (true=c) | false-from (pred=c)")
for c in range(25):
    sup=M[c].sum(); rec=M[c,c]/sup; prec=M[c,c]/M[:,c].sum()
    fn=off[c]; fp=off[:,c]
    a=", ".join(f"{N[j]} {fn[j]}" for j in np.argsort(-fn)[:4] if fn[j])
    b=", ".join(f"{N[j]} {fp[j]}" for j in np.argsort(-fp)[:4] if fp[j])
    print(f"{c:2d} {N[c]:18s} n={sup:6d} R={rec:.3f} P={prec:.3f} FN={fn.sum():4d} FP={fp.sum():4d}\n     true {N[c]} predicted as: {a}\n     predicted {N[c]} but truly: {b}")
print("\n== top symmetric pairs (a<->b), share of all errors")
S = off+off.T; pairs=[(S[i,j],i,j) for i in range(25) for j in range(i+1,25)]
cum=0
for s,i,j in sorted(pairs,reverse=True)[:25]:
    cum+=s; print(f"{s:5d} {s/err:5.1%} cum {cum/err:5.1%}  {N[i]} <-> {N[j]}   ({N[i]}->{N[j]} {off[i,j]}, {N[j]}->{N[i]} {off[j,i]})")
print("\nerrors involving stock:", S[24].sum(), f"{S[24].sum()/err:.1%}")
# model-level
wrong = P!=Y
ng = G.max()+1
epm = np.bincount(G[wrong], minlength=ng); fpm=np.bincount(G,minlength=ng)
print("\nmodels with >=1 error:", (epm>0).sum(), "of", ng, f"{(epm>0).mean():.1%}")
print("errors per bad model: median", np.median(epm[epm>0]), "mean", epm[epm>0].mean())
srt=np.sort(epm)[::-1]; print("share of errors in worst 1% models:", srt[:ng//100].sum()/err, " worst 5%:", srt[:ng//20].sum()/err)
print("faces/model: bad", fpm[epm>0].mean(), "clean", fpm[epm==0].mean())
# number of distinct features (classes) per model vs error rate
nc = np.array([len(set(Y[G==g])) for g in range(ng)]) if False else None
