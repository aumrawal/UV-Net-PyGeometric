import sys, numpy as np, torch, collections
sys.path.insert(0, ".")
from datasets.mfcadpp import MFCADPPDataset
T="results/mfcadpp_full/0930_000433/confusion/"
N = ["chamfer","through_hole","tri_passage","rect_passage","6s_passage","tri_thr_slot","rect_thr_slot","circ_thr_slot",
"rect_thr_step","2s_thr_step","slant_thr_step","Oring","blind_hole","tri_pocket","rect_pocket","6s_pocket","circ_end_pocket",
"rect_blind_slot","v_circ_blind_slot","h_circ_blind_slot","tri_blind_step","circ_blind_step","rect_blind_step","round","stock"]
d=np.load(T+"test_preds.npz"); P,Y,G,C=d["pred"],d["y"],d["graph"],d["conf"]
M=np.load(T+"cm.npy"); off=M.copy(); np.fill_diagonal(off,0); S=off+off.T; err=off.sum()
def tot(pairs): return sum(S[a,b] for a,b in pairs)
cat = collections.OrderedDict()
cat["A through<->blind partner (same profile, floor or no floor)"]=[(1,12),(2,13),(3,14),(4,15),(6,17),(8,22),(7,18),(7,19)]
cat["B open<->closed, same depth (passage/slot/step; pocket/blind slot/blind step)"]=[(3,6),(6,8),(3,8),(2,5),(14,17),(17,22),(14,22),(16,18),(16,19),(18,19),(1,7)]
cat["C profile shape, same type (tri/rect/6-sided; 2-sided/slanted/rect step)"]=[(2,3),(2,4),(3,4),(13,14),(13,15),(14,15),(8,9),(8,10),(9,10),(5,6),(20,21),(20,22),(21,22)]
cat["D round-profile blind family (O-ring / blind hole / circular end pocket)"]=[(11,12),(12,16),(11,16)]
used=set()
for k,v in cat.items():
    print(f"{tot(v):5d} {tot(v)/err:5.1%}  {k}"); used|=set(v)
st=sum(S[24,j] for j in range(24)); print(f"{st:5d} {st/err:5.1%}  E stock <-> any feature")
cr=sum(S[i,j] for i in (0,23) for j in range(24) if j not in (0,23))+S[0,23]; print(f"{cr:5d} {cr/err:5.1%}  F chamfer/round <-> anything but stock")
rest=sum(S[i,j] for i in range(25) for j in range(i+1,25) if (i,j) not in used and 24 not in (i,j) and i not in (0,23) and j not in (0,23))
print(f"{rest:5d} {rest/err:5.1%}  G other (differ in two or more attributes)")
# through-vs-blind status of each class
thr={1,2,3,4,5,6,7,8,9,10}; bl={11,12,13,14,15,16,17,18,19,20,21,22}
x=sum(off[i,j] for i in thr for j in bl)+sum(off[i,j] for i in bl for j in thr)
print(f"\nerrors that cross the through/blind line at all: {x} {x/err:.1%};  stay inside through: {sum(off[i,j] for i in thr for j in thr)}; inside blind: {sum(off[i,j] for i in bl for j in bl)}")
print("confidence: wrong mean %.3f median %.3f | right mean %.3f ; wrong with conf>0.9: %.1f%%"%(C[P!=Y].mean(),np.median(C[P!=Y]),C[P==Y].mean(),100*(C[P!=Y]>0.9).mean()))

# graph-based diagnostics
ds=MFCADPPDataset("data/processed","test")
starts=np.r_[0,np.cumsum(np.bincount(G))]
comp_stats=collections.defaultdict(lambda: collections.defaultdict(lambda:[0,0,0]))  # class -> size -> [faces, wrong, comps]
nb_pred_is_nb_true=0; nb_same_true_correct=0; whole=0; partial=0; nwrong=0
flip_frac=[]
for g in range(len(ds)):
    dt=ds[g]; y=dt.y.numpy(); n=len(y); p=P[starts[g]:starts[g]+n]; assert (Y[starts[g]:starts[g]+n]==y).all()
    ei=dt.edge_index.numpy(); adj=[[] for _ in range(n)]
    for a,b in ei.T: adj[a].append(b)
    comp=-np.ones(n,int); k=0
    for s in range(n):
        if comp[s]>=0: continue
        st=[s]; comp[s]=k
        while st:
            u=st.pop()
            for v in adj[u]:
                if comp[v]<0 and y[v]==y[s]: comp[v]=k; st.append(v)
        k+=1
    for c in range(k):
        idx=np.where(comp==c)[0]; cl=y[idx[0]]; w=(p[idx]!=cl).sum()
        r=comp_stats[cl][len(idx)]; r[0]+=len(idx); r[1]+=w; r[2]+=1
        if w and cl!=24:
            flip_frac.append(w/len(idx))
            if w==len(idx): whole+=w
            else: partial+=w
    for i in np.where(p!=y)[0]:
        nwrong+=1
        if any(y[j]==p[i] for j in adj[i]): nb_pred_is_nb_true+=1
        if any(y[j]==y[i] and p[j]==y[j] for j in adj[i]): nb_same_true_correct+=1
print(f"\nwrong faces: {nwrong}")
print(f" predicted class is the TRUE class of an adjacent face: {nb_pred_is_nb_true} {nb_pred_is_nb_true/nwrong:.1%}")
print(f" has an adjacent face of the same true class that IS predicted correctly: {nb_same_true_correct} {nb_same_true_correct/nwrong:.1%}")
print(f" (feature faces only) errors in fully-flipped connected feature: {whole}; in partially-flipped feature: {partial}")
print("\nerror rate by size of connected same-label face group (proxy for one feature instance)")
for cl in (3,6,8,11,12,14,17,22,2,5,16,23,1,9,10,20):
    row=comp_stats[cl]; tf=sum(v[0] for v in row.values()); tw=sum(v[1] for v in row.values())
    s=" ".join(f"[{k}f: {v[2]} grp, err {v[1]/v[0]:.1%}]" for k,v in sorted(row.items()) if v[2]>=30)
    print(f"{N[cl]:18s} overall {tw/tf:.1%} | {s}")
print("\n== canonical (most common) group size vs all other sizes, feature classes only")
F=W=Fo=Wo=0
for cl in range(24):
    row=comp_stats[cl]; mode=max(row,key=lambda k:row[k][2])
    f=row[mode][0]; w=row[mode][1]; fo=sum(v[0] for k,v in row.items() if k!=mode); wo=sum(v[1] for k,v in row.items() if k!=mode)
    F+=f;W+=w;Fo+=fo;Wo+=wo
    print(f"{N[cl]:18s} canonical {mode}f: {f:6d} faces err {w/f:5.2%} | other sizes: {fo:5d} faces err {wo/max(fo,1):5.2%} | {wo/max(w+wo,1):4.0%} of this class's misses")
print(f"ALL features: canonical {F} faces, {W} errors ({W/F:.2%}) | non-canonical {Fo} faces ({Fo/(F+Fo):.1%} of feature faces), {Wo} errors ({Wo/Fo:.2%}) = {Wo/(W+Wo):.1%} of feature-face errors")
import json
canon=[]
for cl in range(24):
    row=comp_stats[cl]; mode=max(row,key=lambda k:row[k][2])
    canon.append(dict(cls=cl,mode=int(mode),f=int(row[mode][0]),w=int(row[mode][1]),fo=int(sum(v[0] for k,v in row.items() if k!=mode)),wo=int(sum(v[1] for k,v in row.items() if k!=mode))))
json.dump(dict(canon=canon,groups={N[c]:{str(k):[int(x) for x in v] for k,v in comp_stats[c].items()} for c in range(25)},
  nb_pred_is_nb_true=int(nb_pred_is_nb_true),nb_same_true_correct=int(nb_same_true_correct),whole=int(whole),partial=int(partial)), open(T+"group_stats.json","w"))
