import numpy as np, json, math
T="results/mfcadpp_full/0930_000433/confusion/"
L=["Chamfer","Through hole","Triangular passage","Rectangular passage","6-sided passage","Triangular through slot",
"Rectangular through slot","Circular through slot","Rectangular through step","2-sided through step","Slanted through step",
"O-ring","Blind hole","Triangular pocket","Rectangular pocket","6-sided pocket","Circular end pocket","Rectangular blind slot",
"Vertical circular end blind slot","Horizontal circular end blind slot","Triangular blind step","Circular blind step",
"Rectangular blind step","Round","Stock"]
M=np.load(T+"cm.npy"); gs=json.load(open(T+"group_stats.json"))
np.savetxt(T+"confusion_matrix.csv",np.c_[np.array(L,dtype=object)[:,None],M.astype(object)],fmt="%s",delimiter=",",header="true \\ predicted,"+",".join(L),comments="")
off=M.copy(); np.fill_diagonal(off,0); S=off+off.T; err=int(off.sum()); tot=int(M.sum()); sup=M.sum(1)
tp=np.diag(M); iou=tp/(sup+M.sum(0)-tp)
cats=[("Open or closed sides, same depth","passage, through slot and through step; pocket, blind slot and blind step",739),
("Two or more attributes differ","everything not covered by the other groups",488),
("Profile shape, same feature type","triangular, rectangular or 6-sided; 2-sided, slanted or rectangular step",435),
("Round blind features","O-ring, blind hole and circular end pocket",382),
("Chamfer or round with another feature","",223),
("Stock with any feature","",216),
("Through and blind partners","same profile, with or without a floor",206)]
assert sum(c[2] for c in cats)==err
def pct(x,d=1): return f"{100*x:.{d}f}%"
bars="".join(f'<div class="bar{" hl" if n.startswith("Through and") else ""}"><div class="bl"><b>{n}</b>{f"<span>{s}</span>" if s else ""}</div><div class="bt"><i style="width:{100*v/739:.1f}%"></i></div><div class="bv">{v:,}<span>{pct(v/err)}</span></div></div>' for n,s,v in cats)
# heatmap
vmax=0.04
rows=""
for i in range(25):
    tds=""
    for j in range(25):
        n=int(M[i,j]); r=n/sup[i]
        if i==j:
            tds+=f'<td class="d" data-t="{L[i]}: {n:,} of {int(sup[i]):,} faces correct ({pct(r)})">{100*r:.0f}</td>'
        elif n==0: tds+='<td class="z"></td>'
        else:
            a=min(1,math.sqrt(r/vmax)); k=round(8+92*a)
            tds+=f'<td class="h{" dk" if k>52 else ""}" style="--k:{k}%" data-t="True {L[i]}, predicted {L[j]}: {n:,} faces ({pct(r,2)} of this class)">{n if n>=10 else ""}</td>'
    rows+=f'<tr><th class="rh" title="{L[i]}"><span class="num">{i}</span>{L[i]}</th>{tds}<td class="iou">{iou[i]:.3f}</td></tr>'
head='<tr><th class="corner">True class <span>(rows)</span> / predicted <span>(columns, by id)</span></th>'+"".join(f'<th class="ch" title="{L[j]}">{j}</th>' for j in range(25))+'<th class="ch iouh">IoU</th></tr>'
leg="".join(f'<span class="sw"><i style="--k:{round(8+92*min(1,math.sqrt(v/vmax)))}%"></i>{lab}</span>' for v,lab in [(0.0005,"0.05%"),(0.0025,"0.25%"),(0.01,"1%"),(0.02,"2%"),(0.04,"4%")])
# weak classes
def top(vec,k=3): return ", ".join(f"{L[j]} {int(vec[j])}" for j in np.argsort(-vec)[:k] if vec[j])
partner={6:[17],12:[1],17:[6],8:[22],23:[]}
weak=""
for c in (6,23,12,17,8):
    e=int(S[c].sum()); p=int(sum(S[c,j] for j in partner[c]))
    weak+=f'<tr><td><b>{L[c]}</b></td><td class="n">{iou[c]:.3f}</td><td>{top(off[c])}</td><td>{top(off[:,c])}</td><td class="n">{p} of {e}<span class="sub">{pct(p/e,0) if partner[c] else "no partner class"}</span></td></tr>'
# canonical table
canon=sorted(gs["canon"],key=lambda r:-(r["wo"]/max(r["fo"],1)))
ct=""
for r in canon:
    c=r["cls"]; a=r["w"]/r["f"]; b=r["wo"]/max(r["fo"],1)
    ct+=f'<tr><td>{L[c]}</td><td class="n">{r["mode"]}</td><td class="n">{pct(a,2)}</td><td class="n"><b>{pct(b,2)}</b></td><td class="n">{b/a:.0f}×</td><td class="n">{pct(r["wo"]/max(r["w"]+r["wo"],1),0)}</td></tr>'
F=sum(r["f"] for r in gs["canon"]);W=sum(r["w"] for r in gs["canon"]);Fo=sum(r["fo"] for r in gs["canon"]);Wo=sum(r["wo"] for r in gs["canon"])
G=gs["groups"]
def gr(name,k): f,w,g=G[name][str(k)]; return pct(w/f,1 if w/f<0.1 else 0), g
html=f'''<title>UV-Net Confusion Matrix</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column for the verdict and tables; the 25x25 matrix breaks out wider and scrolls on its own */
:root{{--bg:#f4f6f8;--surface:#ffffff;--fg:#17202a;--muted:#5d6b7a;--rule:#d5dce3;--heat:#0d366b;--accent:#1c5cab;--mark:#b4530a;--diag:#e9edf1;
--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#12161b;--surface:#1a2027;--fg:#e6ebf0;--muted:#98a5b3;--rule:#2e3742;--heat:#9ec5f4;--accent:#86b6ef;--mark:#f0a45d;--diag:#252d37;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#12161b;--surface:#1a2027;--fg:#e6ebf0;--muted:#98a5b3;--rule:#2e3742;--heat:#9ec5f4;--accent:#86b6ef;--mark:#f0a45d;--diag:#252d37;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 var(--sans)}}
main{{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:40px 64px;display:flex;flex-direction:column;gap:44px}}
section{{display:flex;flex-direction:column;gap:14px;min-width:0}}
.col{{max-width:70ch}}
h1{{font-size:30px;line-height:1.15;font-weight:600;margin:0;text-wrap:balance;letter-spacing:-.01em}}
h2{{font-size:19px;font-weight:600;margin:0;text-wrap:balance}}
p{{margin:0}} .meta{{font:13px var(--mono);color:var(--muted)}}
.lede{{font-size:17px;line-height:1.5}}
.note{{color:var(--muted);font-size:13.5px}}
.bars{{display:flex;flex-direction:column;gap:9px;max-width:820px}}
.bar{{display:grid;grid-template-columns:minmax(0,300px) minmax(60px,1fr) 104px;gap:14px;align-items:center}}
.bl{{display:flex;flex-direction:column;min-width:0}} .bl b{{font-weight:500}} .bl span{{color:var(--muted);font-size:12.5px;line-height:1.35}}
.bt{{height:14px}} .bt i{{display:block;height:100%;background:var(--accent);border-radius:0 4px 4px 0}}
.bar.hl .bt i{{background:var(--mark)}} .bar.hl .bl b::after{{content:"  ← your question";color:var(--mark);font-weight:400;font-size:12.5px}}
.bv{{font:13.5px var(--mono);font-variant-numeric:tabular-nums;text-align:right}} .bv span{{color:var(--muted);margin-left:8px}}
@media (max-width:620px){{.bar{{grid-template-columns:1fr 92px}} .bl{{grid-column:1/-1}}}}
.scroll{{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px}}
table.cm{{border-collapse:separate;border-spacing:2px;font:11px var(--mono);font-variant-numeric:tabular-nums}}
.cm th{{font-weight:400;color:var(--muted)}}
.cm .corner{{text-align:left;font:12px var(--sans);vertical-align:bottom;white-space:nowrap;color:var(--fg)}} .cm .corner span{{color:var(--muted)}}
.cm .ch{{width:30px;min-width:30px;height:24px}} .cm .rh{{text-align:left;white-space:nowrap;padding-right:10px;font:12.5px var(--sans);color:var(--fg)}}
.cm .num{{display:inline-block;width:22px;font:11px var(--mono);color:var(--muted)}}
.cm td{{width:30px;min-width:30px;height:26px;text-align:center;border-radius:3px;padding:0}}
.cm td.d{{background:var(--diag);color:var(--muted)}}
.cm td.h{{background:color-mix(in oklab,var(--heat) var(--k),var(--surface));color:var(--fg);cursor:default}} .cm td.h.dk{{color:var(--surface)}}
.cm td.h:hover,.cm td.d:hover{{outline:2px solid var(--mark);outline-offset:0}}
.cm td.iou,.cm th.iouh{{width:52px;min-width:52px;padding-left:8px;color:var(--fg);text-align:right}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;font-size:12.5px;color:var(--muted)}}
.sw{{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono)}} .sw i{{width:18px;height:14px;border-radius:3px;background:color-mix(in oklab,var(--heat) var(--k),var(--surface));border:1px solid var(--rule)}}
.tw{{overflow-x:auto}}
table.t{{border-collapse:collapse;width:100%;font-size:13.5px}}
.t th{{text-align:left;font-weight:500;color:var(--muted);font-size:12.5px;border-bottom:1px solid var(--rule);padding:6px 12px 6px 0;vertical-align:bottom}}
.t td{{border-bottom:1px solid var(--rule);padding:7px 12px 7px 0;vertical-align:top}}
.t .n,.t th.n{{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}} .t th.n{{font-family:var(--sans)}}
.sub{{display:block;color:var(--muted);font-size:12px}}
ul{{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:8px}}
#tip{{position:fixed;pointer-events:none;background:var(--fg);color:var(--bg);font-size:12.5px;padding:6px 9px;border-radius:4px;max-width:280px;z-index:5}}
</style>
<main>
<section class="col">
<p class="meta">MFCAD++ test split · checkpoint from epoch 51 · {tot:,} faces · {err:,} wrong ({pct(err/tot,2)})</p>
<h1>The weak classes are not mixing up through and blind</h1>
<p class="lede">Through and blind partners make up {pct(206/err)} of all wrong faces. The largest group, {pct(739/err)}, is features of the same depth that differ in how many sides are open, such as a rectangular passage called a rectangular through slot.</p>
<p class="lede">The stronger pattern is where the errors sit. {pct(Wo/(W+Wo),0)} of wrong feature faces belong to features whose face count differs from the usual one for their class, which is what happens when a neighbouring feature cuts into them. Those features hold only {pct(Fo/(F+Fo),0)} of the feature faces.</p>
</section>

<section>
<h2>Wrong faces by kind of confusion</h2>
<div class="bars">{bars}</div>
<p class="note col">Each wrong face is counted once, by its true and predicted class. {pct(514/err)} of errors cross the through/blind line in any way; the rest stay inside one side of it.</p>
</section>

<section>
<h2>Confusion matrix</h2>
<p class="note col">Each row is a true class. Off-diagonal cells show the number of faces given the column's label, and the shade is that count as a share of the row's faces. Counts under 10 are shaded but not printed. Diagonal cells show recall in percent. Hover a cell for details.</p>
<div class="scroll"><table class="cm">{head}{rows}</table></div>
<div class="legend"><span>Share of the true class's faces (square-root scale):</span>{leg}</div>
</section>

<section>
<h2>What the five weakest classes are confused with</h2>
<div class="tw"><table class="t"><tr><th>Class</th><th class="n">IoU</th><th>Its faces were labelled as</th><th>Faces wrongly given its label were</th><th class="n">Errors with its through/blind partner</th></tr>{weak}</table></div>
<ul class="col">
<li><b>Rectangular through slot</b> is mostly swapped with rectangular passage, in both directions. Both go through the part. They differ in whether the cut is closed on all four sides.</li>
<li><b>Blind hole</b> is mostly swapped with O-ring, and <b>rectangular blind slot</b> with rectangular pocket. In each pair both features are blind.</li>
<li><b>Rectangular through step</b> is the one weak class with a real through/blind share, about a fifth of its errors. A slightly larger share of its errors is confusion with stock.</li>
<li><b>Round</b> is a different case. Its errors are with through hole and chamfer, on ordinary single-face rounds, and there are only {int(S[23].sum())} of them. The class is rare, so few errors move its IoU a lot.</li>
</ul>
</section>

<section>
<h2>Error rate on usual and unusual feature shapes</h2>
<p class="note col">Faces with the same label that touch each other are treated as one feature. For each class, the most common face count is taken as the usual shape. A different count means a neighbouring feature removed or split faces, or two features of the same class touch. Feature instances are not labelled in the data, so this grouping is an approximation.</p>
<div class="tw"><table class="t"><tr><th>Class</th><th class="n">Usual face count</th><th class="n">Error rate, usual count</th><th class="n">Error rate, other counts</th><th class="n">Ratio</th><th class="n">Share of the class's missed faces on other counts</th></tr>{ct}
<tr><td><b>All 24 feature classes</b></td><td class="n"></td><td class="n">{pct(W/F,2)}</td><td class="n"><b>{pct(Wo/Fo,2)}</b></td><td class="n">{(Wo/Fo)/(W/F):.0f}×</td><td class="n">{pct(Wo/(W+Wo),0)}</td></tr></table></div>
<ul class="col">
<li>A rectangular passage with its usual 4 faces is wrong {gr("rect_passage",4)[0]} of the time. With 3 faces it is wrong {gr("rect_passage",3)[0]} of the time.</li>
<li>An O-ring with its usual 3 faces is wrong {gr("Oring",3)[0]} of the time. With 2 faces, the same count as a blind hole, it is wrong {gr("Oring",2)[0]} of the time.</li>
<li>A circular end pocket with 5 faces is wrong {gr("circ_end_pocket",5)[0]} of the time. With 2 faces it is wrong {gr("circ_end_pocket",2)[0]} of the time.</li>
<li>{pct(gs["nb_same_true_correct"]/err,0)} of wrong faces touch a face of the same feature that the model labelled correctly, so the model often splits one feature between two labels.</li>
</ul>
</section>

<section class="col">
<h2>What this means for the next variant</h2>
<ul>
<li><b>Whether a cut goes through the part is not the main gap.</b> A change aimed only at that would address under a tenth of the errors.</li>
<li><b>Features cut by neighbours are the main gap.</b> A passage that has lost a wall looks like a slot. This analysis cannot tell whether the surrounding faces hold enough information to recover the right label, or whether some of these cases are ambiguous in the data.</li>
<li><b>Round is a local geometry problem</b>, between convex and concave curved faces and chamfers. Edge convexity or face curvature are the candidate inputs. It is {pct(S[23].sum()/err,0)} of all errors.</li>
<li><b>The split in the last table is now a metric.</b> Any variant can be scored separately on usual and unusual features, which shows what it fixed.</li>
</ul>
</section>
</main>
<div id="tip" hidden></div>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('td[data-t]').forEach(td=>{{
  td.addEventListener('mousemove',e=>{{tip.textContent=td.dataset.t;tip.hidden=false;
    const x=Math.min(e.clientX+14,innerWidth-300),y=e.clientY+16;tip.style.left=Math.max(8,x)+'px';tip.style.top=y+'px';}});
  td.addEventListener('mouseleave',()=>tip.hidden=true);
}});
</script>
'''
open(T+"confusion_report.html","w").write(html); print(len(html), "IoU check", iou[[6,23,12,17,8]].round(3), "round errs", int(S[23].sum()), S[23].sum()/err)
