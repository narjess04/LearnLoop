"use client";
import { useState } from "react";
export default function AdvancedEvaluationWidget({kind}:{kind:string}){
 const [contributions,setContributions]=useState([25,18,-10,-5]);
 const [income,setIncome]=useState(3000);
 const [price,setPrice]=useState(350000),[coverage,setCoverage]=useState(90);
 const [fairness,setFairness]=useState([90,15,68,25]);
 const [threshold,setThreshold]=useState(.5),[bootstrapReps,setBootstrapReps]=useState(200);
 const [monitoring,setMonitoring]=useState([.14,.03,4]),[governance,setGovernance]=useState([.86,.06,.17]);
 const [missingAuc,setMissingAuc]=useState(false),[critical,setCritical]=useState(false);
 if(kind==="shapadditive"){
  const sum=50+contributions.reduce((a,b)=>a+b,0);
  return <figure className="interactive-lab"><span className="eyebrow">SIMULATION D’ADDITIVITÉ · PAS DE VALEURS SHAP CALCULÉES</span><h3>De la référence à la prédiction</h3><p>Référence : 50 % · Somme expliquée : <strong>{sum} %</strong></p>{["Endettement","Revenu","Historique crédit","Ancienneté"].map((name,i)=><label className="lab-slider" key={name}><span>{name}<output>{contributions[i]>0?"+":""}{contributions[i]} points</output></span><input type="range" min={-30} max={30} step={1} value={contributions[i]} onChange={e=>setContributions(values=>values.map((v,j)=>i===j?Number(e.target.value):v))}/></label>)}{(sum<0||sum>100)&&<p role="status">La somme sort de [0,100] : ces contributions arbitraires ne décrivent pas une probabilité valide. Aucune borne artificielle n’est appliquée au calcul.</p>}<button onClick={()=>setContributions([25,18,-10,-5])}>Réinitialiser</button><figcaption>Les contributions fictives s’ajoutent sur une même échelle. Avec un vrai explainer, vérifie l’échelle : probabilité ou marge brute/log-odds.</figcaption></figure>;
 }
 if(kind==="robustnessincome"){
  const probability=100/(1+Math.exp((income-2500)/500)),base=100/(1+Math.exp(1));
  return <figure className="interactive-lab"><span className="eyebrow">MODÈLE FICTIF · AUCUNE DONNÉE BANCAIRE RÉELLE</span><h3>Sensibilité au revenu déclaré</h3><label className="lab-slider"><span>Revenu mensuel<output>{income.toLocaleString("fr-FR")} DT</output></span><input type="range" min={1000} max={5000} step={25} value={income} onChange={e=>setIncome(Number(e.target.value))}/></label><div className="lab-output"><span>Probabilité prédite</span><strong>{probability.toFixed(1)} %</strong></div><div className="lab-output"><span>Variation depuis 3 000 DT</span><strong>{(probability-base).toFixed(1)} points</strong></div><button onClick={()=>setIncome(3000)}>Réinitialiser</button><figcaption>p = 1 / (1 + exp((revenu − 2 500) / 500)). Référence : 26,9 %. La sensibilité observée dépend du modèle et des perturbations choisies.</figcaption></figure>;
 }

 if(kind==="conformalinterval"){
  const scores=[5,7,8,10,11,12,14,15,18,20],k=Math.ceil((scores.length+1)*coverage/100),q=k>scores.length?Infinity:scores[k-1]*1000;
  const money=(x:number)=>Number.isFinite(x)?x.toLocaleString("fr-FR")+" DT":x<0?"−∞":"+∞";
  return <figure className="interactive-lab"><span className="eyebrow">SPLIT CONFORMAL · DIX SCORES FICTIFS</span><h3>Couverture et largeur de l’intervalle</h3><label className="lab-slider"><span>Prix prédit<output>{money(price)}</output></span><input type="range" min={150000} max={600000} step={5000} value={price} onChange={e=>setPrice(Number(e.target.value))}/></label><label>Couverture cible <select value={coverage} onChange={e=>setCoverage(Number(e.target.value))}>{[70,80,90,95].map(v=><option key={v} value={v}>{v} %</option>)}</select></label><div className="lab-output"><span>Rang ceil(11 × couverture)</span><strong>{k}/10</strong></div><div className="lab-output"><span>Quantile conforme</span><strong>{money(q)}</strong></div><p aria-live="polite">Intervalle : [{money(price-q)} ; {money(price+q)}]</p>{!Number.isFinite(q)&&<p>À 95 %, k = 11 dépasse les 10 scores disponibles. La borne conforme est infinie ; le maximum observé ne suffit pas à garantir ce niveau de couverture.</p>}<button onClick={()=>{setPrice(350000);setCoverage(90);}}>Réinitialiser</button><figcaption>Scores triés, en milliers de DT : {scores.join(", ")}. Garantie marginale sous échangeabilité, aucune garantie individuelle à 90 %.</figcaption></figure>;
 }
 if(kind==="fairnessmetrics"){
  const [tpA,fpA,tpB,fpB]=fairness,rate=(n:number)=>n.toFixed(1)+" %";
  const precision=(tp:number,fp:number)=>tp+fp?rate(100*tp/(tp+fp)):"Non mesurable";
  const eo=Math.max(Math.abs(tpA-tpB),Math.abs(fpA-fpB));
  return <figure className="interactive-lab"><span className="eyebrow">SIMULATION · DÉFAUT = CLASSE POSITIVE</span><h3>Comparer les erreurs entre groupes</h3><p>Chaque groupe contient 100 défauts et 100 non-défauts.</p>{["Groupe A : vrais positifs (TP)","Groupe A : faux positifs (FP)","Groupe B : vrais positifs (TP)","Groupe B : faux positifs (FP)"].map((name,i)=><label className="lab-slider" key={name}><span>{name}<output>{fairness[i]}</output></span><input type="range" min={0} max={100} step={1} value={fairness[i]} onChange={e=>setFairness(v=>v.map((x,j)=>i===j?Number(e.target.value):x))}/></label>)}<div className="course-table-wrap"><table><thead><tr><th>Métrique</th><th>Groupe A</th><th>Groupe B</th></tr></thead><tbody>{[["TPR",rate(tpA),rate(tpB)],["FPR",rate(fpA),rate(fpB)],["Prédictions positives",rate((tpA+fpA)/2),rate((tpB+fpB)/2)],["Précision",precision(tpA,fpA),precision(tpB,fpB)]].map(row=><tr key={row[0]}>{row.map((v,i)=><td key={i}>{v}</td>)}</tr>)}</tbody></table></div><p aria-live="polite">Écart de TPR : {Math.abs(tpA-tpB).toFixed(1)} points · Écart d’Equalized Odds : {eo.toFixed(1)} points</p><button onClick={()=>setFairness([90,15,68,25])}>Réinitialiser</button><figcaption>Écarts = max − min ; Equalized Odds retient le plus grand écart entre TPR et FPR. Ces données fictives ont des taux de base identiques.</figcaption></figure>;
 }



 if(kind==="diagnosticthreshold"){
  const y=[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0];
  const p=[.95,.85,.75,.65,.55,.45,.35,.25,.15,.05,.9,.7,.6,.4,.3,.2,.1,.05,.15,.25];
  let tp=0,fp=0,fn=0,tn=0;
  p.forEach((v,i)=>{if(v>=threshold){if(y[i])tp++;else fp++;}else{if(y[i])fn++;else tn++;}});
  const percent=(a:number,b:number)=>b?(100*a/b).toFixed(1)+" %":"Non mesurable";
  return <figure className="interactive-lab"><span className="eyebrow">20 CLIENTS FICTIFS · DÉFAUT = 1</span><h3>Changer le seuil de décision</h3><label className="lab-slider"><span>Seuil<output>{Math.round(threshold*100)} %</output></span><input type="range" min={0} max={1} step={.05} value={threshold} onChange={e=>setThreshold(Number(e.target.value))}/></label><div className="course-table-wrap"><table><thead><tr><th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr></thead><tbody><tr>{[tp,fp,fn,tn].map((v,i)=><td key={i}>{v}</td>)}</tr></tbody></table></div><p aria-live="polite">Précision : {percent(tp,tp+fp)} · Rappel : {percent(tp,tp+fn)} · F1 : {percent(2*tp,2*tp+fp+fn)}</p><button onClick={()=>setThreshold(.5)}>Réinitialiser</button><figcaption>La décision utilise p ≥ seuil. Choisis le seuil sur validation, puis fige-le avant le test final.</figcaption></figure>;
 }
 if(kind==="monitoringstatus"){
  const [psi,drop,missing]=monitoring;
  const status=(v:number,g:number,a:number)=>v>a?2:v>g?1:0;
  const severity=Math.max(status(psi,.10,.25),status(drop,.02,.05),status(missing,5,10));
  return <figure className="interactive-lab"><span className="eyebrow">SEUILS FICTIFS · INVESTIGATION AVANT ACTION</span><h3>Simulateur de monitoring</h3>{[["PSI",.5,.01],["Baisse d’AUC (référence − courant)",.15,.005],["Valeurs manquantes (%)",20,1]].map(([name,max,step],i)=><label className="lab-slider" key={String(name)}><span>{name}<output>{monitoring[i].toFixed(i===2?0:3)}</output></span><input type="range" min={0} max={Number(max)} step={Number(step)} value={monitoring[i]} onChange={e=>setMonitoring(v=>v.map((x,j)=>i===j?Number(e.target.value):x))}/></label>)}<p aria-live="polite"><strong>{["Vert : contrôles dans les limites","Orange : à surveiller","Rouge : investigation prioritaire"][severity]}</strong></p><button onClick={()=>setMonitoring([.14,.03,4])}>Réinitialiser</button><figcaption>Le plus haut niveau l’emporte. La baisse d’AUC suppose des labels disponibles. Une alerte ne déclenche pas automatiquement un réentraînement.</figcaption></figure>;
 }
 if(kind==="pairedbootstrap"){
  const labels=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1];
  const a=[.1,.2,.3,.4,.5,.1,.2,.3,.4,.6,.9,.8,.7,.6,.5,.9,.8,.7,.6,.4];
  const b=[.2,.3,.4,.5,.6,.2,.3,.4,.5,.7,.8,.7,.6,.5,.4,.8,.7,.6,.5,.3];
  const auc=(indices:number[],scores:number[])=>{
   const pos=indices.filter(i=>labels[i]===1),neg=indices.filter(i=>labels[i]===0);
   if(!pos.length||!neg.length)return NaN;
   let wins=0;for(const p of pos)for(const n of neg)wins+=scores[p]>scores[n]?1:scores[p]===scores[n]?.5:0;
   return wins/(pos.length*neg.length);
  };
  let seed=42;const random=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296;};
  const differences:number[]=[];
  for(let r=0;r<bootstrapReps;r++){const idx=Array.from({length:20},()=>Math.floor(random()*20)),d=auc(idx,a)-auc(idx,b);if(Number.isFinite(d))differences.push(d);}
  differences.sort((x,y)=>x-y);
  const low=differences[Math.floor((differences.length-1)*.025)],high=differences[Math.ceil((differences.length-1)*.975)];
  const counts=Array(20).fill(0) as number[];
  differences.forEach(d=>counts[Math.min(19,Math.max(0,Math.floor((d+.4)/.8*20)))]++);
  const height=Math.max(...counts,1);
  return <figure className="interactive-lab"><span className="eyebrow">BOOTSTRAP APPARIÉ CALCULÉ · ÉCHANTILLON FICTIF</span><h3>Rééchantillonner les mêmes clients</h3><label className="lab-slider"><span>Réplications<output>{bootstrapReps}</output></span><input type="range" min={50} max={500} step={50} value={bootstrapReps} onChange={e=>setBootstrapReps(Number(e.target.value))}/></label><svg viewBox="0 0 480 180" className="lab-chart" role="img" aria-label="Histogramme des différences AUC appariées, de moins 0,4 à plus 0,4">{counts.map((n,i)=><rect key={i} x={20+i*22} y={145-n/height*120} width={20} height={n/height*120} fill="#7b965d"/>)}<path d="M240 15V148" stroke="#a55e43" strokeDasharray="4 4"/><text x="18" y="170">−0,4</text><text x="235" y="170">0</text><text x="430" y="170">+0,4</text></svg><p aria-live="polite">Différence observée : {(auc(Array.from({length:20},(_,i)=>i),a)-auc(Array.from({length:20},(_,i)=>i),b)).toFixed(3)} · IC percentile approximatif : [{low?.toFixed(3)} ; {high?.toFixed(3)}]</p><button onClick={()=>setBootstrapReps(200)}>Réinitialiser</button><figcaption>Graine fixe 42, {differences.length} réplications avec les deux classes. Ce petit exemple suppose des observations indépendantes et n’évalue pas l’incertitude du réentraînement.</figcaption></figure>;
 }
 if(kind==="governanceverdict"){
  const [auc,brier,psi]=governance;
  const criteria=[["AUC",missingAuc?2:auc>=.8?0:auc>=.7?1:2],["Brier",brier<=.05?0:brier<=.08?1:2],["PSI",psi<=.10?0:psi<=.25?1:2]] as const;
  const worst=Math.max(...criteria.map(c=>c[1]),critical?2:0);
  return <figure className="interactive-lab"><span className="eyebrow">TRIAGE PÉDAGOGIQUE · REVUE HUMAINE REQUISE</span><h3>Moteur de règles explicites</h3>{[["AUC",.5,1,.01],["Brier",0,.2,.005],["PSI",0,.5,.01]].map(([name,min,max,step],i)=><label className="lab-slider" key={String(name)}><span>{name}<output>{i===0&&missingAuc?"Absent":governance[i].toFixed(3)}</output></span><input type="range" disabled={i===0&&missingAuc} min={Number(min)} max={Number(max)} step={Number(step)} value={governance[i]} onChange={e=>setGovernance(v=>v.map((x,j)=>i===j?Number(e.target.value):x))}/></label>)}<label><input type="checkbox" checked={missingAuc} onChange={e=>setMissingAuc(e.target.checked)}/> AUC manquante ou non définie</label><br/><label><input type="checkbox" checked={critical} onChange={e=>setCritical(e.target.checked)}/> Contrôle critique bloquant</label><p>{criteria.map(([name,level])=>name+" : "+["Vert","Orange","Rouge"][level]).join(" · ")}</p><p aria-live="polite"><strong>{["Vert : critères de la simulation satisfaits","Orange : réserves à examiner","Rouge : critères bloquants"][worst]}</strong></p><button onClick={()=>{setGovernance([.86,.06,.17]);setMissingAuc(false);setCritical(false);}}>Réinitialiser</button><figcaption>Seuils illustratifs. Une métrique manquante ou un contrôle critique bloque. Le Brier dépend notamment de la prévalence ; le verdict ne vaut pas approbation de production.</figcaption></figure>;
 }

 return null;
}
