"use client";
import { useState, type ReactNode } from "react";
function Lab({title,children,note}:{title:string;children:ReactNode;note:string}){return <figure className="interactive-lab"><span className="eyebrow">{title}</span>{children}<figcaption>{note}</figcaption></figure>;}
function Slider({label,value,min,max,step=1,onChange}:{label:string;value:number;min:number;max:number;step?:number;onChange:(n:number)=>void}){return <label className="lab-slider"><span>{label}<output>{value.toFixed(step<1?2:0)}</output></span><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(Number(e.target.value))}/></label>;}
function Probabilistic(){
 const [tau,setTau]=useState(.9);const [prediction,setPrediction]=useState(100);const [width,setWidth]=useState(15);
 const y=120;const error=y-prediction;const loss=Math.max(tau*error,(tau-1)*error);const values=[80,94,100,108,130];const covered=values.filter(v=>v>=100-width&&v<=100+width).length/values.length;
 return <Lab title="QUANTILES · PERTE ET INTERVALLES" note="Cinq observations fictives servent à calculer la couverture descriptive. La largeur et la couverture se comparent ensemble ; cette interaction ne calibre pas de modèle et ne donne aucune garantie temporelle."><Slider label="Quantile τ" min={.05} max={.95} step={.05} value={tau} onChange={setTau}/><Slider label="Prévision quantile" min={70} max={150} value={prediction} onChange={setPrediction}/><div className="lab-output"><span>Cible observée : 120 · Pinball Loss</span><strong>{loss.toFixed(2)}</strong></div><Slider label="Demi-largeur autour de 100" min={0} max={40} value={width} onChange={setWidth}/><p aria-live="polite">Intervalle [{100-width}, {100+width}] · largeur {2*width} · couverture {(covered*100).toFixed(0)} %.</p></Lab>;
}
function Anomalies(){
 const [h,setH]=useState(25);const values=Array.from({length:40},(_,t)=>100+3*Math.sin(t)+ (t>=25?15:0));let cumulative=0;const alerts:number[]=[];
 const sums=values.map((v,t)=>{cumulative=Math.max(0,cumulative+v-100-3);const value=cumulative;if(cumulative>h){alerts.push(t);cumulative=0;}return value;});
 return <Lab title="CUSUM · CHANGEMENT PERSISTANT" note="Signal synthétique : référence 100, tolérance k = 3 et hausse durable à l’indice 25. CUSUM est réinitialisé après chaque alerte ; une référence inchangée peut produire des alertes répétées après rupture. Le seuil exige une calibration opérationnelle."><Slider label="Seuil d’alerte h" value={h} min={5} max={80} onChange={setH}/><svg className="lab-chart" viewBox="0 0 540 200" role="img" aria-label="Somme CUSUM et seuil d’alerte"><path d="M25 15V170H520" fill="none" stroke="#cbd9b8"/><polyline points={sums.map((v,t)=>`${25+t*12},${170-v*1.5}`).join(" ")} fill="none" stroke="#7a935c" strokeWidth="2"/><path d={`M25 ${170-h*1.5}H520`} stroke="#b47c4d" strokeDasharray="5 5"/></svg><p aria-live="polite">Indices d’alerte : {alerts.join(", ")||"aucun"}.</p></Lab>;
}
function Voting(){
 const [a,setA]=useState(.8);const [b,setB]=useState(.6);const [c,setC]=useState(.7);const probability=(a+b+c)/3;const votes=[a,b,c].filter(p=>p>=.5).length;
 return <Lab title="HARD VOTING ET SOFT VOTING" note="Probabilités fictives, poids égaux et seuil 0,5. La qualité des probabilités et la corrélation des erreurs doivent être évaluées sur validation."><Slider label="Probabilité modèle 1" value={a} min={0} max={1} step={.01} onChange={setA}/><Slider label="Probabilité modèle 2" value={b} min={0} max={1} step={.01} onChange={setB}/><Slider label="Probabilité modèle 3" value={c} min={0} max={1} step={.01} onChange={setC}/><div className="lab-output"><span>Soft Voting · probabilité moyenne</span><strong>{(probability*100).toFixed(1)} %</strong></div><p aria-live="polite">Soft : classe {probability>=.5?1:0}. Hard : classe {votes>=2?1:0} ({votes}/3 votes positifs).</p></Lab>;
}
function CostDecision(){
 const [threshold,setThreshold]=useState(.5);const [fnCost,setFnCost]=useState(500);const p=[.93,.81,.72,.64,.57,.48,.39,.31,.24,.18,.11,.05];const y=[1,1,0,1,0,1,0,0,1,0,0,0];let tp=0,fp=0,fn=0,tn=0;
 p.forEach((v,i)=>{if(v>=threshold){if(y[i])tp++;else fp++;}else{if(y[i])fn++;else tn++;}});const cost=20*tp+10*fp+fnCost*fn;const bayes=10/(10+fnCost-20);
 return <Lab title="SEUIL ET COÛT DES DÉCISIONS" note="Douze observations fictives. Coûts : FP = 10, TP = 20, TN = 0 ; coût FN réglable. Le seuil théorique suppose des probabilités calibrées. Pour un modèle réel, choisir la règle sur validation avant l’audit test."><Slider label="Seuil de décision" value={threshold} min={0} max={1} step={.01} onChange={setThreshold}/><Slider label="Coût d’un faux négatif" value={fnCost} min={50} max={1000} step={10} onChange={setFnCost}/><p>TP {tp} · FP {fp} · FN {fn} · TN {tn}</p><div className="lab-output" aria-live="polite"><span>Coût total illustratif</span><strong>{cost} €</strong></div><p>Seuil théorique : {(100*bayes).toFixed(2)} %.</p></Lab>;
}
function Survival(){
 const [observed,setObserved]=useState(false);const [time,setTime]=useState(5);const times=[2,4,5,7,8];const events=[1,1,observed?1:0,1,0];let survival=1;let current=1;const points=["25,25"];
 times.forEach((t,i)=>{points.push(`${25+t*58},${175-survival*150}`);survival*=1-events[i]/(5-i);points.push(`${25+t*58},${175-survival*150}`);if(t<=time)current=survival;});
 return <Lab title="KAPLAN-MEIER · ÉVÉNEMENT OU CENSURE" note="Cinq individus fictifs. Une censure au temps 5 retire une personne des risques ultérieurs sans provoquer de chute immédiate de la survie. Une censure n’est pas une absence définitive d’événement."><label><input type="checkbox" checked={observed} onChange={e=>setObserved(e.target.checked)}/> Au temps 5, considérer un événement observé plutôt qu’une censure</label><Slider label="Temps évalué" value={time} min={0} max={8} onChange={setTime}/><svg className="lab-chart" viewBox="0 0 540 210" role="img" aria-label="Courbe en escalier Kaplan-Meier"><path d="M25 15V175H510" fill="none" stroke="#cbd9b8"/><polyline points={points.join(" ")} fill="none" stroke="#7a935c" strokeWidth="3"/><text x="28" y="200">0</text><text x="480" y="200">8</text></svg><p aria-live="polite">Survie estimée S({time}) : {(current*100).toFixed(1)} %.</p></Lab>;
}
function Ranking(){
 const initial=[{name:"XGBoost",rel:3},{name:"CNN",rel:0},{name:"Random Forest",rel:2},{name:"Transformers",rel:1}];const [items,setItems]=useState(initial);
 const dcg=(rels:number[])=>rels.reduce((sum,rel,i)=>sum+(2**rel-1)/Math.log2(i+2),0);const ndcg=dcg(items.map(x=>x.rel))/dcg([3,2,1,0]);
 function move(index:number,delta:number){setItems(old=>{const next=[...old];[next[index],next[index+delta]]=[next[index+delta],next[index]];return next;});}
 return <Lab title="RÉORGANISER UN CLASSEMENT" note="Pertinences fictives fixes ; NDCG@4 avec gain exponentiel 2^rel−1 et décote logarithmique. Les résultats se comparent par requête. Une recommandation exige aussi diversité, exclusion des items consommés et évaluation hors échantillon."><div className="lab-output" aria-live="polite"><span>NDCG@4</span><strong>{ndcg.toFixed(3)}</strong></div><ol>{items.map((item,i)=><li key={item.name}>{item.name} · pertinence {item.rel} <button disabled={i===0} aria-label={`Monter ${item.name}`} onClick={()=>move(i,-1)}>↑</button> <button disabled={i===items.length-1} aria-label={`Descendre ${item.name}`} onClick={()=>move(i,1)}>↓</button></li>)}</ol><button onClick={()=>setItems(initial)}>Réinitialiser</button></Lab>;
}
function DataCentric(){
 const [budget,setBudget]=useState(2);const [strategy,setStrategy]=useState("entropy");const rows=[[.98,.01,.01],[.34,.33,.33],[.48,.47,.05],[.7,.2,.1],[.05,.15,.8]];
 const ranked=rows.map((p,i)=>{const sorted=[...p].sort((a,b)=>b-a);return {i,score:strategy==="entropy"?-p.reduce((s,v)=>s+v*Math.log(v),0):strategy==="margin"?-(sorted[0]-sorted[1]):1-sorted[0]};}).sort((a,b)=>b.score-a.score).slice(0,budget).map(x=>x.i);
 return <Lab title="ACTIVE LEARNING · BUDGET DE REVUE" note="Probabilités fictives. L’incertitude sélectionne des candidats à annoter ; elle ne prouve pas une erreur de label. Un lot utile peut aussi exiger diversité, représentativité et revue humaine."><div className="lab-toggle">{[["entropy","Entropie"],["margin","Marge"],["uncertainty","Incertitude"]].map(([id,name])=><button key={id} aria-pressed={strategy===id} onClick={()=>setStrategy(id)}>{name}</button>)}</div><Slider label="Nombre d’exemples à revoir" min={1} max={5} value={budget} onChange={setBudget}/><ol>{rows.map((p,i)=><li key={i}>Exemple {i+1} · [{p.map(v=>v.toFixed(2)).join(", ")}] {ranked.includes(i)?"→ sélectionné pour revue":""}</li>)}</ol></Lab>;
}
export default function SpecializedWidget({kind}:{kind:string}){
 if(kind==="probabilistic")return <Probabilistic/>;
 if(kind==="anomalies")return <Anomalies/>;
 if(kind==="voting")return <Voting/>;
 if(kind==="costdecision")return <CostDecision/>;
 if(kind==="survival")return <Survival/>;
 if(kind==="ranking")return <Ranking/>;
 if(kind==="datacentric")return <DataCentric/>;
 return null;
}
