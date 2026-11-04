"use client";
import { accountStorage } from "@/lib/account-storage";
import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, Check } from "lucide-react";
import chapterOne from "@/lib/ml-chapter-one.json";
type CourseId=keyof typeof chapterOne.courses;
function MiniQuiz({courseId}:{courseId:CourseId}){
 const items=chapterOne.quizzes[courseId];
 const [choices,setChoices]=useState<number[]>(Array(items.length).fill(-1));
 const [show,setShow]=useState(false);const [ready,setReady]=useState(false);const [error,setError]=useState(false);
 const key="recap-quiz-"+courseId;
 useEffect(()=>{try{const saved=JSON.parse(accountStorage.getItem(key)||"null");if(Array.isArray(saved?.choices)&&saved.choices.length===items.length&&saved.choices.every((v:unknown,i:number)=>Number.isInteger(v)&&Number(v)>=-1&&Number(v)<items[i].options.length)){setChoices(saved.choices);setShow(saved.show===true);}}catch{setError(true);}setReady(true);},[key,items]);
 useEffect(()=>{if(ready)try{accountStorage.setItem(key,JSON.stringify({choices,show}));}catch{setError(true);}},[choices,show,ready,key]);
 if(!items.length)return null;
 const score=choices.filter((v,i)=>v===items[i].answer).length;
 return <div className="ml-scenarios"><div className="scenario-heading"><div><span className="eyebrow">MISE EN SITUATION</span><h3>Teste ton raisonnement.</h3></div><span className="scenario-score" aria-live="polite">{show?score:choices.filter(v=>v>=0).length}/{items.length}</span></div>{items.map((item,i)=><fieldset key={item.question}><legend>{i+1}. {item.question}</legend><div className="scenario-options">{item.options.map((option,j)=><label key={option} className={(choices[i]===j?"is-chosen ":"")+(show&&j===item.answer?"is-answer":"")}><input type="radio" name={courseId+"-quiz-"+i} checked={choices[i]===j} onChange={()=>{setChoices(c=>c.map((v,k)=>k===i?j:v));setShow(false);}}/>{option}{show&&j===item.answer&&<Check size={16}/>}</label>)}</div>{show&&<p className="scenario-explanation">{item.explanation}</p>}</fieldset>)}<button className="button-dark" disabled={choices.some(v=>v<0)} onClick={()=>setShow(true)}>Vérifier mes réponses<ArrowRight size={16}/></button>{show&&<p role="status" className="scenario-summary">{score}/{items.length} réponses correctes. Justifie tes choix et reprends les explications des erreurs.</p>}{error&&<p>L’enregistrement du compte de cet exercice n’est pas disponible.</p>}</div>;
}
function Folds(){
 const [fold,setFold]=useState(0);
 return <figure className="interactive-lab"><span className="eyebrow">5-FOLD CROSS-VALIDATION EXPLORER</span><h3>Un pli de validation à chaque itération.</h3><div className="fold-blocks">{Array.from({length:5},(_,i)=><div key={i} className={fold===i?"fold-validation":""}><span>Fold {i+1}</span><strong>{fold===i?"Validation":"Train"}</strong></div>)}</div><div className="lab-actions"><button className="text-button" disabled={fold===0} onClick={()=>setFold(v=>v-1)}><ArrowLeft size={15}/>Précédent</button><span>Itération {fold+1}/5</span><button className="text-button" disabled={fold===4} onClick={()=>setFold(v=>v+1)}>Suivant<ArrowRight size={15}/></button></div><figcaption>Le fold {fold+1} est réservé à la validation. Les quatre autres servent à ajuster toutes les étapes de la pipeline. Aucun test final ne participe à ces cinq itérations.</figcaption></figure>;
}
type Series={label:string;color:string;values:number[]};
function CurveFigure({series,labels,note}:{series:Series[];labels:string[];note:string}){
 const low=Math.min(0,...series.flatMap(s=>s.values));const high=Math.max(.01,...series.flatMap(s=>s.values));
 const x=(i:number)=>35+i/Math.max(1,labels.length-1)*415;
 const y=(v:number)=>180-(v-low)/(high-low)*145;
 return <figure className="interactive-lab"><svg viewBox="0 0 480 215" className="lab-chart" role="img" aria-label={series.map(s=>s.label).join(", ")}><path d="M35 25V180H450" fill="none" stroke="#d1dec0"/>{series.map(s=><polyline key={s.label} points={s.values.map((v,i)=>x(i).toFixed(3)+","+y(v).toFixed(3)).join(" ")} fill="none" stroke={s.color} strokeWidth={2.5}/>)}{labels.map((label,i)=><text key={i} x={x(i)} y={203} textAnchor="middle">{label}</text>)}</svg><div className="curve-legend">{series.map(s=><span key={s.label}><i style={{background:s.color}}/>{s.label}</span>)}</div><figcaption>{note}</figcaption></figure>;
}
function Polynomial(){
 const [degree,setDegree]=useState(1);const data=chapterOne.figures.polynomial;
 const model=data.models.find(m=>m.degree===degree)!;
 const x=(v:number)=>30+(v+3)/6*420;const y=(v:number)=>180-(v+3)/6*150;
 return <figure className="interactive-lab"><span className="eyebrow">COMPLEXITÉ POLYNOMIALE · MODÈLES RÉELLEMENT AJUSTÉS</span><div className="lab-toggle" role="group" aria-label="Degré du polynôme">{data.models.map(m=><button key={m.degree} aria-pressed={m.degree===degree} onClick={()=>setDegree(m.degree)}>Degré {m.degree}</button>)}</div><svg className="lab-chart" viewBox="0 0 480 210" role="img" aria-label={"Régression polynomiale de degré "+degree+" ; points train et validation"}><defs><clipPath id="poly-clip"><rect x={20} y={20} width={440} height={170}/></clipPath></defs><path d="M30 20V180H450" fill="none" stroke="#d3dec1"/><g clipPath="url(#poly-clip)">{data.train.map(([a,b],i)=><circle key={"tr"+i} cx={x(a)} cy={y(b)} r={2.5} fill="#9db77e"/>)}{data.validation.map(([a,b],i)=><circle key={"va"+i} cx={x(a)} cy={y(b)} r={3} fill="#c8ab7f"/>)}<polyline points={model.curve.map((v,i)=>x(data.x[i]).toFixed(3)+","+y(v).toFixed(3)).join(" ")} fill="none" stroke="#637e45" strokeWidth={2.5}/></g><text x={30} y={204}>−3</text><text x={440} y={204}>3</text></svg><div className="lab-output"><span>MSE train</span><strong>{model.trainMSE.toFixed(4)}</strong></div><div className="lab-output"><span>MSE validation</span><strong>{model.validationMSE.toFixed(4)}</strong></div><div className="curve-legend"><span><i style={{background:"#9db77e"}}/>Train</span><span><i style={{background:"#c8ab7f"}}/>Validation</span></div><figcaption>Courbes calculées avec le protocole du Coding Lab, seed 42, scikit-learn 1.7.1. L’axe vertical est borné à [−3, 3] ; certaines prédictions peuvent sortir du cadre. Ce jeu de validation sert à étudier les réglages, pas à annoncer une performance finale.</figcaption></figure>;
}
function GridSearch(){
 const [choice,setChoice]=useState(0);const values=[.01,.1,1,10];
 return <figure className="interactive-lab"><span className="eyebrow">GRID SEARCH · 8 CONFIGURATIONS</span><div className="search-grid"><span>C</span><strong>Sans pondération</strong><strong>balanced</strong>{values.flatMap((value,i)=>[<strong key={"l"+i}>{value}</strong>,...[0,1].map((w)=><button key={i+"-"+w} aria-pressed={choice===2*i+w} onClick={()=>setChoice(2*i+w)}>Config. {2*i+w+1}</button>)])}</div><div className="lab-output"><span>C = {values[Math.floor(choice/2)]} · class_weight = {choice%2?"balanced":"None"}</span><strong>#{choice+1}</strong></div><figcaption>Chaque configuration est évaluée sur les plis internes. Avec cinq plis, cette grille nécessite 8 × 5 ajustements, puis un refit si demandé. Aucun score n’est simulé ici.</figcaption></figure>;
}
function Workflow(){return <figure className="lifecycle-figure"><div>{["Besoin métier","Données","Protocole","Baseline","Pipelines","Validation","Tuning","Test final","Déploiement","Monitoring"].map((label,i)=><span key={label}><small>{i+1}</small>{label}{i<9&&<ArrowRight size={15}/>}</span>)}</div><figcaption>Schéma de travail itératif. Le test final reste à l’écart de la sélection ; déploiement et monitoring demandent ensuite leurs propres contrôles.</figcaption></figure>;}
function ModelComparison(){return <figure className="interactive-lab"><span className="eyebrow">COMPARAISON ILLUSTRATIVE · VALEURS FICTIVES</span><div className="model-bars">{[["Dummy",.5],["Logistique",.86],["Arbre",.82],["Random Forest",.9],["Boosting",.91]].map(([name,score])=><div key={String(name)}><span>{name}</span><div><i style={{width:Number(score)*100+"%"}}/></div><strong>{Number(score).toFixed(2)}</strong></div>)}</div><figcaption>Ces scores fictifs montrent uniquement comment lire un tableau comparatif. Les résultats observés dépendent des données, des plis et de la métrique.</figcaption></figure>;}
export default function MLAdvancedWidget({kind,courseId}:{kind:string;courseId:CourseId}){
 switch(kind){
 case "quiz":return <MiniQuiz courseId={courseId}/>;
 case "folds":return <Folds/>;
 case "polynomial":return <Polynomial/>;
 case "biasvariance":return <CurveFigure labels={["Très faible","Faible","Moyenne","Élevée","Très élevée"]} series={[{label:"Biais²",color:"#b6a078",values:[.8,.5,.25,.12,.06]},{label:"Variance",color:"#93aa75",values:[.05,.1,.2,.4,.75]},{label:"Erreur attendue",color:"#697c58",values:[.95,.7,.55,.62,.91]}]} note="Valeurs fictives : bruit irréductible fixé à 0,10. Cette illustration n’est pas une loi générale de l’évolution de toute famille de modèles."/>;
 case "learningcurves":return <CurveFigure labels={["50","100","200","400","800","1600"]} series={[{label:"Score train",color:"#8ea968",values:[.99,.98,.96,.94,.93,.92]},{label:"Score validation",color:"#c0a178",values:[.65,.71,.77,.83,.88,.9]}]} note="Exemple simulé selon la taille du train ; les learning curves d’un vrai modèle doivent être mesurées et accompagnées de leur variabilité."/>;
 case "splitbar":return <figure className="interactive-lab"><span className="eyebrow">EXEMPLE · 10 000 OBSERVATIONS</span><div className="split-bar"><span style={{flex:6}}>Train<br/>60 % · 6 000</span><span style={{flex:2}}>Validation<br/>20 % · 2 000</span><span style={{flex:2}}>Test<br/>20 % · 2 000</span></div><figcaption>Proportions illustratives, à adapter au problème, à la dépendance des données et à la quantité disponible.</figcaption></figure>;
 case "gridsearch":return <GridSearch/>;
 case "workflow":return <Workflow/>;
 case "modelcomparison":return <ModelComparison/>;
 case "capstone":return <div className="ml-next-course"><span>PROJET FINAL · CHAPITRE 01</span><h3>Classification tabulaire de A à Z</h3><p>Construis le dépôt, fournis tes résultats et prépare la soutenance. La lecture des huit cours ne remplace pas la réalisation de ce projet.</p><a href="/?projet=capstone-chapter-one">Ouvrir les jalons du projet<ArrowRight size={17}/></a><p><a href="/parcours/machine-learning">Revenir au schéma des douze chapitres</a></p></div>;
 default:return null;
 }
}
