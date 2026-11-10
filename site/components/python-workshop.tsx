"use client";
import { useEffect, useId, useRef, useState } from "react";
import { Play, Square, RotateCcw, Download, Code2 } from "lucide-react";
import { accountStorage } from "@/lib/account-storage";

type Check = { name: string; passed: boolean; detail?: string };
type Result = { output?: string; error?: string; line?: number; figures?: string[]; checks?: Check[] };
type Props = { lessonId: string; starterUrl?: string; initialCode?: string; solutionUrl?: string };
export default function PythonWorkshop({ lessonId, starterUrl, initialCode = "", solutionUrl }: Props) {
 const id=useId(), key="recap-python-"+lessonId;
 const [code,setCode]=useState(""), [starter,setStarter]=useState(""), [loaded,setLoaded]=useState(false);
 const [loadError,setLoadError]=useState(""), [busy,setBusy]=useState(false), [status,setStatus]=useState("Prêt à charger Python");
 const [result,setResult]=useState<Result|null>(null), [syntax,setSyntax]=useState<Result|null>(null), [stale,setStale]=useState(false);
 const [engine,setEngine]=useState(false), [solution,setSolution]=useState<string|null>(null), [solutionError,setSolutionError]=useState("");
 const worker=useRef<Worker|null>(null), timer=useRef<ReturnType<typeof setTimeout>|null>(null), runId=useRef(0);
 const syntaxId=useRef(0), editor=useRef<HTMLTextAreaElement>(null), gutter=useRef<HTMLDivElement>(null);
 const active=useRef(false), latest=useRef<string|null>(null), submitted=useRef("");
 latest.current=loaded?code:latest.current;
 useEffect(()=>{
  let cancelled=false;
  async function load(){
   try{
    const response=starterUrl?await fetch(starterUrl):null;
    if(response&&!response.ok)throw Error("Le fichier de départ est indisponible.");
    const text=response?await response.text():initialCode;
    if(cancelled)return;
    const saved=accountStorage.getItem(key);
    let draft:unknown; try{draft=saved?JSON.parse(saved):null;}catch{draft=null;}
    setStarter(text);setCode(typeof draft==="string"?draft:text);setLoaded(true);setLoadError("");
   }catch(e){if(!cancelled)setLoadError((e as Error).message);}
  }
  void load();
  return()=>{cancelled=true;if(latest.current!==null)accountStorage.setItem(key,JSON.stringify(latest.current));worker.current?.terminate();worker.current=null;if(timer.current)clearTimeout(timer.current);};
 },[key,starterUrl,initialCode]);
 useEffect(()=>{
  if(!loaded)return;
  const save=setTimeout(()=>accountStorage.setItem(key,JSON.stringify(code)),700);
  return()=>clearTimeout(save);
 },[code,key,loaded]);
 useEffect(()=>{
  if(!engine||busy)return;
  const version=++syntaxId.current;
  const handle=setTimeout(()=>worker.current?.postMessage({type:"syntax",code,id:version}),600);
  return()=>clearTimeout(handle);
 },[code,engine,busy]);
 function stop(message="Exécution arrêtée. Tu peux corriger puis relancer."){
  runId.current++;worker.current?.terminate();worker.current=null;active.current=false;
  if(timer.current)clearTimeout(timer.current);
  setEngine(false);setBusy(false);setStatus(message);
 }
 function getWorker(){
  if(worker.current)return worker.current;
  const w=new Worker("/python/runner.js",{type:"module"});worker.current=w;
  w.onmessage=({data})=>{
   if(data.type==="syntax"){if(data.id===syntaxId.current)setSyntax(data);return;}
   if(data.id!==runId.current)return;
   if(data.type==="status"){setStatus(data.message);return;}
   active.current=false;if(timer.current)clearTimeout(timer.current);
   setBusy(false);setEngine(data.engineReady===true);setResult(data);setStale(latest.current!==submitted.current);
   setStatus(data.error?"À corriger":data.checks?.length?"Vérification terminée":"Exécution terminée");
  };
  w.onerror=()=>{stop("Python n’a pas pu démarrer.");setResult({error:"Vérifie ta connexion puis relance. Le moteur Python est téléchargé au premier lancement."});};
  return w;
 }
 function run(){
  if(!loaded||active.current)return;
  active.current=true;submitted.current=code;setBusy(true);setResult(null);setSyntax(null);setStale(false);
  setStatus("Chargement de Python et des bibliothèques…");
  const current=++runId.current;
  try{getWorker().postMessage({type:"run",id:current,code,starterUrl});}
  catch(e){stop();setResult({error:(e as Error).message});return;}
  timer.current=setTimeout(()=>{stop("Délai dépassé (180 s).");setResult({error:"Le chargement ou le calcul a dépassé 180 secondes. Vérifie ta connexion, tes boucles et la taille des données, puis relance."});},180000);
 }
 function change(value:string){setCode(value);setSyntax(null);syntaxId.current++;if(result)setStale(true);}
 function download(){
  const url=URL.createObjectURL(new Blob([code],{type:"text/x-python;charset=utf-8"}));
  const link=document.createElement("a");link.href=url;link.download=starterUrl?.split("/").pop()?.replace("_starter","_mon_code")||lessonId+".py";link.click();URL.revokeObjectURL(url);
 }
 async function showSolution(){
  if(solution!==null){setSolution(null);return;}
  if(!solutionUrl)return;
  try{const response=await fetch(solutionUrl);if(!response.ok)throw Error();setSolution(await response.text());setSolutionError("");}
  catch{setSolutionError("Le corrigé n’est pas disponible. Réessaie dans quelques instants.");}
 }
 const todos=code.split("\n").filter(line=>/^\s*#.*\bTODO\b/i.test(line)).length;
 const badLine=!stale?(syntax?.line||result?.line):syntax?.line;
 return <section className="python-workshop" aria-label="Atelier Python">
  <header className="python-heading"><div><span className="eyebrow">TON ATELIER PYTHON</span><h3><Code2 size={21}/>Complète les TODO et teste ton code</h3></div><span className="python-language">Python · {todos} TODO</span></header>
  <p>Écris dans le fichier ci-dessous, puis lance-le pour voir les résultats et les erreurs. Après le premier lancement, la syntaxe est vérifiée pendant la saisie.</p>
  <p className="python-help">Tab : indenter · Ctrl / ⌘ + Entrée : exécuter · Code enregistré dans ton compte. Le premier chargement nécessite une connexion et peut prendre une minute.</p>
  {loadError&&<p role="alert">{loadError} Recharge la page pour réessayer.</p>}
  <label className="sr-only" htmlFor={id}>Code Python à compléter</label>
  <div className="python-editor">
   <div className="python-lines" aria-hidden="true" ref={gutter}>{code.split("\n").map((_,i)=><div key={i} className={badLine===i+1?"python-bad-line":""}>{i+1}</div>)}</div>
   <textarea id={id} ref={editor} value={code} disabled={!loaded} spellCheck={false} autoCapitalize="off" autoCorrect="off" wrap="off" maxLength={100000} aria-describedby={id+"-syntax"} onChange={e=>change(e.target.value)} onScroll={e=>{if(gutter.current)gutter.current.scrollTop=e.currentTarget.scrollTop;}} onKeyDown={e=>{
    if((e.ctrlKey||e.metaKey)&&e.key==="Enter"){e.preventDefault();run();}
    if(e.key==="Tab"){e.preventDefault();const el=e.currentTarget,a=el.selectionStart,b=el.selectionEnd;change(code.slice(0,a)+"    "+code.slice(b));requestAnimationFrame(()=>{el.selectionStart=el.selectionEnd=a+4;});}
   }}/>
  </div>
  <div id={id+"-syntax"} className="python-syntax" aria-live="polite">{syntax?.error?<span className="python-error">Ligne {syntax.line} : {syntax.error}</span>:syntax?"Syntaxe valide. Lance les tests pour vérifier les résultats.":"Les lignes numérotées t’aident à retrouver une erreur."}</div>
  <div className="python-toolbar"><button className="button-dark" disabled={!loaded||busy} onClick={run}><Play size={16}/>{busy?"Exécution…":"Exécuter et vérifier"}</button>{busy&&<button type="button" onClick={()=>stop()}><Square size={15}/>Arrêter</button>}<button type="button" disabled={!loaded||busy} onClick={()=>{if(window.confirm("Remettre le code de départ ? Ton code actuel sera remplacé.")){change(starter);setResult(null);}}}><RotateCcw size={15}/>Recommencer</button><button type="button" disabled={!loaded} onClick={download}><Download size={15}/>Mon fichier .py</button></div>
  <div className="python-status" role="status">{status}{stale&&" · Résultats du lancement précédent : relance après tes modifications."}</div>
  {result&&<div className="python-results">
   {result.error&&<div className="python-error" role="alert"><strong>{result.line?"Erreur à la ligne "+result.line:"Erreur Python"}</strong><pre>{result.error}</pre></div>}
   {result.checks&&result.checks.length>0&&<div><strong>{result.checks.filter(c=>c.passed).length}/{result.checks.length} tests réussis</strong><ul>{result.checks.map(c=><li key={c.name} className={c.passed?"python-pass":"python-error"}>{c.passed?"✓":"✗"} {c.name}{c.detail&&<pre>{c.detail}</pre>}</li>)}</ul></div>}
   {!result.error&&!result.checks?.length&&<p>Le script s’exécute sans erreur. Cet exercice n’a pas encore de tests automatiques : vérifie les résultats avec la consigne et le corrigé.</p>}
   <strong>Console</strong><pre className="python-console">{result.output||"Aucune sortie. Utilise print(...) pour observer tes résultats."}</pre>
   {result.figures?.map((src,i)=><figure key={i}><img src={src} alt={"Graphique produit par ton code, numéro "+(i+1)}/></figure>)}
  </div>}
  {solutionUrl&&<button className="python-solution-button" type="button" onClick={()=>void showSolution()}>{solution===null?"Voir le corrigé pour comparer":"Masquer le corrigé"}</button>}
  {solutionError&&<p role="alert">{solutionError}</p>}
  {solution!==null&&<div className="python-reference"><strong>Corrigé de référence</strong><p>Compare les étapes et les résultats avec ton essai.</p><pre><code>{solution}</code></pre></div>}
 </section>;
}
