"use client";
import { blockingLesson } from "@/lib/learning-paths";
import LearnLoopLogo from "@/components/learnloop-logo";

import { completeQuiz } from "@/lib/account-storage";
import { accountStorage } from "@/lib/account-storage";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, BookOpen, Check, Clock3, MessageCircle, Target } from "lucide-react";
import { lessons, modules, type Lesson } from "@/lib/curriculum";
import { gradientCourse } from "@/lib/pilot";
import { pythonCourse } from "@/lib/full-python";
import { toolsCourse } from "@/lib/full-tools";
import PythonWorkshop from "@/components/python-workshop";
import Tutor from "@/components/tutor";
import CourseText, { headingId } from "@/components/course-text";
import MLIntroCourse, { MLInterview } from "@/components/ml-intro-course";
import { mlIntroHeadings } from "@/lib/ml-introduction";
import MLSuppliedCourse, { type SuppliedCourseId } from "@/components/ml-supplied-course";
import { MLCourseInterview } from "@/components/ml-course-widgets";
import chapterOne from "@/lib/ml-chapter-one.json";
import MLDataCourse, { DataInterview, DataQuiz, type DataCourseId } from "@/components/ml-data-course";
import { richCourses, richCourseChapter } from "@/lib/ml-rich-courses";
const STORAGE="recap-learning-v1";
type Progress={passed:string[];steps:string[];notes:Record<string,string>;dates:string[]};
const initial:Progress={passed:[],steps:[],notes:{},dates:[]};
export default function LessonReader({lesson}:{lesson:Lesson}){
 const [tab,setTab]=useState("Cours");
 const [progress,setProgress]=useState<Progress>(initial);
 const [ready,setReady]=useState(false);
 const [error,setError]=useState("");
 const [choice,setChoice]=useState<number|null>(null);
 const [feedback,setFeedback]=useState(false);
 useEffect(()=>{
  try{const data=JSON.parse(accountStorage.getItem(STORAGE)||"null");if(data&&Array.isArray(data.passed)&&Array.isArray(data.steps)&&Array.isArray(data.dates)&&data.notes&&typeof data.notes==="object")setProgress({...data,passed:data.passed.filter((id:string)=>lessons.some(l=>l.id===id))});}catch{setError("La progression n’a pas pu être lue.");}
  const requested=new URLSearchParams(window.location.search).get("section");
  if(requested&&["Cours","Exercices","Quiz","Questions"].includes(requested))setTab(requested);
  setReady(true);
 },[]);
 useEffect(()=>{if(ready)try{accountStorage.setItem(STORAGE,JSON.stringify(progress));}catch{setError("L’enregistrement du compte a échoué. Copie tes notes avant de quitter.");}},[progress,ready]);
 const dataId=lesson.id in richCourses?lesson.id as DataCourseId:null;
 const dataCourse=dataId?richCourses[dataId]:null;
 const isData=!!dataId;
 const dataChapter=dataId?richCourseChapter(dataId):null;
 const isMLIntro=lesson.id==="ml-introduction";
 const suppliedId=lesson.id in chapterOne.courses?lesson.id as SuppliedCourseId:null;
 const mlNumber=dataCourse?dataCourse.number:lessons.filter(l=>l.module==="ml").findIndex(l=>l.id===lesson.id)+1;
 const fullText=dataCourse?dataCourse.text:suppliedId?chapterOne.courses[suppliedId]:lesson.id==="python-foundations"?pythonCourse:lesson.id==="tools-foundations"?toolsCourse:lesson.id==="gradient"?gradientCourse:null;
 const headings=isMLIntro?mlIntroHeadings:fullText?Array.from(fullText.replace(/~~~[\s\S]*?~~~|\x60{3}[\s\S]*?\x60{3}/g, "").matchAll(/^#{1,3} (.+)$/gm)).map(m=>m[1]):["L’intuition","Le raisonnement","L’exemple","Le code","Les pièges"];
 const index=lessons.findIndex(l=>l.id===lesson.id);
 const next=lessons.slice(index+1).find(l=>l.module===lesson.module);
 const module=modules.find(m=>m.id===lesson.module);
 function note(key:string,value:string){setProgress(p=>({...p,notes:{...p.notes,[key]:value}}));}
 async function passDataQuiz(answers:number[]){try{const result=await completeQuiz(lesson.id,answers);setProgress(p=>({...p,passed:[...new Set([...p.passed,lesson.id])],dates:[...new Set([...p.dates,result.date])]}));}catch(e){setError((e as Error).message);}}
 function validate(){setFeedback(true);if(choice===lesson.answer && choice!==null)void passDataQuiz([choice]);}
 return <div className="reader-shell">
  <header className="reader-top"><Link className="reader-brand" href="/"><LearnLoopLogo/></Link><Link href="/"><ArrowLeft size={16}/>Mon espace</Link><button className="reader-tutor-trigger" onClick={()=>window.dispatchEvent(new Event("learnloop-open-tutor"))}><MessageCircle size={17}/>Questionner le tuteur</button></header>
  <div className="reader-banner"><div><div className="eyebrow">LE PARCOURS · {module?.name}</div><h1>{lesson.title}</h1><p className="reader-intro">{lesson.goal}</p><div className="reader-meta"><span><Clock3 size={16}/>{dataCourse?dataCourse.duration+" indicatives":isMLIntro?"60–90 min indicatives":suppliedId?chapterOne.config[suppliedId].duration+" indicatives":["python-foundations","tools-foundations"].includes(lesson.id)?"Module à étudier sur plusieurs séances":lesson.minutes+" min indicatives"}</span><span><BookOpen size={16}/>{lesson.module==="ml"?(dataChapter?"Chapitre "+String(dataChapter).padStart(2,"0")+" · Cours ":"Chapitre 01 · Cours ")+String(mlNumber).padStart(2,"0"):fullText?"Support approfondi":"Leçon introductive"}</span>{progress.passed.includes(lesson.id)&&<span><Check size={16}/>Quiz déjà réussi</span>}</div></div><div className="reader-number">{lesson.module==="ml"?String(mlNumber).padStart(2,"0"):String(index+1).padStart(2,"0")}<span>COMPRENDRE<br/>PUIS CONSTRUIRE</span></div></div>
  <div className="reader-layout"><aside className="reader-outline"><span className="eyebrow">DANS CETTE LEÇON</span><nav aria-label="Sommaire du cours">{headings.map((h,i)=><a key={i} href={"#"+headingId(h)} onClick={()=>setTab("Cours")}><span>{String(i+1).padStart(2,"0")}</span>{h}</a>)}</nav><div className="reader-tip"><Target size={22}/><strong>Avant de lire la solution</strong><p>Formule une hypothèse, tente un calcul ou écris quelques lignes. Puis compare ton raisonnement.</p></div><Link className="text-button" href="/tuteur">Ouvrir le tuteur seul ↗</Link></aside>
  <main className="reader-main">
   <div className="reader-tabs" role="tablist" aria-label="Sections de la leçon">{["Cours","Exercices","Quiz","Questions"].map(t=><button key={t} role="tab" aria-selected={tab===t} onClick={()=>setTab(t)}>{t}</button>)}</div>
   {error&&<p className="storage-alert" role="status">{error}</p>}
   <section className="reader-content" role="tabpanel" aria-label={tab}>
   {tab==="Cours"&&<>{dataId?<MLDataCourse courseId={dataId} onPassed={passDataQuiz}/>:isMLIntro?<MLIntroCourse/>:suppliedId?<MLSuppliedCourse courseId={suppliedId}/>:fullText?<CourseText text={fullText}/>:<div className="course-prose"><div className="reader-status">Cette fiche introduit la notion. Le développement complet, les démonstrations et plusieurs exercices restent à produire.</div><h2 id={headingId("L’intuition")}>L’intuition</h2><p>{lesson.intuition}</p><h2 id={headingId("Le raisonnement")}>Le raisonnement</h2><p>{lesson.theory}</p>{lesson.id==="backprop"&&<figure className="teaching-figure"><img src="/images/neural-network.png" alt="Réseau de neurones avec une couche d’entrée, une couche cachée et une sortie scalaire." width={783} height={851}/><figcaption>Lire de gauche à droite : les activations se propagent vers la sortie ; la rétropropagation calcule les sensibilités en sens inverse. Ce schéma montre l’architecture, pas une mise à jour des poids. <a href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html" target="_blank" rel="noreferrer">Documentation scikit-learn, licence BSD</a>.</figcaption></figure>}<h2 id={headingId("L’exemple")}>L’exemple</h2><p>{lesson.example}</p><h2 id={headingId("Le code")}>Le code</h2><pre><code>{lesson.code}</code></pre><p className="code-note">Essaie ce code dans l’atelier Python de l’onglet Exercices.</p>{lesson.id==="numpy"&&<figure className="teaching-figure"><img src="/images/numpy-broadcasting.png" alt="Un scalaire est diffusé pour multiplier chaque élément d’un vecteur NumPy." width={640} height={310}/><figcaption>Un scalaire participe au calcul sur chaque élément. La répétition dessinée est conceptuelle ; elle ne nécessite pas une copie complète du scalaire. <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html" target="_blank" rel="noreferrer">Documentation NumPy · licence BSD</a>.</figcaption></figure>}<h2 id={headingId("Les pièges")}>Les pièges</h2><ul>{lesson.pitfalls.map(p=><li key={p}>{p}</li>)}</ul></div>}<a className="source-link" href={lesson.source} target="_blank" rel="noreferrer">Consulter la documentation de référence ↗</a><button className="button-dark" onClick={()=>setTab("Exercices")}>Passer à la pratique<ArrowRight size={17}/></button></>}
   {tab==="Exercices"&&<div className="course-prose"><span className="eyebrow">ESSAYER · OBSERVER · EXPLIQUER</span><h2>À toi de construire le raisonnement.</h2><p>{lesson.exercise}</p>{lesson.id!=="tools-foundations"&&<PythonWorkshop key={lesson.id} lessonId={lesson.id} starterUrl={dataCourse?"/downloads/ch"+String(dataChapter).padStart(2,"0")+"_cours"+String(dataCourse.number).padStart(2,"0")+(dataChapter===5&&dataCourse.number===3?"_browser_starter.py":"_starter.py"):suppliedId?chapterOne.config[suppliedId].starter:undefined} solutionUrl={dataCourse?"/downloads/ch"+String(dataChapter).padStart(2,"0")+"_cours"+String(dataCourse.number).padStart(2,"0")+(dataChapter===5&&dataCourse.number===3?"_browser_solution.py":"_solution.py"):undefined} initialCode={dataCourse||suppliedId?undefined:lesson.code}/>}<label className="field-label" htmlFor="exercise">Ton raisonnement et tes observations</label><textarea id="exercise" rows={10} value={progress.notes[lesson.id+"-exercise"]||""} onChange={e=>note(lesson.id+"-exercise",e.target.value)} placeholder="D’abord mon hypothèse, ensuite ma vérification…"/><details className="correction"><summary>Ouvrir la correction expliquée</summary>{dataCourse?<CourseText text={"~~~python\n"+dataCourse.solution+"\n~~~"}/>:<p>{lesson.solution}</p>}</details><h2>Une variante pour vérifier le transfert</h2><p>Change une entrée ou une contrainte, prédis le résultat, puis vérifie. Explique ce qui reste vrai et ce qui doit être adapté. Le support approfondi contient les exercices supplémentaires lorsqu’il est disponible.</p><button className="button-dark" onClick={()=>setTab("Quiz")}>Tester ma compréhension<ArrowRight size={17}/></button></div>}
   {tab==="Quiz"&&dataId&&<DataQuiz courseId={dataId} onPassed={passDataQuiz}/>}
   {tab==="Quiz"&&!isData&&<div className="course-prose"><span className="eyebrow">QUIZ FORMATIF · +100 XP À LA PREMIÈRE RÉUSSITE</span><h2>{lesson.question}</h2><div className="quiz-options">{lesson.options.map((o,i)=><button key={o} className={(choice===i?"chosen ":"")+(feedback&&i===lesson.answer?"correct":"")} onClick={()=>{setChoice(i);setFeedback(false);}}><span>{String.fromCharCode(65+i)}</span>{o}{feedback&&i===lesson.answer&&<Check size={20}/>}</button>)}</div><button className="button-dark" disabled={choice===null} onClick={validate}>Vérifier ma réponse<ArrowRight size={17}/></button>{feedback&&<div className="quiz-feedback" role="status"><strong>{choice===lesson.answer?"Le raisonnement est correct.":"Reprenons le raisonnement."}</strong><p>{lesson.explanation}</p><small>Une bonne réponse valide ce point précis. Les exercices et la revue du projet démontrent d’autres compétences.</small></div>}</div>}
   {tab==="Questions"&&<div className="course-prose"><h2>Les questions qui font progresser.</h2>{dataId?<DataInterview courseId={dataId}/>:suppliedId?<MLCourseInterview courseId={suppliedId}/>:isMLIntro?<MLInterview/>:[["Quel problème cette notion résout-elle ?",lesson.goal+" "+lesson.intuition],["Comment relier le concept à un exemple ?",lesson.example],["Quelles erreurs dois-je reconnaître ?",lesson.pitfalls.join(" ")],["Comment vérifier ma compréhension ?",lesson.exercise]].map(([q,a])=><details className="correction" key={q}><summary>{q}</summary><p>{a}</p></details>)}<p>Tu peux poursuivre avec une question personnelle au tuteur ci-dessous.</p></div>}
   </section>
   <div className="reader-next"><Link href="/"><ArrowLeft size={17}/>Tous les parcours</Link>{next&&blockingLesson(next.id,progress.passed)&&<span className="next-course-locked">Valide le quiz pour débloquer le cours suivant.</span>}{next&&!blockingLesson(next.id,progress.passed)&&<Link href={"/cours/"+next.id}><span>LEÇON SUIVANTE<strong>{next.title}</strong></span><ArrowRight size={20}/></Link>}</div>

  </main></div>
 </div>;
}
