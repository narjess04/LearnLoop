"use client";
import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, ArrowRight, Check, LockKeyhole, Play, BookOpen, Flag } from "lucide-react";
import { mlChapters } from "@/lib/ml-roadmap";
import { chapterCourses, blockingLesson, blockingChapter } from "@/lib/learning-paths";
export function ChapterPath({number,passed}:{number:number;passed:string[]}){
 const chapter=mlChapters.find(c=>c.number===number)!;
 const nodes=chapterCourses(number);
 const lockedBy=blockingChapter(number,passed);
 const done=nodes.filter(n=>n.id&&passed.includes(n.id)).length;
 const total=nodes.length;
 const minutes=nodes.reduce((s,n)=>s+n.minutes,0);
 if(lockedBy)return <section className="course-access-lock"><LockKeyhole size={28}/><h2>Chapitre {number} verrouillé</h2><p>Termine tous les cours du chapitre {lockedBy.number} — {lockedBy.title} — pour ouvrir ce chapitre.</p><Link className="button-dark" href={"/parcours/machine-learning?chapitre="+lockedBy.number}>Reprendre le chapitre précédent →</Link></section>;
 return <section className="chapter-path"><div className="chapter-path-heading"><div><span className="eyebrow">CHAPITRE {String(number).padStart(2,"0")}</span><h2>{chapter.title}</h2><p>{chapter.description}</p></div><span className="chapter-progress-badge">{done}/{total||"—"} cours terminés</span></div>{total>0&&<><div className="chapter-overview"><span><BookOpen size={16}/>{total} cours{number===4?" + 1 mini-projet":""}</span><span>{Math.floor(minutes/60)} h {String(minutes%60).padStart(2,"0")} indicatives{number===4?", hors exercices":""}</span><span>{done===0?"À commencer":done===total?"Terminé":"En cours"}</span></div><div className="chapter-track-progress"><i style={{width:(done/total*100)+"%"}}/></div><ol className="course-path">{nodes.map((node,i)=>{
 const completed=!!node.id&&passed.includes(node.id);
 const blocker=node.id?blockingLesson(node.id,passed):undefined;
 const locked=!!blocker;
 const available=!!node.id&&!locked;
 const current=available&&!completed;
 const content=<><span className="course-path-marker">{completed?<Check size={21}/>:locked||!node.id?<LockKeyhole size={18}/>:<Play size={19}/>}</span><div><small>COURS {String(i+1).padStart(2,"0")} · {node.minutes} MIN</small><h3>{node.title}</h3><p>{node.summary}</p><span className="course-path-state">{completed?"Terminé · revoir le cours":!node.id?"À venir":locked?"Termine d’abord : "+blocker?.title:"Prochain cours · commencer"}</span></div>{available&&<ArrowRight size={19}/>}</>;
 return <li key={node.title} className={(completed?"completed ":current?"current ":"locked ")+(!node.id?"pending":"")}>{available?<Link href={"/cours/"+node.id}>{content}</Link>:<div className="course-path-disabled" aria-disabled="true">{content}</div>}</li>;
 })}</ol></>}{!total&&<div className="chapter-pending"><LockKeyhole size={24}/><p>Le programme et les supports de ce chapitre seront ajoutés avec les prochains cours fournis.</p></div>}{number===4&&<Link className="chapter-project" href="/?projet=customer-segmentation-anomaly"><Flag size={25}/><div><span className="eyebrow">MINI-PROJET · À RÉALISER APRÈS LES 10 COURS</span><h3>Customer Segmentation &amp; Anomaly Detection System</h3><p>Clustering, PCA, détection d’anomalies et visualisations. Consulter le cahier des charges.</p></div><ArrowRight size={19}/></Link>}{number===2&&<Link className="chapter-project" href="/?projet=data-preparation-pipeline"><Flag size={25}/><div><h3>Data Preparation Pipeline</h3><p>Consulter le projet de synthèse du chapitre.</p></div><ArrowRight size={19}/></Link>}</section>;
}
export default function LearningChapters({passed,query=""}:{passed:string[];query?:string}){
 const [selected,setSelected]=useState<number|null>(null);
 const relevant=mlChapters.filter(c=>(c.title+" "+chapterCourses(c.number).map(n=>n.title+" "+n.summary).join(" ")).toLocaleLowerCase("fr").includes(query.toLocaleLowerCase("fr")));
 if(selected!==null)return <div className="chapter-browser"><button className="text-button" onClick={()=>setSelected(null)}><ArrowLeft size={16}/>Tous les chapitres</button><ChapterPath number={selected} passed={passed}/></div>;
 return <section className="chapter-browser"><div className="section-title"><div><h2>Machine Learning · par chapitre</h2><p>Ouvre un chapitre pour voir ta roadmap. La réussite d’un quiz débloque le cours suivant.</p></div><Link className="text-button" href="/parcours/machine-learning">Vue d’ensemble <ArrowRight size={16}/></Link></div><div className="chapter-cards">{relevant.map(chapter=>{
 const nodes=chapterCourses(chapter.number),done=nodes.filter(n=>n.id&&passed.includes(n.id)).length,available=nodes.filter(n=>n.id).length;
 return <button key={chapter.number} className={"chapter-card "+(done&&done===nodes.length?"completed":"")} disabled={!!blockingChapter(chapter.number,passed)} title={blockingChapter(chapter.number,passed)?"Termine le chapitre "+blockingChapter(chapter.number,passed)?.number:undefined} onClick={()=>setSelected(chapter.number)}><span className="chapter-card-number">{String(chapter.number).padStart(2,"0")}</span><div><span className="eyebrow">CHAPITRE {chapter.number}</span><h3>{chapter.title}</h3><p>{chapter.description}</p><small>{blockingChapter(chapter.number,passed)?"Verrouillé · termine le chapitre "+blockingChapter(chapter.number,passed)?.number:nodes.length?done+"/"+nodes.length+" terminés · "+available+" disponibles":"Programme à venir"}</small></div><ArrowRight size={19}/><div className="chapter-card-progress"><i style={{width:(nodes.length?done/nodes.length*100:0)+"%"}}/></div></button>;
 })}</div>{!relevant.length&&<p className="empty-state">Aucun chapitre ne correspond à ta recherche.</p>}</section>;
}
