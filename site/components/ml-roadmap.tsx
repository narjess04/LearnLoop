"use client";
import { blockingChapter } from "@/lib/learning-paths";
import { ChapterPath } from "@/components/learning-chapters";
import LearnLoopLogo from "@/components/learnloop-logo";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowRight, BookOpen, Check, ChevronRight, Compass } from "lucide-react";
import { mlChapters } from "@/lib/ml-roadmap";
import { lessons } from "@/lib/curriculum";
import { accountStorage } from "@/lib/account-storage";
import chapterFour from "@/lib/ml-chapter-four.json";
import chapterThree from "@/lib/ml-chapter-three.json";
import chapterTwo from "@/lib/ml-chapter-two.json";
import chapterOne from "@/lib/ml-chapter-one.json";
export default function MLRoadmap(){
 const [selected,setSelected]=useState(1);
 const [passed,setPassed]=useState<string[]>([]);
 useEffect(()=>{try{setPassed(JSON.parse(accountStorage.getItem("recap-learning-v1")||"{}").passed||[]);}catch{}},[]);
 useEffect(()=>{const requested=Number(new URLSearchParams(window.location.search).get("chapitre"));if(mlChapters.some(c=>c.number===requested))setSelected(requested);},[]);
 function selectChapter(number:number){setSelected(number);if(window.matchMedia("(max-width: 1050px)").matches)requestAnimationFrame(()=>document.getElementById("chapter-detail")?.scrollIntoView({behavior:"smooth",block:"start"}));}
 const chapter=mlChapters.find(c=>c.number===selected)!;
 const program=selected===2?chapterTwo:selected===3?chapterThree:null;
 const available=chapter.lessonIds.map(id=>lessons.find(l=>l.id===id)).filter(l=>!!l);
 return <div className="reader-shell ml-roadmap-page"><header className="reader-top"><Link href="/" className="reader-brand"><LearnLoopLogo/></Link><Link href="/">← Mon espace</Link><Link href="/cours/ml-introduction">Commencer le cours 1<ArrowRight size={16}/></Link></header><div className="roadmap-page-heading"><span className="eyebrow"><Compass size={16}/>MACHINE LEARNING · TON PARCOURS EN 12 CHAPITRES</span><h1>Une direction claire.<br/><span>Un chapitre après l’autre.</span></h1><p>Suis le chemin des fondements aux systèmes ML. Sélectionne une étape pour découvrir son objectif et les cours disponibles.</p><div className="roadmap-legend"><span><i className="available-dot"/>Cours disponible</span><span><i/>À compléter avec les prochains supports</span></div></div>
 <main className="roadmap-schema-layout"><div className="roadmap-flow"><div className="roadmap-phases"><span>01–03 · LES FONDATIONS</span><span>04–08 · LES MODÈLES</span><span>09–12 · APPROFONDIR</span></div><div className="roadmap-canvas"><svg viewBox="0 0 1000 640" preserveAspectRatio="none" aria-hidden="true" className="roadmap-connector"><path d="M166 80 H833 Q972 80 972 160 Q972 240 833 240 H166 Q28 240 28 320 Q28 400 166 400 H833 Q972 400 972 480 Q972 560 833 560 H166" fill="none" stroke="#d3ddbe" strokeWidth="3" strokeDasharray="5 7"/><circle cx="166" cy="80" r="8" fill="#82995e"/><path d="m172 553-9 7 9 7" fill="none" stroke="#a7b78a" strokeWidth="3"/></svg><ol className="roadmap-nodes">{mlChapters.map(c=>{const row=Math.floor((c.number-1)/3);const col=(c.number-1)%3;return <li key={c.number} style={{gridRow:row+1,gridColumn:row%2?3-col:col+1}}><button disabled={!!blockingChapter(c.number,passed)} title={blockingChapter(c.number,passed)?"Termine le chapitre "+blockingChapter(c.number,passed)?.number:undefined} onClick={()=>selectChapter(c.number)} aria-pressed={selected===c.number} aria-controls="chapter-detail" className={(selected===c.number?"selected ":"")+(c.lessonIds.length?"available":"planned")}><span className="roadmap-node-number">{String(c.number).padStart(2,"0")}</span><strong>{c.title}</strong><span className="roadmap-node-status">{c.lessonIds.length?<><span/>{c.lessonIds.length} cours disponibles</>:<>À venir<ChevronRight size={12}/></>}</span></button></li>;})}</ol></div><p className="roadmap-flow-note"><BookOpen size={15}/>Les chapitres 1, 2 et 3 sont disponibles. Les dix cours du chapitre 4 sont disponibles, ainsi que le projet de synthèse.</p></div>
 <aside className="chapter-detail" id="chapter-detail" aria-live="polite"><ChapterPath key={selected} number={selected} passed={passed}/></aside>
 </main></div>;
}
