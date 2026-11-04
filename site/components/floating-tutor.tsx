"use client";
import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { Bot, Maximize2, X } from "lucide-react";
import { lessons } from "@/lib/curriculum";
import Tutor from "@/components/tutor";
export default function FloatingTutor(){
 const [open,setOpen]=useState(false),[mounted,setMounted]=useState(false);
 const launcher=useRef<HTMLButtonElement>(null),panel=useRef<HTMLDivElement>(null);
 const pathname=usePathname();
 const lesson=lessons.find(l=>pathname==="/cours/"+l.id);
 function show(){setMounted(true);setOpen(true);}
 function close(){setOpen(false);launcher.current?.focus();}
 useEffect(()=>{const handler=()=>show();window.addEventListener("learnloop-open-tutor",handler);return()=>window.removeEventListener("learnloop-open-tutor",handler);},[]);
 useEffect(()=>{if(!open)return;const frame=requestAnimationFrame(()=>panel.current?.querySelector<HTMLTextAreaElement>("textarea")?.focus({preventScroll:true}));const escape=(e:KeyboardEvent)=>{if(e.key==="Escape"){setOpen(false);launcher.current?.focus();}};window.addEventListener("keydown",escape);return()=>{cancelAnimationFrame(frame);window.removeEventListener("keydown",escape);};},[open]);
 return <><button ref={launcher} className={"assistant-orb "+(open?"is-open":"")} aria-label={open?"Fermer l’assistant LearnLoop":"Questionner l’assistant LearnLoop"} title="Assistant LearnLoop" aria-expanded={open} aria-controls="learnloop-assistant" onClick={()=>open?close():show()}>{open?<X size={23}/>:<Bot size={27}/>}<span className="assistant-orb-dot"/></button>{mounted&&<div ref={panel} className="assistant-window" id="learnloop-assistant" role="dialog" aria-modal="false" aria-labelledby="assistant-title" hidden={!open}><header className="assistant-window-header"><span className="assistant-avatar"><Bot size={22}/></span><div><strong id="assistant-title">Assistant LearnLoop</strong><small>{lesson?"À propos de : "+lesson.title:"Python, maths et Machine Learning"}</small></div><a href="/tuteur" title="Ouvrir la page du tuteur" aria-label="Ouvrir la page du tuteur"><Maximize2 size={17}/></a><button aria-label="Fermer la discussion" onClick={close}><X size={19}/></button></header><Tutor key={lesson?.id||"general"} lessonId={lesson?.id} compact/></div>}</>;
}
