"use client";
import { useRef, useState, type ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Check, Copy } from "lucide-react";
function CodeBlock({children}:{children:ReactNode}){
 const ref=useRef<HTMLPreElement>(null),[copied,setCopied]=useState(false),[error,setError]=useState(false);
 return <div className="chat-code"><div className="chat-code-toolbar"><span>Code</span><button type="button" aria-label="Copier le code" onClick={async()=>{try{await navigator.clipboard.writeText(ref.current?.textContent||"");setCopied(true);setError(false);setTimeout(()=>setCopied(false),2000);}catch{setError(true);}}}>{copied?<Check size={14}/>:<Copy size={14}/>} {copied?"Copié":error?"Sélectionne le code":"Copier"}</button></div><pre ref={ref}>{children}</pre></div>;
}
export function normalizeTutorMath(text:string){
 // Préserver les blocs de code et le code en ligne.
 const ticks=String.fromCharCode(96);
 const pattern=new RegExp("("+ticks+"{3}[\\s\\S]*?"+ticks+"{3}|~~~[\\s\\S]*?~~~|"+ticks+"[^"+ticks+"\\n]+"+ticks+")","g");
 return text.split(pattern).map((part,i)=>i%2?part:part.replace(/\\\[([\s\S]*?)\\\]/g,(_,math)=>"\n\n$$\n"+math.trim()+"\n$$\n\n").replace(/\\\(([\s\S]*?)\\\)/g,(_,math)=>"$"+math+"$")).join("");
}
export default function ChatMarkdown({text}:{text:string}){
 return <div className="chat-markdown"><Markdown remarkPlugins={[remarkGfm,remarkMath]} rehypePlugins={[[rehypeKatex,{throwOnError:false,trust:false}]]} skipHtml components={{
 pre:({children})=><CodeBlock>{children}</CodeBlock>,
 table:({children})=><div className="chat-table"><table>{children}</table></div>,
 a:({href,children})=><a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
 img:({src,alt})=><a href={typeof src==="string"?src:undefined} target="_blank" rel="noopener noreferrer">{alt||"Voir l’image"}</a>
 }}>{normalizeTutorMath(text)}</Markdown></div>;
}
