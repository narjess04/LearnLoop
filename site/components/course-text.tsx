import type { ReactNode } from "react";
import katex from "katex";
export function headingId(text:string){return text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g,"").replace(/[^a-z0-9]+/g,"-").replace(/^-|-$/g,"");}
function Formula({expression,display=false}:{expression:string;display?:boolean}){
 const html=katex.renderToString(expression,{displayMode:display,throwOnError:false,trust:false,maxExpand:1000});
 return display?<div className="rendered-equation" dangerouslySetInnerHTML={{__html:html}}/>:<span dangerouslySetInnerHTML={{__html:html}}/>;
}
function inline(text:string):ReactNode[]{return text.split(/(\\\([\s\S]*?\\\)|\*\*[^*]+\*\*|\x60[^\x60]+\x60)/g).map((s,i)=>s.startsWith("\\(")?<Formula key={i} expression={s.slice(2,-2)}/>:s.startsWith("**")?<strong key={i}>{s.slice(2,-2)}</strong>:s.charCodeAt(0)===96?<code key={i}>{s.slice(1,-1)}</code>:s);}
export default function CourseText({text}:{text:string}){
 return <div className="course-prose">{text.split(/((?:~~~|\x60{3})[^\n]*\n[\s\S]*?(?:~~~|\x60{3})|\\\[[\s\S]*?\\\])/g).map((part,i)=>{
 if(part.startsWith("\\["))return <Formula key={i} expression={part.slice(2,-2)} display/>;
 if(part.startsWith("~~~")||part.startsWith(String.fromCharCode(96).repeat(3)))return <pre key={i}><code>{part.replace(/^(?:~~~|\x60{3})[^\n]*\n/,"").replace(/(?:~~~|\x60{3})$/,"").trimEnd()}</code></pre>;
 return part.trim().split(/\n\s*\n/).filter(Boolean).map((block,j)=>{
 const key=i+"-"+j;
 if(/^#{1,3} /.test(block)){const title=block.replace(/^#+ /,"").trim();return <h2 id={headingId(title)} key={key}>{inline(title)}</h2>;}
 if(block.includes("\t")){const rows=block.split("\n").filter(Boolean).map(r=>r.split("\t"));return <div className="course-table-wrap" key={key}><table><thead><tr>{rows[0].map((cell,k)=><th key={k}>{inline(cell)}</th>)}</tr></thead><tbody>{rows.slice(1).map((row,k)=><tr key={k}>{row.map((cell,m)=><td key={m}>{inline(cell)}</td>)}</tr>)}</tbody></table></div>;}
 if(/^[-*] /.test(block))return <ul key={key}>{block.split("\n").map((s,k)=><li key={k}>{inline(s.replace(/^[-*] /,""))}</li>)}</ul>;
 if(/^\d+[.)] /.test(block))return <ol key={key}>{block.split("\n").map((s,k)=><li key={k}>{inline(s.replace(/^\d+[.)] /,""))}</li>)}</ol>;
 return <p key={key}>{inline(block)}</p>;
 });
 })}</div>;
}
