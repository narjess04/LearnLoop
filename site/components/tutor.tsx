"use client";
import { useEffect, useRef, useState } from "react";
import { ArrowUp, LoaderCircle, MessageCircle, RotateCcw, Sparkles } from "lucide-react";
type Message = { role: "user" | "assistant"; content: string };
export default function Tutor({ lessonId }: { lessonId?: string }) {
  const [messages,setMessages]=useState<Message[]>([]);
  const [question,setQuestion]=useState("");
  const [model,setModel]=useState("deepseek");
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState("");
  const [cost,setCost]=useState<number|null>(null);
  const [configured,setConfigured]=useState<boolean|null>(null);
  const end=useRef<HTMLDivElement>(null);
  useEffect(()=>{let active=true;fetch("/api/tutor").then(r=>r.ok?r.json():Promise.reject()).then(d=>{if(active)setConfigured(Boolean((d as {configured?:boolean}).configured));}).catch(()=>{if(active)setError("Impossible de joindre le tuteur. Recharge la page ou vérifie ta connexion.");});return()=>{active=false;};},[]);
  useEffect(()=>{if(messages.length||loading)end.current?.scrollIntoView({block:"nearest"});},[messages,loading]);
  async function send(text: string) {
    if(loading||!text.trim())return;
    const user: Message={role:"user",content:text.trim()};
    const history=[...messages,user];
    setLoading(true);setError("");setMessages(history);setQuestion("");
    try{
      const response=await fetch("/api/tutor",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model,lessonId,messages:history.slice(-5)}),signal:AbortSignal.timeout(70000)});
      const result=await response.json() as {answer?:string;error?:string;cost:number|null;truncated?:boolean};
      if(!response.ok||!result.answer)throw new Error(result.error||"Le tuteur n’a pas pu répondre.");
      setMessages([...history,{role:"assistant",content:result.answer+(result.truncated?"\n\nLa réponse a atteint la limite de longueur. Demande une suite pour continuer.":"")}]);
      setCost(result.cost);
    }catch(e){setError(e instanceof Error?e.message:"La demande a échoué.");setQuestion(text);setMessages(history.slice(0,-1));}
    finally{setLoading(false);}
  }
  return <section className="tutor-panel" aria-label="Tuteur pédagogique">
    <div className="tutor-header"><div><span className="tutor-badge"><Sparkles size={15}/>TON PARTENAIRE DE RÉFLEXION</span><h2>Une question ? On la travaille.</h2></div><button className="icon-button" aria-label="Nouvelle conversation" disabled={loading} onClick={()=>{setMessages([]);setCost(null);setError("");}}><RotateCcw size={18}/></button></div>
    <label className="tutor-model">Modèle<select value={model} disabled={loading} onChange={e=>setModel(e.target.value)}><option value="deepseek">DeepSeek V3.2 · code & maths</option><option value="economy">Gemini Flash-Lite · économique</option></select></label>
    <div className="tutor-conversation" aria-live="polite" aria-busy={loading}>
      {!messages.length&&<div className="tutor-welcome"><MessageCircle size={30}/><h3>Comprendre, pas simplement obtenir la réponse.</h3><p>{lessonId?"Je reçois un extrait de cette leçon pour t’aider à raisonner.":"Pose une question sur Python, les maths, le ML ou ton projet."}</p><div className="tutor-prompts">{["Explique-moi l’idée avec un exemple simple.","Propose-moi un exercice progressif.","Quelles erreurs dois-je éviter ?"].map(q=><button key={q} disabled={loading||configured===false} onClick={()=>send(q)}>{q}</button>)}</div></div>}
      {messages.map((m,i)=><div key={i} className={"chat-message "+m.role}><span>{m.role==="user"?"TOI":"TUTEUR"}</span><div>{m.content}</div></div>)}
      {loading&&<div className="tutor-thinking" role="status"><LoaderCircle size={17}/>Le tuteur prépare une explication…</div>}
      <div ref={end}/>
    </div>
    {configured===false&&<p className="tutor-error" role="status">Le tuteur est préparé. Sa clé serveur OpenRouter doit être configurée sur cet environnement.</p>}
    {error&&<p className="tutor-error" role="alert">{error}</p>}
    <form className="tutor-input" onSubmit={e=>{e.preventDefault();send(question);}}><textarea aria-label="Ta question au tuteur" rows={3} maxLength={3000} value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Qu’est-ce qui te bloque ?" disabled={loading}/><button aria-label="Envoyer ma question" disabled={!question.trim()||loading||configured===false}>{loading?<LoaderCircle size={18}/>:<ArrowUp size={19}/>}</button></form>
    <div className="tutor-footnote"><span>1 000 tokens maximum par réponse{cost!==null?" · Dernier coût : "+cost.toFixed(6)+" $":""}</span><p>Ta question, les derniers échanges et un extrait du cours sont transmis à OpenRouter et au fournisseur choisi. Les réponses peuvent contenir des erreurs. Ne partage pas de secrets.</p><a href="/signin-with-chatgpt?return_to=%2Ftuteur" target="_top">Se connecter avec ChatGPT si nécessaire ↗</a></div>
  </section>;
}
