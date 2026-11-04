"use client";
import {AccountContext,AccountActionsContext} from "@/lib/account-context";
import FloatingTutor from "@/components/floating-tutor";
import LearnLoopLogo from "@/components/learnloop-logo";

import { useEffect, useState } from "react";
import { initializeAccount, flushAccount, accountRequest, type User } from "@/lib/account-storage";
export default function AccountProvider({children}:{children:React.ReactNode}){
 const [user,setUser]=useState<User|null>(null),[loaded,setLoaded]=useState(false),[error,setError]=useState(""),[saving,setSaving]=useState(false),[failed,setFailed]=useState(false);
 const [mode,setMode]=useState<"login"|"register">("login"),[busy,setBusy]=useState(false);
 const [email,setEmail]=useState(""),[password,setPassword]=useState(""),[confirmation,setConfirmation]=useState(""),[signupName,setSignupName]=useState("");
 const [editing,setEditing]=useState(false),[name,setName]=useState("");
 async function load(){
  const response=await fetch("/api/account",{cache:"no-store"});
  if(response.status===401){initializeAccount({});setUser(null);setLoaded(true);return;}
  const data=await response.json() as {error?:string;values:Record<string,string>;user:User};
  if(!response.ok)throw new Error(data.error||"Chargement impossible.");
  initializeAccount(data.values,data.user.id);setUser(data.user);setName(data.user.name);setLoaded(true);
 }
 useEffect(()=>{
  void load().catch(e=>{setError(e.message);setLoaded(true);});
  const listener=(event:Event)=>{const detail=(event as CustomEvent).detail;setSaving(detail.pending>0);setFailed(detail.failed);};
  window.addEventListener("account-save",listener);
  return()=>window.removeEventListener("account-save",listener);
 },[]);
 useEffect(()=>{
  const leave=(e:BeforeUnloadEvent)=>{if(saving||failed){e.preventDefault();e.returnValue="";}};
  window.addEventListener("beforeunload",leave);return()=>window.removeEventListener("beforeunload",leave);
 },[saving,failed]);
 async function authenticate(e:React.FormEvent){
  e.preventDefault();setError("");
  if(mode==="register"&&password!==confirmation){setError("Les mots de passe ne correspondent pas.");return;}
  setBusy(true);
  try{
   await accountRequest("/api/account/auth/"+mode,{email,password,name:signupName});
   setPassword("");setConfirmation("");window.location.reload();
  }catch(e){setError((e as Error).message);}finally{setBusy(false);}
 }
 async function logout(){
  setError("");setBusy(true);
  try{await flushAccount();await accountRequest("/api/account/auth/logout",{});initializeAccount({});setUser(null);setEditing(false);setFailed(false);setSaving(false);}
  catch(e){setError((e as Error).message);}finally{setBusy(false);}
 }
 if(!loaded)return <main className="account-entry"><div className="account-card"><a href="/" className="reader-brand"><LearnLoopLogo/></a><p role="status">Chargement de ton espace…</p></div></main>;
 if(!user)return <main className="account-entry"><div className="account-card"><a href="/" className="reader-brand"><LearnLoopLogo/></a><span className="eyebrow">TON ESPACE PERSONNEL</span><div><h1>Ton parcours.<br/>Tes progrès.</h1><p>Un compte, ta progression et tes points. Retrouve ton apprentissage à chaque connexion.</p></div><div className="auth-tabs" aria-label="Accès au compte"><button className={mode==="login"?"selected":""} onClick={()=>{setMode("login");setError("");}}>Connexion</button><button className={mode==="register"?"selected":""} onClick={()=>{setMode("register");setError("");}}>Créer un compte</button></div><form className="auth-form" onSubmit={authenticate}>{mode==="register"&&<label htmlFor="signup-name">Ton nom<input id="signup-name" autoComplete="name" value={signupName} onChange={e=>setSignupName(e.target.value)} required maxLength={80}/></label>}<label htmlFor="account-email">Adresse email<input id="account-email" type="email" autoComplete="email" value={email} onChange={e=>setEmail(e.target.value)} required maxLength={254} placeholder="toi@exemple.com"/></label><label htmlFor="account-password">Mot de passe<input id="account-password" type="password" autoComplete={mode==="register"?"new-password":"current-password"} value={password} onChange={e=>setPassword(e.target.value)} required minLength={mode==="register"?10:1} maxLength={128}/></label>{mode==="register"&&<><small>Au moins 10 caractères.</small><label htmlFor="account-confirmation">Confirmer le mot de passe<input id="account-confirmation" type="password" autoComplete="new-password" value={confirmation} onChange={e=>setConfirmation(e.target.value)} required minLength={10} maxLength={128}/></label></>}{error&&<p className="auth-error" role="alert">{error}</p>}<button className="button-dark" disabled={busy}>{busy?"Un instant…":mode==="register"?"Créer mon compte":"Me connecter"}<span aria-hidden="true">→</span></button></form><small>+100 XP à la première réussite de chaque quiz.</small></div></main>;
 return <AccountContext.Provider value={user}><AccountActionsContext.Provider value={{edit:()=>setEditing(true),logout:()=>void logout(),disabled:busy||saving,saving,failed}}>{error&&!editing&&<p className="storage-alert" role="alert">{error}</p>}{editing&&<form className="account-settings" onSubmit={async e=>{e.preventDefault();setError("");try{await accountRequest("/api/account/profile",{name});setUser({...user,name:name.trim()});setEditing(false);}catch(e){setError((e as Error).message);}}}><label>Ton nom<input value={name} onChange={e=>setName(e.target.value)} required maxLength={80}/></label><span>{user.email}</span><button className="button-dark">Enregistrer</button><button type="button" onClick={()=>setEditing(false)}>Fermer</button>{error&&<p role="alert">{error}</p>}</form>}{children}<FloatingTutor/></AccountActionsContext.Provider></AccountContext.Provider>;
}
