"use client";

import { useEffect, useState } from "react";
import { ArrowUpRight, ArrowRight, BookOpen, Code2, Check, ChevronRight, CircleHelp, Flame, FlaskConical, GraduationCap, LayoutDashboard, Library, Menu, Network, Play, Search, Sparkles, Target, Trophy, X, Zap, Clock3, Bookmark, CheckCircle2, RotateCcw, Video, Sigma, GitBranch } from "lucide-react";
import { lessons, modules, projects, type Lesson } from "@/lib/curriculum";
import { gradientCourse } from "@/lib/pilot";

type View = "dashboard" | "courses" | "projects" | "review" | "library";
type Progress = { passed: string[]; steps: string[]; notes: Record<string,string>; dates: string[] };
const empty: Progress = { passed: [], steps: [], notes: {}, dates: [] };
const icons = [Code2, Sigma, Network, GitBranch, Sparkles, FlaskConical];
const nextTopics = ["Python avancé · typage, POO, pytest", "SQL · fenêtres et modélisation", "Statistiques · estimation et A/B tests", "ML · ensembles et séries temporelles", "Vision · CNN et segmentation", "LLM · mini-GPT et post-training", "AI engineering · agents et évaluation", "MLOps · CI/CD, serving et monitoring"];
const day = () => new Date().toLocaleDateString("en-CA");
function Art({ kind = 0 }: { kind?: number }) {
  return <svg viewBox="0 0 360 170" className={"course-art art-" + kind} fill="none" aria-hidden="true">
    <defs><pattern id={"grid"+kind} width="22" height="22" patternUnits="userSpaceOnUse"><path d="M22 0H0V22" stroke="currentColor" strokeOpacity=".08"/></pattern></defs>
    <rect width="360" height="170" fill={"url(#grid"+kind+")"} />
    {kind === 0 ? <><g transform="translate(92 27) rotate(-7 80 60)"><rect x="0" y="0" width="176" height="116" rx="10" fill="#f8fbef" stroke="currentColor" strokeWidth="1.5"/><path d="M0 25H176" stroke="currentColor" strokeOpacity=".3"/><circle cx="14" cy="13" r="3" fill="currentColor"/><circle cx="25" cy="13" r="3" fill="currentColor" opacity=".4"/><path d="m48 52-16 15 16 15m75-30 16 15-16 15m-33-37-9 45" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/></g><path d="m286 43 4 9 10 2-10 4-4 10-2-10-10-4 10-2Z" fill="currentColor"/></> :
    kind === 1 ? <><path d="M73 131V34m0 97h220" stroke="currentColor" strokeOpacity=".4"/><path d="M86 115C124 114 133 60 169 76s56-39 105-38" stroke="currentColor" strokeWidth="2.5"/>{[[100,106],[122,92],[141,71],[162,84],[187,63],[209,66],[238,43],[264,50]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="4" fill="currentColor" opacity=".65"/>)}<rect x="196" y="101" width="87" height="29" rx="7" fill="#f9f5e9" stroke="currentColor" strokeOpacity=".4"/><text x="209" y="120" fill="currentColor" fontSize="14" fontFamily="monospace">ŷ = Xw+b</text></> :
    <>{[65,140,215,290].map((x,i)=><g key={x}>{[45,85,125].map((y,j)=><g key={y}>{i<3&&[45,85,125].map(z=><path key={z} d={"M"+x+" "+y+"L"+(x+75)+" "+z} stroke="currentColor" strokeOpacity=".19"/>)}<circle cx={x} cy={y} r={i===1||i===2?10:8} fill="#f6f2fc" stroke="currentColor" strokeWidth="1.7"/></g>)}</g>)}</>}
  </svg>;
}
export default function Home() {
  const [view, setView] = useState<View>("dashboard");
  const [progress, setProgress] = useState<Progress>(empty);
  const [loaded, setLoaded] = useState(false);
  const [storageError, setStorageError] = useState(false);
  const [selected, setSelected] = useState<Lesson | null>(null);
  const [tab, setTab] = useState("Cours");
  const [choice, setChoice] = useState<number | null>(null);
  const [feedback, setFeedback] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [mobile, setMobile] = useState(false);
  const [projectId, setProjectId] = useState<string | null>(null);
  useEffect(() => {
    try {
      const data = JSON.parse(localStorage.getItem("recap-learning-v1") || "null");
      if (data && Array.isArray(data.passed) && Array.isArray(data.steps) && Array.isArray(data.dates) && data.notes && typeof data.notes === "object") setProgress({...data, passed: data.passed.filter((id:string)=>lessons.some(l=>l.id===id))});
    } catch { setStorageError(true); }
    const requestedProject=new URLSearchParams(window.location.search).get("projet");
    if(requestedProject&&projects.some(p=>p.id===requestedProject))setProjectId(requestedProject);
    setLoaded(true);
  }, []);
  useEffect(() => {
    if (!loaded) return;
    try { localStorage.setItem("recap-learning-v1", JSON.stringify(progress)); } catch { setStorageError(true); }
  }, [progress, loaded]);
  useEffect(() => {
    if (!projectId) return;
    const previous = document.activeElement as HTMLElement | null;
    const old = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const modal = document.querySelector<HTMLElement>('[role="dialog"]');
    modal?.focus();
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setProjectId(null); }
      if (e.key === "Tab" && modal) {
        const controls = Array.from(modal.querySelectorAll<HTMLElement>('button, a[href], input, textarea, select, [tabindex="0"]')).filter(el => !el.hasAttribute("disabled") && el.getClientRects().length);
        const first = controls[0], last = controls[controls.length-1];
        if (e.shiftKey && (document.activeElement === first || document.activeElement === modal)) { e.preventDefault(); last?.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
      }
    };
    window.addEventListener("keydown", handler);
    return () => { document.body.style.overflow = old; window.removeEventListener("keydown", handler); previous?.focus(); };
  }, [projectId]);
  const openLesson = (l: Lesson, section = "Cours") => { window.location.assign("/cours/" + l.id + "?section=" + encodeURIComponent(section)); };
  const navigate = (v: View) => { setView(v); setMobile(false); setQuery(""); };
  const passed = progress.passed.length;
  const percent = Math.round(passed / lessons.length * 100);
  const resume = lessons.find(l => !progress.passed.includes(l.id)) || lessons[0];
  const currentProject = projects.find(p => p.id === projectId);
  const nav = [{id:"dashboard",label:"Mon espace",icon:LayoutDashboard},{id:"courses",label:"Les parcours",icon:BookOpen},{id:"projects",label:"Les projets",icon:FlaskConical},{id:"review",label:"Mes révisions",icon:RotateCcw},{id:"library",label:"La bibliothèque",icon:Library}];
  const relevant = lessons.filter(l => (filter === "all" || l.module === filter) && (l.title + " " + l.goal).toLocaleLowerCase("fr").includes(query.toLocaleLowerCase("fr")));
  const activity = Array.from({length:7},(_,i) => { const d=new Date(); d.setDate(d.getDate()-6+i); return {label:d.toLocaleDateString("fr",{weekday:"narrow"}),active:progress.dates.includes(d.toLocaleDateString("en-CA"))}; });
  let streak = 0;
  const cursor = new Date();
  if (!progress.dates.includes(day())) cursor.setDate(cursor.getDate()-1);
  while (progress.dates.includes(cursor.toLocaleDateString("en-CA"))) { streak++; cursor.setDate(cursor.getDate()-1); }
  const validate = () => {
    setFeedback(true);
    if (selected && choice === selected.answer) setProgress(p => ({...p, passed:[...new Set([...p.passed,selected.id])], dates:[...new Set([...p.dates,day()])]}));
  };
  const exportProgress = () => {
    const blob = new Blob([JSON.stringify(progress,null,2)], {type:"application/json"});
    const url = URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download="recap-ml-progression.json"; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  return <div className="app-shell">
    {mobile && <button className="sidebar-scrim" aria-label="Fermer le menu" onClick={()=>setMobile(false)}/>}
    <aside className={"sidebar "+(mobile?"visible":"")}>
      <a href="#" className="brand" onClick={e=>{e.preventDefault();navigate("dashboard");}}><span className="brand-mark"><Network size={23}/></span>recap<span className="brand-ml">ml</span><span className="brand-dot">.</span></a>
      <div className="workspace-label">TON LABORATOIRE D’APPRENTISSAGE</div>
      <nav aria-label="Navigation principale">{nav.map(n=><button key={n.id} className={view===n.id?"nav-item active":"nav-item"} onClick={()=>navigate(n.id as View)}><n.icon size={19}/>{n.label}{n.id==="courses"&&<span className="nav-count">6</span>}</button>)}</nav>
      <a className="nav-item tutor-nav" href="/tuteur"><CircleHelp size={19}/>Mon tuteur IA<ArrowUpRight size={15}/></a>
      <div className="sidebar-label">L’OBJECTIF</div>
      <div className="goal-note"><span className="little-star">✳</span><p>Comprendre en profondeur.<br/>Construire pour de vrai.</p><span>Un concept, puis un projet.</span></div>
      <div className="sidebar-bottom"><div className="level"><div className="level-icon"><GraduationCap size={20}/></div><div><strong>Chaque pas compte</strong><small>{passed * 100} XP · {passed} quiz réussis</small></div></div><div className="mini-progress"><i style={{width:percent+"%"}}/></div><p>L’expertise se construit par la pratique.</p></div>
      <div className="profile"><span className="avatar">N</span><div><strong>Mon apprentissage</strong><small>Espace personnel</small></div><span className="online-dot"/></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb"><button className="mobile-menu icon-button" aria-label="Ouvrir le menu" onClick={()=>setMobile(!mobile)}><Menu size={22}/></button><span>Mon laboratoire</span><ChevronRight size={14}/><strong>{nav.find(n=>n.id===view)?.label}</strong></div><div className="topbar-right"><span className="local-tag"><span/>Sauvegarde locale</span><span className="top-avatar">N</span></div></header>
      <main id="main-content">
        {storageError && <div className="storage-alert" role="status">Le navigateur n’a pas pu lire ou enregistrer la progression. Exporte tes données avant de quitter.</div>}
        <div className="page-heading"><div><div className="eyebrow"><span/>L’ENVIE D’APPRENDRE, LE POUVOIR DE CONSTRUIRE</div><h1>{view==="dashboard" ? <>Petit à petit, <span>tu vas loin.</span></> : view==="courses" ? <>Des bases <span>à la maîtrise.</span></> : view==="projects" ? <>Apprendre, c’est <span>construire.</span></> : view==="review" ? <>Comprendre. Retenir. <span>Recommencer.</span></> : <>La curiosité <span>comme moteur.</span></>}</h1><p>{view==="dashboard"?"Ton prochain déclic commence ici. On continue l’aventure ?":view==="courses"?"Des concepts expliqués, des maths qui ont du sens et du code à comprendre.":view==="projects"?"Un vrai problème. Des décisions à défendre. Un résultat reproductible.":view==="review"?"Reviens sur tes acquis et transforme tes erreurs en points forts.":"Tes notes, les scénarios vidéo et une méthode pour explorer les nouveautés."}</p></div><div className="date-pill"><Flame size={17}/><strong>{streak}</strong> jour{streak>1?"s":""} de pratique</div></div>
        {view==="dashboard" && <div className="dashboard-grid"><div className="primary-column">
          <section className="continue-card"><div className="continue-content"><span className="pill"><span/>TON PROCHAIN PAS</span><h2>{resume.title}</h2><p>{resume.goal} Une idée claire vaut mieux qu’une formule apprise par cœur.</p><div className="lesson-meta"><span><BookOpen size={14}/>Cours + pratique</span><span><Clock3 size={14}/>{["python-foundations","tools-foundations"].includes(resume.id)?"Plusieurs séances":resume.minutes+" min"}</span><span>+100 XP au quiz</span></div><button className="button-dark" onClick={()=>openLesson(resume)}> {passed?"Continuer à apprendre":"Commencer à apprendre"}<ArrowRight size={17}/></button></div><div className="hero-art" aria-hidden="true"><div className="orbit o1"/><div className="orbit o2"/><div className="orbit o3"/><div className="hero-code">{"{ }"}</div><div className="float-symbol s1">✦</div><div className="float-symbol s2">x²</div><div className="float-symbol s3">↗</div><span className="art-caption">LA THÉORIE PREND VIE.</span></div></section>
          <div className="section-title"><div><h2>Un parcours, de grandes possibilités<span className="count-badge">6</span></h2><p>Construis tes fondations, puis repousse tes limites.</p></div><button className="text-button" onClick={()=>navigate("courses")}>Tout explorer<ArrowUpRight size={16}/></button></div>
          <div className="course-grid">{[modules[0],modules[2],modules[3]].map((m,i)=>{const Icon=icons[modules.indexOf(m)]; const list=lessons.filter(l=>l.module===m.id); const done=list.filter(l=>progress.passed.includes(l.id)).length;return <button className="course-card" key={m.id} onClick={()=>{setFilter(m.id);navigate("courses");}}><Art kind={i}/><div className="course-body"><span className={"course-category color-"+i}><Icon size={13}/>{i===0?"FONDATIONS":i===1?"MODÉLISATION":"RÉSEAUX DE NEURONES"}</span><h3>{m.name}</h3><p>{m.description}</p><div className="course-footer"><span>{list.length} leçons disponibles</span><ArrowUpRight size={17}/></div><div className="card-progress"><i style={{width:done/list.length*100+"%"}}/></div></div></button>})}</div>
          <section className="project-feature"><div className="project-illustration"><FlaskConical size={34}/><span>BUILD<br/>TO LEARN.</span></div><div><span className="eyebrow">DE LA NOTION AU VRAI PROJET</span><h3>Ton premier modèle, de bout en bout.</h3><p>Données, baseline, évaluation : construis un projet que tu sauras expliquer.</p><span className="small-meta">5 étapes guidées · Machine learning</span></div><button className="round-button" aria-label="Découvrir le projet tabulaire" onClick={()=>setProjectId("tabular")}><ArrowUpRight size={22}/></button></section>
          <div className="learning-principle"><span>✳</span><p><strong>Pas seulement savoir. Savoir faire.</strong> Chaque parcours associe théorie, exercices et projets concrets.</p><span className="principle-line"/></div>
        </div><aside className="right-column">
          <section className="progress-panel"><div className="panel-heading"><h3>Ton élan de la semaine</h3><Zap size={17}/></div><div className="weekly-stat"><strong>{activity.filter(d=>d.active).length}<span>/ 7</span></strong><span>jours de pratique<br/>sur les 7 derniers jours</span></div><div className="week-days">{activity.map((d,i)=><div key={i}><span>{d.label}</span><i className={d.active?"done":""}>{d.active?<Check size={14}/>:<span/>}</i></div>)}</div><div className="weekly-bottom"><Target size={16}/><span>Un peu chaque jour, beaucoup à la fin.</span></div></section>
          <section className="journey-panel"><div className="panel-heading"><h3>Ta progression</h3><Trophy size={17}/></div><div className="progress-ring" style={{background:"conic-gradient(#74843f "+percent+"%, #ebece3 0)"}}><div><strong>{percent}<span>%</span></strong><small>quiz validés</small></div></div><p>{passed} sur {lessons.length} quiz disponibles</p><div className="xp-line"><span><span className="xp-dot"/>Comprendre avant d’avancer</span><strong>{passed*100} XP</strong></div></section>
          <section className="daily-quiz"><span className="eyebrow"><CircleHelp size={15}/>LE DÉCLIC DU JOUR</span><h3>Un programme qui prédit fait-il forcément du ML ?</h3><p>Une règle écrite à la main utilise aussi des données. Qu’est-ce qui distingue l’apprentissage ?</p><button className="text-button" onClick={()=>openLesson(lessons.find(l=>l.id==="ml-introduction")!)}>Explorer la question<ArrowRight size={15}/></button></section>
          <div className="small-quote">« Ce que l’on comprend,<br/>on peut le reconstruire. »<span>LA PHILOSOPHIE RECAP ML</span></div>
        </aside></div>}
        {view==="courses" && <><div className="course-toolbar"><label className="search-box"><Search size={18}/><input aria-label="Rechercher une leçon" placeholder="Une notion à explorer…" value={query} onChange={e=>setQuery(e.target.value)}/></label><span>{lessons.length} leçons disponibles · parcours en construction</span></div><div className="filter-row"><button className={filter==="all"?"selected":""} onClick={()=>setFilter("all")}>Tout explorer</button>{modules.map(m=><button key={m.id} className={filter===m.id?"selected":""} onClick={()=>setFilter(m.id)}>{m.name}</button>)}</div><div className="lesson-list">{relevant.map((l,i)=><button className="lesson-row" key={l.id} onClick={()=>openLesson(l)}><span className={"lesson-number "+(progress.passed.includes(l.id)?"completed":"")}>{progress.passed.includes(l.id)?<Check size={20}/>:String(i+1).padStart(2,"0")}</span><div><small>{modules.find(m=>m.id===l.module)?.name}</small><h3>{l.title}</h3><p>{l.goal}</p></div><span className="lesson-duration">{["python-foundations","tools-foundations"].includes(l.id)?"Plusieurs séances":l.minutes+" min"}</span><ArrowUpRight size={20}/></button>)}{!relevant.length&&<p className="empty-state">Aucune leçon trouvée. Essaie une autre notion ou un autre parcours.</p>}</div><section className="roadmap-panel"><span className="eyebrow">MACHINE LEARNING · 12 CHAPITRES</span><h2>Ton programme, chapitre par chapitre.</h2><p>Données, métriques, modèles, séries temporelles et robustesse.</p><a className="button-dark" style={{marginTop:18}} href="/parcours/machine-learning">Explorer la roadmap ML <ArrowRight size={16}/></a></section><section className="roadmap-panel"><span className="eyebrow">LE CAP À LONG TERME</span><h2>Un socle large. Puis de la profondeur.</h2><p>Ces approfondissements sont planifiés ; leurs cours restent à rédiger et à valider.</p><div className="roadmap-grid">{nextTopics.map((t,i)=><div key={t}><span>{String(i+1).padStart(2,"0")}</span>{t}<small>À concevoir</small></div>)}</div></section></>}
        {view==="projects" && <><div className="project-intro"><Target size={26}/><p>Chaque défi devient un projet complet : un cahier des charges, des étapes, des livrables et une soutenance. Les étapes cochées constituent ton suivi personnel ; elles ne remplacent pas une revue du code.</p></div><div className="project-grid">{projects.map((p,i)=><button className="large-project" key={p.id} onClick={()=>setProjectId(p.id)}><span className={"project-emblem emblem-"+i}>{i===0?<Network/>:i===1?<GitBranch/>:<Sparkles/>}</span><span className="eyebrow">{p.tag}</span><h2>{p.title}</h2><p>{p.description}</p><div className="project-status">{p.steps.filter((_,j)=>progress.steps.includes(p.id+"-"+j)).length}/{p.steps.length} étapes préparées<ArrowUpRight size={19}/></div></button>)}</div><section className="roadmap-panel"><h2>La preuve d’autonomie</h2><p>Reprendre un projet depuis un dossier vide, justifier les choix, comparer à une baseline, reproduire les résultats et défendre les limites. La documentation reste autorisée.</p><div className="rubric-grid">{["Cadrage & protocole","Qualité du code & tests","Résultats & analyse","Reproduction & soutenance"].map(t=><div key={t}><CheckCircle2 size={18}/>{t}</div>)}</div></section></>}
        {view==="review" && <><div className="review-stats"><div><strong>{passed}</strong><span>quiz réussis</span></div><div><strong>{lessons.length-passed}</strong><span>notions à explorer</span></div><div><strong>{lessons.filter(l=>progress.notes[l.id]).length}</strong><span>notes personnelles</span></div></div><h2 className="subheading">Le rappel actif, avant la relecture.</h2><p className="section-description">Explique la notion sans support, puis ouvre le quiz. Reviens le lendemain, après une semaine et après un mois selon tes difficultés.</p><div className="lesson-list">{lessons.filter(l=>progress.passed.includes(l.id)).map(l=><button className="lesson-row" key={l.id} onClick={()=>{openLesson(l,"Quiz");}}><RotateCcw size={22}/><div><h3>{l.title}</h3><p>{l.question}</p></div><ArrowRight size={20}/></button>)}</div>{passed===0&&<div className="empty-state"><BookOpen size={30}/><h3>Le premier déclic t’attend.</h3><p>Réussis un quiz pour retrouver ici la notion à réviser.</p><button className="button-dark" onClick={()=>openLesson(resume)}>Ouvrir la première leçon<ArrowRight size={16}/></button></div>}</>}
        {view==="library" && <><div className="library-grid"><section className="library-card"><BookOpen size={27}/><h2>Les supports pour pratiquer.</h2><p>Retrouve les explications, les laboratoires Python, les challenges et les quiz dans chaque cours.</p><button className="text-button" onClick={()=>{openLesson(resume);}}>Ouvrir un cours<ArrowRight size={16}/></button></section><section className="library-card"><Bookmark size={27}/><h2>Ton carnet de compréhension</h2><p>Les réponses aux exercices et les notes déjà enregistrées restent dans ce navigateur. Exporte ta progression pour les conserver.</p><button className="text-button" onClick={exportProgress}>Exporter ma progression<ArrowUpRight size={16}/></button></section></div><section className="roadmap-panel"><span className="eyebrow">VEILLE SCIENTIFIQUE</span><h2>Une nouveauté mérite une expérience.</h2><p>Ce prototype ne dispose pas de flux d’actualités. Pour chaque sujet étudié, consigne la date, la source primaire, la contribution, les preuves, les limites et un essai reproductible.</p><div className="rubric-grid">{["1. Lire la source originale","2. Identifier ce qui est mesuré","3. Reproduire un petit cas","4. Décider de son utilité"].map(t=><div key={t}>{t}</div>)}</div></section><h2 className="subheading">Tes notes</h2>{lessons.filter(l=>progress.notes[l.id]).map(l=><button key={l.id} className="note-card" onClick={()=>{openLesson(l,"Exercices");}}><strong>{l.title}</strong><p>{progress.notes[l.id]}</p></button>)}</>}
        <footer><span>recap ml<span className="brand-dot">.</span> <span className="footer-separator">/</span> Comprendre. Pratiquer. Maîtriser.</span><span>Apprendre est un chemin, pas une course.<span className="footer-star">✳</span></span></footer>
      </main>
    </div>
    {currentProject && <div className="modal-backdrop" onClick={()=>setProjectId(null)}><section className="lesson-modal" role="dialog" aria-modal="true" aria-labelledby="project-title" tabIndex={-1} onClick={e=>e.stopPropagation()}><div className="modal-heading"><span className="eyebrow">{currentProject.tag} · PROJET GUIDÉ</span><button className="icon-button" aria-label="Fermer le projet" onClick={()=>setProjectId(null)}><X/></button></div><h2 id="project-title">{currentProject.title}</h2><p>{currentProject.description}</p><div className="callout"><strong>Le contrat de réalisation</strong><p>Construis un dépôt reproductible avec code source, instructions, configuration, protocole d’évaluation et rapport des limites. Présente une baseline avant d’ajouter de la complexité.</p></div><h3>Les jalons du projet</h3><div className="project-checklist">{currentProject.steps.map((s,i)=><label key={s}><input type="checkbox" checked={progress.steps.includes(currentProject.id+"-"+i)} onChange={()=>setProgress(p=>({...p,steps:p.steps.includes(currentProject.id+"-"+i)?p.steps.filter(x=>x!==currentProject.id+"-"+i):[...p.steps,currentProject.id+"-"+i]}))}/><span><strong>Étape {i+1}</strong>{s}</span></label>)}</div><h3>Les livrables attendus</h3><ul><li>README : besoin, données, installation et commande de reproduction.</li><li>Code organisé et tests des transformations ou invariants critiques.</li><li>Tableau des expériences : baseline, métriques, configuration et résultats réellement obtenus.</li><li>Rapport : dix erreurs analysées, limites, coût et prochaines expériences.</li></ul><h3>La soutenance</h3><p>Pourquoi ce protocole ? Qu’est-ce qui pourrait invalider tes résultats ? Comment le système échoue-t-il ? Que changerais-tu si le coût ou la latence devait être divisé par deux ?</p><label className="field-label" htmlFor="project-notes">Lien du dépôt, décisions et questions pour la revue</label><textarea id="project-notes" rows={5} value={progress.notes[currentProject.id+"-project"]||""} onChange={e=>setProgress(p=>({...p,notes:{...p.notes,[currentProject.id+"-project"]:e.target.value}}))}/><small>Suivi personnel : la validation technique nécessite une revue des livrables.</small></section></div>}
  </div>;
}
