"use client";

import { useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Check, RotateCcw, Search, Shuffle, Sparkles } from "lucide-react";
import cards from "@/lib/chapter-one-memory-cards.json";

export default function ChapterOneMemoryCards() {
  const [order, setOrder] = useState(cards.map(card => card.id));
  const [flipped, setFlipped] = useState(false);
  const [position, setPosition] = useState(0);
  const [query, setQuery] = useState("");
  const visible = useMemo(() => {
    const search = query.trim().toLocaleLowerCase("fr");
    return order.map(id => cards.find(card => card.id === id)!).filter(card =>
      (card.question + " " + card.topic).toLocaleLowerCase("fr").includes(search)
    );
  }, [order, query]);

  function shuffle() {
    setOrder(previous => {
      const next = [...previous];
      for (let i = next.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [next[i], next[j]] = [next[j], next[i]];
      }
      return next;
    });
    setFlipped(false);
    setPosition(0);
  }

  return <section className="memory-deck" aria-labelledby="memory-title">
    <div className="memory-heading">
      <div><span className="eyebrow"><Sparkles size={15}/>CHAPITRE 01 · MEMORY CARDS</span>
        <h2 id="memory-title">Une question. Un déclic.</h2>
        <p>Prends le temps de chercher, puis retourne la carte pour découvrir la réponse et son explication.</p>
      </div>
      <span className="memory-count">20 cartes <span>· Débutant</span></span>
    </div>
    <div className="memory-toolbar">
      <label className="search-box"><Search size={17}/><input value={query} onChange={event => { setQuery(event.target.value); setPosition(0); setFlipped(false); }} placeholder="Chercher une question ou un thème…" aria-label="Rechercher une memory card"/></label>
      <div className="memory-actions">
        <button type="button" onClick={shuffle}><Shuffle size={16}/>Mélanger</button>
        <button type="button" onClick={() => setFlipped(false)} disabled={!flipped}><RotateCcw size={16}/>Masquer la réponse</button>
      </div>
    </div>
    <div className="memory-progress"><span aria-live="polite">{visible.length ? "Carte " + (position + 1) + " sur " + visible.length : "Aucune carte"}</span><div role="progressbar" aria-label="Position dans les cartes" aria-valuemin={0} aria-valuemax={visible.length || 1} aria-valuenow={visible.length ? position + 1 : 0}><i style={{width: (visible.length ? (position + 1) / visible.length * 100 : 0) + "%"}}/></div></div>
    <div className="memory-grid memory-single">
      {visible.slice(position, position + 1).map(card => {

        return <button type="button" className={"memory-card" + (flipped ? " is-flipped" : "")} key={card.id}
          aria-pressed={flipped} aria-label={flipped ? card.topic + " : " + card.answer + " " + card.explanation + " Masquer la réponse." : card.question + " Révéler la réponse."}
          onClick={() => setFlipped(previous => !previous)}>
          <span className="memory-card-inner">
            <span className="memory-face memory-front" aria-hidden={flipped}>
              <span className="memory-card-top"><span>{card.topic}</span><span>{card.id.slice(-3)}</span></span>
              <span className="memory-label">LA QUESTION</span>
              <span className="memory-question">{card.question}</span>
              <span className="memory-card-hint">Retourner la carte<ArrowRight size={16}/></span>
            </span>
            <span className="memory-face memory-back" aria-hidden={!flipped}>
              <span className="memory-card-top"><span><Check size={14}/>La réponse</span><span>{card.id.slice(-3)}</span></span>
              <span className="memory-answer">{card.answer}</span>
              <span className="memory-explanation">{card.explanation}</span>
              <span className="memory-card-hint">Revenir à la question<RotateCcw size={15}/></span>
            </span>
          </span>
        </button>;
      })}
    </div>
    {visible.length > 0 && <nav className="memory-navigation" aria-label="Parcourir les memory cards">
      <button type="button" disabled={position === 0} onClick={() => { setPosition(previous => previous - 1); setFlipped(false); }}><ArrowLeft size={17}/>Précédente</button>
      <span aria-live="polite">{position + 1} / {visible.length}</span>
      <button type="button" disabled={position === visible.length - 1} onClick={() => { setPosition(previous => previous + 1); setFlipped(false); }}>Suivante<ArrowRight size={17}/></button>
    </nav>}
    {!visible.length && <p className="empty-state">Aucune carte trouvée. Essaie un autre thème.</p>}
    <p className="memory-footnote">Un clic pour retourner · Entrée ou Espace au clavier · Un nouveau clic pour masquer</p>
  </section>;
}
