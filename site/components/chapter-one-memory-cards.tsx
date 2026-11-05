"use client";

import { useMemo, useState } from "react";
import { ArrowRight, Check, RotateCcw, Search, Shuffle, Sparkles } from "lucide-react";
import cards from "@/lib/chapter-one-memory-cards.json";

export default function ChapterOneMemoryCards() {
  const [order, setOrder] = useState(cards.map(card => card.id));
  const [revealed, setRevealed] = useState<string[]>([]);
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
    setRevealed([]);
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
      <label className="search-box"><Search size={17}/><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Chercher une question ou un thème…" aria-label="Rechercher une memory card"/></label>
      <div className="memory-actions">
        <button type="button" onClick={shuffle}><Shuffle size={16}/>Mélanger</button>
        <button type="button" onClick={() => setRevealed([])} disabled={!revealed.length}><RotateCcw size={16}/>Tout masquer</button>
      </div>
    </div>
    <div className="memory-progress"><span aria-live="polite">{revealed.length} / {cards.length} réponses affichées</span><div role="progressbar" aria-label="Réponses actuellement affichées" aria-valuemin={0} aria-valuemax={cards.length} aria-valuenow={revealed.length}><i style={{width: (revealed.length / cards.length * 100) + "%"}}/></div></div>
    <div className="memory-grid">
      {visible.map(card => {
        const flipped = revealed.includes(card.id);
        return <button type="button" className={"memory-card" + (flipped ? " is-flipped" : "")} key={card.id}
          aria-pressed={flipped} aria-label={flipped ? card.topic + " : " + card.answer + " " + card.explanation + " Masquer la réponse." : card.question + " Révéler la réponse."}
          onClick={() => setRevealed(previous => previous.includes(card.id) ? previous.filter(id => id !== card.id) : [...previous, card.id])}>
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
    {!visible.length && <p className="empty-state">Aucune carte trouvée. Essaie un autre thème.</p>}
    <p className="memory-footnote">Un clic pour retourner · Entrée ou Espace au clavier · Un nouveau clic pour masquer</p>
  </section>;
}

