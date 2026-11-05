"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowRight, Check, CircleHelp, LockKeyhole, RotateCcw, Trophy } from "lucide-react";
import quiz from "@/lib/chapter-one-quiz.json";
import { accountStorage } from "@/lib/account-storage";

const storageKey = "recap-ch01-qcm-v1";
const labels = ["Débutant", "Intermédiaire", "Avancé", "Expert", "Extrêmement difficile"];
const groups = quiz.levels.map(level => quiz.questions.filter(question => question.difficulty === level.name));
type Answers = Record<string, string>;

export default function ChapterOneQuiz() {
  const [answers, setAnswers] = useState<Answers>({});
  const [loaded, setLoaded] = useState(false);
  const [level, setLevel] = useState(0);
  const [position, setPosition] = useState(0);
  const [choice, setChoice] = useState("");
  const [screen, setScreen] = useState<"overview" | "question" | "result">("overview");
  const [storageError, setStorageError] = useState(false);
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      const stored: unknown = JSON.parse(accountStorage.getItem(storageKey) || "{}");
      if (stored && typeof stored === "object" && !Array.isArray(stored)) {
        const valid: Answers = {};
        for (const question of quiz.questions) {
          const answer = (stored as Record<string, unknown>)[question.id];
          if (typeof answer === "string" && question.options.some(option => option.id === answer)) valid[question.id] = answer;
        }
        setAnswers(valid);
      }
    } catch { setStorageError(true); }
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!loaded) return;
    try { accountStorage.setItem(storageKey, JSON.stringify(answers)); }
    catch { setStorageError(true); }
  }, [answers, loaded]);

  function focusPanel() { requestAnimationFrame(() => panel.current?.focus({preventScroll: true})); }
  const completed = groups.map(questions => questions.every(question => !!answers[question.id]));
  const questions = groups[level];
  const question = questions[position];
  const submitted = answers[question.id];
  const points = questions.reduce((sum, item) => sum + (answers[item.id] === item.correct_answer ? item.points : 0), 0);
  const maximum = questions.reduce((sum, item) => sum + item.points, 0);
  const correct = questions.filter(item => answers[item.id] === item.correct_answer).length;

  function start(index: number) {
    if (index > 0 && !completed[index - 1]) return;
    setLevel(index);
    const pending = groups[index].findIndex(item => !answers[item.id]);
    setPosition(Math.max(0, pending));
    setChoice("");
    setScreen(pending < 0 ? "result" : "question");
    focusPanel();
  }
  function next() {
    setChoice("");
    if (position === questions.length - 1) setScreen("result");
    else setPosition(position + 1);
    focusPanel();
  }
  function retry() {
    setAnswers(previous => Object.fromEntries(Object.entries(previous).filter(([id]) => !questions.some(item => item.id === id))));
    setPosition(0); setChoice(""); setScreen("question"); focusPanel();
  }
  function pause() { setScreen("overview"); setChoice(""); focusPanel(); }

  return <section className="chapter-quiz" aria-labelledby="chapter-quiz-title">
    <div className="memory-heading"><div><span className="eyebrow"><CircleHelp size={15}/>CHAPITRE 01 · QUIZ PAR DIFFICULTÉ</span>
      <h2 id="chapter-quiz-title">Monte d’un niveau, à ton rythme.</h2>
      <p>100 questions, 5 niveaux. Termine les 20 questions d’un niveau, puis choisis de poursuivre ou de t’arrêter. Aucun score minimum requis.</p></div><span className="memory-count">100 QCM</span></div>
    {storageError && <p className="storage-alert" role="alert">La progression du quiz n’a pas pu être enregistrée ou relue.</p>}
    <div className="chapter-quiz-panel" ref={panel} tabIndex={-1}>
      {!loaded ? <p role="status">Chargement du quiz…</p> : screen === "overview" ? <>
        <div className="quiz-levels">{quiz.levels.map((item, index) => {
          const locked = index > 0 && !completed[index - 1];
          const answered = groups[index].filter(entry => answers[entry.id]).length;
          const earned = groups[index].reduce((sum, entry) => sum + (answers[entry.id] === entry.correct_answer ? entry.points : 0), 0);
          return <button type="button" className={"quiz-level" + (completed[index] ? " is-complete" : "")} disabled={locked} key={item.name} onClick={() => start(index)}>
            <span className="quiz-level-number">{locked ? <LockKeyhole size={18}/> : completed[index] ? <Check size={20}/> : String(index + 1).padStart(2, "0")}</span>
            <span><strong>{labels[index]}</strong><small>20 questions · {item.points_per_question} point{item.points_per_question > 1 ? "s" : ""} par bonne réponse</small><small>{locked ? "Termine le niveau " + labels[index - 1] : completed[index] ? earned + " / " + (item.count * item.points_per_question) + " points · Voir le bilan" : answered ? answered + "/20 réponses · Reprendre" : "Commencer ce niveau"}</small></span>
            {!locked && <ArrowRight size={18}/>}
          </button>;
        })}</div>
        <p className="memory-footnote">Tu peux t’arrêter après chaque niveau ou mettre une session en pause. Ta progression est liée à ton compte.</p>
      </> : screen === "question" ? <>
        <div className="quiz-session-top"><span className="memory-count">{labels[level]}</span><span>Question {position + 1} / {questions.length}</span><button type="button" className="text-button" onClick={pause}>Faire une pause</button></div>
        <div className="quiz-session-progress" role="progressbar" aria-label="Questions répondues dans ce niveau" aria-valuemin={0} aria-valuemax={questions.length} aria-valuenow={questions.filter(item => answers[item.id]).length}><i style={{width: questions.filter(item => answers[item.id]).length / questions.length * 100 + "%"}}/></div>
        <div className="chapter-question" key={question.id}><span className="eyebrow">{question.lesson_title}</span><h3>{question.question}</h3>
          <fieldset className="chapter-options" disabled={!!submitted}><legend className="sr-only">Choisis une réponse</legend>{question.options.map(option => <label key={option.id} className={(choice === option.id || submitted === option.id ? "chosen " : "") + (submitted && option.id === question.correct_answer ? "correct " : "") + (submitted === option.id && submitted !== question.correct_answer ? "incorrect" : "")}>
            <input type="radio" name={question.id} value={option.id} checked={(submitted || choice) === option.id} onChange={() => setChoice(option.id)}/><span className="option-letter">{option.id}</span><span>{option.text}</span>{submitted && option.id === question.correct_answer && <Check size={19}/>}
          </label>)}</fieldset>
          {submitted ? <><div className="chapter-feedback" role="status"><strong>{submitted === question.correct_answer ? "Bonne réponse !" : "À retenir pour la prochaine fois"}</strong><p><b>Réponse {question.correct_answer} :</b> {question.correct_answer_text}</p><p>{question.explanation}</p></div><button type="button" className="button-dark" onClick={next}>{position === questions.length - 1 ? "Voir mon bilan" : "Question suivante"}<ArrowRight size={17}/></button></> :
            <button type="button" className="button-dark" disabled={!choice} onClick={() => { if (choice) setAnswers(previous => ({...previous, [question.id]: choice})); }}>Valider ma réponse<Check size={17}/></button>}
        </div>
      </> : <>
        <div className="quiz-result"><span className="quiz-result-icon"><Trophy size={30}/></span><span className="eyebrow">NIVEAU {labels[level].toLocaleUpperCase("fr")} TERMINÉ</span>
          <h3>{correct === questions.length ? "Un sans-faute !" : "Une étape de plus dans ton parcours."}</h3>
          <div className="quiz-result-score"><strong>{correct}<span> / {questions.length}</span></strong><span>bonnes réponses · {points} / {maximum} points</span></div>
          <p>{level < groups.length - 1 ? "Tu peux continuer avec le niveau " + labels[level + 1] + " ou t’arrêter ici et revenir plus tard." : "Tu as terminé les cinq niveaux du quiz du chapitre 1."}</p>
          <div className="quiz-result-actions">{level < groups.length - 1 && <button type="button" className="button-dark" onClick={() => start(level + 1)}>Continuer · {labels[level + 1]}<ArrowRight size={17}/></button>}<button type="button" className="quiz-secondary" onClick={pause}>{level < groups.length - 1 ? "M’arrêter pour le moment" : "Retour aux niveaux"}</button><button type="button" className="text-button" onClick={retry}><RotateCcw size={15}/>Refaire ce niveau</button></div>
        </div>
        <details className="quiz-corrections"><summary>Revoir les 20 réponses et leurs explications</summary>{questions.map(item => <div key={item.id}><strong>{answers[item.id] === item.correct_answer ? "✓" : "✗"} {item.question}</strong><p>Ta réponse : {item.options.find(option => option.id === answers[item.id])?.text || "Non répondue"}</p><p><b>Bonne réponse :</b> {item.correct_answer_text}</p><p>{item.explanation}</p></div>)}</details>
      </>}
    </div>
  </section>;
}

