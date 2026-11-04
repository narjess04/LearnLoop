import { blockingLesson, blockingChapter, chapterNumber } from "./learning-paths";
import { handleAuth, sessionUser } from "./password-auth";
import { lessons } from "./curriculum";
import { richCourses } from "./ml-rich-courses";

const json = (data: unknown, status = 200) => Response.json(data, { status, headers: { "Cache-Control": "private, no-store" } });
export async function handleAccount(request: Request, env: Cloudflare.Env) {
  if (!env.DB) return json({error:"La base de données est indisponible."},503);
  if(new URL(request.url).pathname.startsWith("/api/account/auth/"))return handleAuth(request,env.DB);
  const user=await sessionUser(request,env.DB);
  if(!user)return json({error:"Connexion requise."},401);
  const userId=user.id;
  if(request.method!=="GET" && request.headers.get("X-Recap-User")!==userId)return json({error:"Le compte actif a chang\u00e9. Recharge la page avant de continuer."},409);
  if (request.method !== "GET") {
    const origin = request.headers.get("origin");
    if (origin !== new URL(request.url).origin || request.headers.get("sec-fetch-site") === "cross-site") return json({ error: "Origine refusée." }, 403);
    if (!request.headers.get("content-type")?.startsWith("application/json")) return json({ error: "JSON requis." }, 415);
    if (Number(request.headers.get("content-length")) > 250000) return json({ error: "Données trop volumineuses." }, 413);
  }
  const db = env.DB;
  const path = new URL(request.url).pathname;
  if (request.method === "GET" && path === "/api/account") {
    const profile = await db.prepare("SELECT name,email FROM learners WHERE id=?").bind(userId).first<{name:string;email:string}>();
    const records = await db.prepare("SELECT key,value FROM learning_records WHERE user_id=?").bind(userId).all<{key:string;value:string}>();
    const completed = await db.prepare("SELECT lesson_id,date FROM quiz_completions WHERE user_id=?").bind(userId).all<{lesson_id:string;date:string}>();
    const values: Record<string,string> = Object.fromEntries(records.results.map(r=>[r.key,r.value]));
    const progress = JSON.parse(values["recap-learning-v1"] || '{"steps":[],"notes":{}}');
    progress.passed = completed.results.map(r=>r.lesson_id);
    progress.dates = [...new Set(completed.results.map(r=>r.date))];
    values["recap-learning-v1"] = JSON.stringify(progress);
    return json({ user: {id:userId,...profile}, values, points: completed.results.length * 100 });
  }
  if (request.method !== "POST") return json({error:"Méthode refusée."},405);
  let body;
  try { const raw=await request.text(); if(raw.length>250000)return json({error:"Données trop volumineuses."},413); body=JSON.parse(raw); } catch {return json({error:"Données invalides."},400);}
  if (path === "/api/account/profile") {
    if(typeof body.name!=="string" || !body.name.trim() || body.name.trim().length>80)return json({error:"Le nom doit contenir de 1 à 80 caractères."},400);
    await db.prepare("UPDATE learners SET name=? WHERE id=?").bind(body.name.trim(),userId).run();
    return json({ok:true});
  }
  if (path === "/api/account/quiz") {
    const lesson=lessons.find(l=>l.id===body.lessonId);
    if(!lesson || !Array.isArray(body.answers))return json({error:"Quiz invalide."},400);
    const done=await db.prepare("SELECT lesson_id FROM quiz_completions WHERE user_id=?").bind(userId).all<{lesson_id:string}>();
    const passed=done.results.map(r=>r.lesson_id);
    const chapterBlock=blockingChapter(chapterNumber(lesson.id)||1,passed);
    if(chapterBlock)return json({error:"Termine tous les cours du chapitre "+chapterBlock.number+" avant de continuer."},403);
    const blocker=blockingLesson(lesson.id,passed);
    if(blocker)return json({error:"Termine d’abord le quiz de : "+blocker.title},403);
    const rich=richCourses[lesson.id as keyof typeof richCourses];
    const expected=rich?rich.quiz.map(q=>q.answer):[lesson.answer];
    const correct=body.answers.length===expected.length && expected.every((a,i)=>a===body.answers[i]);
    if(!correct)return json({error:"Reprends les réponses du quiz."},400);
    const date=new Intl.DateTimeFormat("en-CA",{timeZone:"Africa/Tunis"}).format(new Date());
    const result=await db.prepare("INSERT OR IGNORE INTO quiz_completions (user_id,lesson_id,date) VALUES (?,?,?)").bind(userId,lesson.id,date).run();
    return json({ok:true,awarded:result.meta.changes?100:0,date});
  }
  if (path === "/api/account/record") {
    if(typeof body.key!=="string" || !/^recap-[a-zA-Z0-9-]{1,100}$/.test(body.key) || typeof body.value!=="string" || body.value.length>200000)return json({error:"Enregistrement invalide."},400);
    let value;try{value=JSON.parse(body.value);}catch{return json({error:"Valeur invalide."},400);}
    if(body.key==="recap-learning-v1"){
      if(!value || !Array.isArray(value.steps) || !value.steps.every((s:unknown)=>typeof s==="string") || !value.notes || typeof value.notes!=="object" || Array.isArray(value.notes) || !Object.values(value.notes).every(v=>typeof v==="string"))return json({error:"Progression invalide."},400);
      value={steps:[...new Set(value.steps)],notes:value.notes};
    }
    await db.prepare("INSERT INTO learning_records (user_id,key,value) VALUES (?,?,?) ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value").bind(userId,body.key,JSON.stringify(value)).run();
    return json({ok:true});
  }
  return json({error:"Route inconnue."},404);
}
