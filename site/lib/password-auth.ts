import { pbkdf2, randomBytes, createHash, timingSafeEqual } from "node:crypto";
const cookieName="recap_session";
const ttl=30*24*60*60;
const reply=(data:unknown,status=200,cookie?:string)=>Response.json(data,{status,headers:{"Cache-Control":"private, no-store",...(cookie?{"Set-Cookie":cookie}:{})}});
const digest=(value:string)=>createHash("sha256").update(value).digest("hex");
function tokenFrom(request:Request){return request.headers.get("cookie")?.split(";").map(x=>x.trim()).find(x=>x.startsWith(cookieName+"="))?.slice(cookieName.length+1)||"";}
function cookie(request:Request,token:string,age=ttl){return cookieName+"="+token+"; Path=/; HttpOnly; SameSite=Lax; Max-Age="+age+(new URL(request.url).protocol==="https:"?"; Secure":"");}
function derive(password:string,salt:string){return new Promise<Buffer>((resolve,reject)=>pbkdf2(password,salt,600000,32,"sha256",(error,key)=>error?reject(error):resolve(key)));}
async function hashPassword(password:string){const salt=randomBytes(16).toString("hex");return salt+":"+(await derive(password,salt)).toString("hex");}
async function verifyPassword(password:string,stored:string){const [salt,hash]=stored.split(":");const actual=await derive(password,salt);const expected=Buffer.from(hash,"hex");return actual.length===expected.length&&timingSafeEqual(actual,expected);}
export async function sessionUser(request:Request,db:D1Database){
 const token=tokenFrom(request);if(!/^[a-f0-9]{64}$/.test(token))return null;
 return db.prepare("SELECT l.id,l.email,l.name FROM account_sessions s JOIN learners l ON l.id=s.user_id WHERE s.hash=? AND s.expires_at>?").bind(digest(token),Date.now()).first<{id:string;email:string;name:string}>();
}
async function limited(db:D1Database,key:string,max:number){
 const now=Date.now(),window=15*60*1000;
 const row=await db.prepare("INSERT INTO auth_limits (key,count,started_at) VALUES (?,1,?) ON CONFLICT(key) DO UPDATE SET count=CASE WHEN auth_limits.started_at<? THEN 1 ELSE auth_limits.count+1 END,started_at=CASE WHEN auth_limits.started_at<? THEN excluded.started_at ELSE auth_limits.started_at END RETURNING count").bind(key,now,now-window,now-window).first<{count:number}>();
 return (row?.count||0)>max;
}
export async function handleAuth(request:Request,db:D1Database){
 if(request.method!=="POST")return reply({error:"Méthode refusée."},405);
 if(request.headers.get("origin")!==new URL(request.url).origin||request.headers.get("sec-fetch-site")==="cross-site")return reply({error:"Origine refusée."},403);
 const action=new URL(request.url).pathname.split("/").pop();
 if(action==="logout"){
  const token=tokenFrom(request);if(token)await db.prepare("DELETE FROM account_sessions WHERE hash=?").bind(digest(token)).run();
  return reply({ok:true},200,cookie(request,"",0));
 }
 if(action!=="login"&&action!=="register")return reply({error:"Route inconnue."},404);
 if(!request.headers.get("content-type")?.startsWith("application/json"))return reply({error:"JSON requis."},415);
 let body;try{const raw=await request.text();if(raw.length>4096)return reply({error:"Formulaire trop volumineux."},413);body=JSON.parse(raw);}catch{return reply({error:"Formulaire invalide."},400);}
 if(!body||typeof body.email!=="string"||typeof body.password!=="string")return reply({error:"Email et mot de passe requis."},400);
 const email=body.email.trim().toLowerCase(),password=body.password;
 if(email.length>254||!/^\S+@\S+\.\S+$/.test(email)||password.length>128||password.length<1)return reply({error:"Email ou mot de passe invalide."},400);
 const ip=request.headers.get("cf-connecting-ip")||"local";
 if(await limited(db,"ip:"+digest(ip),60)||await limited(db,"email:"+digest(email),15))return reply({error:"Trop de tentatives. Réessaie dans 15 minutes."},429);
 await db.batch([db.prepare("DELETE FROM account_sessions WHERE expires_at<?").bind(Date.now()),db.prepare("DELETE FROM auth_limits WHERE started_at<?").bind(Date.now()-86400000)]);
 let userId:string;
 if(action==="register"){
  if(typeof body.name!=="string"||!body.name.trim()||body.name.trim().length>80)return reply({error:"Indique un nom de 1 à 80 caractères."},400);
  if(password.length<10)return reply({error:"Choisis un mot de passe d’au moins 10 caractères."},400);
  userId=crypto.randomUUID();
  const passwordHash=await hashPassword(password);
  try{await db.batch([
   db.prepare("INSERT INTO learners (id,email,name) VALUES (?,?,?)").bind(userId,email,body.name.trim()),
   db.prepare("INSERT INTO account_credentials (user_id,email,password_hash) VALUES (?,?,?)").bind(userId,email,passwordHash)
  ]);}catch(e){
   if(String(e).includes("UNIQUE"))return reply({error:"Un compte utilise déjà cet email. Connecte-toi."},409);
   throw e;
  }
 }else{
  const credentials=await db.prepare("SELECT user_id,password_hash FROM account_credentials WHERE email=?").bind(email).first<{user_id:string;password_hash:string}>();
  const fallback="00000000000000000000000000000000:"+ "0".repeat(64);
  const valid=await verifyPassword(password,credentials?.password_hash||fallback);
  if(!credentials||!valid)return reply({error:"Email ou mot de passe incorrect."},401);
  userId=credentials.user_id;
 }
 const token=randomBytes(32).toString("hex");
 await db.prepare("INSERT INTO account_sessions (hash,user_id,expires_at) VALUES (?,?,?)").bind(digest(token),userId,Date.now()+ttl*1000).run();
 return reply({ok:true},200,cookie(request,token));
}
