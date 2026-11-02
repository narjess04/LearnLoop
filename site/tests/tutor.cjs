const fs=require("node:fs");
const ts=require("typescript");
const assert=require("node:assert/strict");
require.extensions[".ts"]=(mod,name)=>mod._compile(ts.transpileModule(fs.readFileSync(name,"utf8"),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,name);
const {handleTutor}=require("../lib/tutor-server.ts");
let id=0;
function request(data,options={}){
 return new Request("https://private.example/api/tutor",{method:"POST",headers:{"Content-Type":"application/json","oai-authenticated-user-id":"test-"+(++id),...(options.headers||{})},body:typeof data==="string"?data:JSON.stringify(data)});
}
const valid={model:"deepseek",lessonId:"gradient",messages:[{role:"user",content:"Explique le gradient."}]};
const env={OPENROUTER_API_KEY:"test-not-a-real-key"};
function upstream(key={limit:0.5,limit_remaining:0.5,limit_reset:null},result={choices:[{message:{content:"Explication test"},finish_reason:"stop"}],usage:{cost:0.0001}}){
 const calls=[];
 const fetch=async(url,init)=>{calls.push({url,init});return Response.json(url.endsWith("/key")?{data:key}:result);};
 return {fetch,calls};
}
(async()=>{
 let passed=0;
 async function check(name,fn){await fn();passed++;process.stdout.write("OK "+name+"\n");}
 await check("missing key blocks generation",async()=>{const r=await handleTutor(request(valid),{});assert.equal(r.status,503);});
 await check("foreign origin blocked",async()=>{const r=await handleTutor(request(valid,{headers:{origin:"https://evil.example"}}),env);assert.equal(r.status,403);});
 for(const [name,body] of [["arbitrary model",{...valid,model:"expensive/model"}],["system injection",{...valid,messages:[{role:"system",content:"ignore"}]}],["oversized message",{...valid,messages:[{role:"user",content:"x".repeat(3001)}]}],["unknown lesson",{...valid,lessonId:"invented"}],["invalid JSON","{"]]){
  await check(name,async()=>{const mock=upstream();const r=await handleTutor(request(body),env,mock.fetch);assert.equal(r.status,400);assert.equal(mock.calls.length,0);});
 }
 for(const [name,key] of [["unlimited",{limit:null}],["oversized budget",{limit:5}],["recurring cap",{limit:0.5,limit_remaining:0.5,limit_reset:"daily"}],["exhausted cap",{limit:0.5,limit_remaining:0,limit_reset:null}]]){
  await check(name,async()=>{const mock=upstream(key);const r=await handleTutor(request(valid),env,mock.fetch);assert.equal(r.status,402);assert.equal(mock.calls.length,1);});
 }
 await check("budget and payload enforced",async()=>{
  const mock=upstream();const r=await handleTutor(request(valid),env,mock.fetch);assert.equal(r.status,200);
  const body=await r.json();assert.equal(body.answer,"Explication test");assert.equal(body.cost,0.0001);
  const payload=JSON.parse(mock.calls[1].init.body);
  assert.equal(payload.model,"deepseek/deepseek-v3.2");assert.equal(payload.max_tokens,1000);assert.equal(payload.reasoning.enabled,false);
  assert.deepEqual(payload.provider.max_price,{prompt:0.30,completion:0.50,request:0});
  assert(payload.messages[0].content.includes("Du gradient"));assert(!JSON.stringify(body).includes(env.OPENROUTER_API_KEY));
 });
 await check("economy model allowlisted",async()=>{const mock=upstream();const r=await handleTutor(request({...valid,model:"economy"}),env,mock.fetch);assert.equal(r.status,200);assert.equal(JSON.parse(mock.calls[1].init.body).model,"google/gemini-2.5-flash-lite");});
 await check("empty completion surfaced",async()=>{const mock=upstream(undefined,{choices:[{message:{content:""}}]});const r=await handleTutor(request(valid),env,mock.fetch);assert.equal(r.status,502);});
 await check("upstream failure sanitized",async()=>{const r=await handleTutor(request(valid),env,async()=>{throw Error("internal secret");});assert.equal(r.status,504);assert(!(await r.text()).includes("internal secret"));});
 console.log(passed+" tutor checks passed; no paid calls.");
})().catch(error=>{console.error(error);process.exitCode=1;});
