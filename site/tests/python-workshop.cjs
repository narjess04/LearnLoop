const fs=require("node:fs"),assert=require("node:assert/strict"),crypto=require("node:crypto");
const base="http://127.0.0.1:5173";
const chapters=["one","two","three","four","five"].map(n=>JSON.parse(fs.readFileSync("lib/ml-chapter-"+n+".json","utf8")));
const lessonList=[{id:"ml-introduction",answer:1},...chapters.flatMap(c=>c.lessons)].filter((l,i,a)=>a.findIndex(x=>x.id===l.id)===i);
assert.equal(lessonList.length,48);
const rich=Object.assign({},...chapters.slice(1).map(c=>c.courses));
let cookie="",user="";
async function request(path,body){
 const response=await fetch(base+path,{method:body?"POST":"GET",headers:{
  ...(cookie?{Cookie:cookie}:{}),...(body?{"Content-Type":"application/json",Origin:base,"X-Recap-User":user}:{})
 },...(body?{body:JSON.stringify(body)}:{})});
 const data=await response.json();
 assert.equal(response.status,200,path+" "+JSON.stringify(data));
 return {response,data};
}
(async()=>{
 const suffix=crypto.randomUUID();
 const registration=await request("/api/account/auth/register",{name:"QA atelier",email:"qa-python-"+suffix+"@example.invalid",password:"Test-"+suffix});
 cookie=registration.response.headers.get("set-cookie").split(";")[0];
 user=(await request("/api/account")).data.user.id;
 for(const lesson of lessonList)await request("/api/account/quiz",{lessonId:lesson.id,answers:rich[lesson.id]?rich[lesson.id].quiz.map(q=>q.answer):[lesson.answer]});
 console.log("PASS local test account: 48 lessons unlocked");
 const page=await (await fetch("http://127.0.0.1:9334/json/new?about:blank",{method:"PUT"})).json();
 const socket=new WebSocket(page.webSocketDebuggerUrl),pending=new Map();let next=0;
 await new Promise((resolve,reject)=>{socket.addEventListener("open",resolve,{once:true});socket.addEventListener("error",reject,{once:true});});
 const exceptions=[];
 socket.addEventListener("message",event=>{
  const message=JSON.parse(event.data);
  if(message.id){const entry=pending.get(message.id);if(entry){pending.delete(message.id);message.error?entry.reject(Error(JSON.stringify(message.error))):entry.resolve(message.result);}}
  if(message.method==="Runtime.exceptionThrown")exceptions.push(message.params.exceptionDetails.text);
 });
 function cdp(method,params={}){return new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});}
 async function evaluate(expression){const r=await cdp("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
 async function wait(expression,timeout=30000){const start=Date.now();while(Date.now()-start<timeout){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,250));}throw Error("Timeout: "+expression+"; body="+(await evaluate("document.body.innerText")).slice(-3000));}
 async function navigate(path){await cdp("Page.navigate",{url:base+path});await wait("!!document.querySelector('.reader-content')");}
 async function fill(code){await evaluate("(()=>{const t=document.querySelector('.python-editor textarea');Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(t,"+JSON.stringify(code)+");t.dispatchEvent(new Event('input',{bubbles:true}));})()");await new Promise(r=>setTimeout(r,100));}
 async function run(){await evaluate("document.querySelector('.python-toolbar button').click()");}
 await cdp("Page.enable");await cdp("Runtime.enable");await cdp("Network.enable");
 const split=cookie.indexOf("=");
 await cdp("Network.setCookie",{name:cookie.slice(0,split),value:cookie.slice(split+1),url:base,httpOnly:true});
 await navigate("/cours/ml-explainable-ai?section=Exercices");
 await wait("document.querySelector('.python-editor textarea')?.value.includes('def reconstruct')");
 const solution=await (await fetch(base+"/downloads/ch05_cours03_browser_solution.py")).text();
 await fill(solution);await run();
 console.log("Python engine loading in browser...");
 await wait("document.querySelector('.python-results')?.innerText.includes('3/3 tests réussis')",180000);
 console.log("PASS actual Python execution and behavioral checks in browser");
 await fill("def broken(:\n    pass\n");
 await wait("document.querySelector('.python-syntax')?.innerText.includes('Ligne 1')");
 console.log("PASS live syntax errors");
 await fill("raise ValueError('essai de diagnostic')");await run();
 await wait("document.querySelector('.python-results')?.innerText.includes('essai de diagnostic')");
 assert.equal(await evaluate("document.querySelector('.python-results').innerText.includes('Erreur à la ligne 1')"),true);
 console.log("PASS runtime traceback and line number");
 await fill("import matplotlib.pyplot as plt\nplt.plot([1,2,3], [3,1,4])\nprint('graphique OK')\nplt.show()\n");await run();
 await wait("!!document.querySelector('.python-results img')");
 assert.equal(await evaluate("document.querySelector('.python-console').innerText.includes('graphique OK')"),true);
 console.log("PASS console and plot capture");
 await fill("while True:\n    pass\n");await run();
 await wait("document.querySelector('.python-status')?.innerText.includes('Exécution du code')");
 await evaluate("Array.from(document.querySelectorAll('.python-toolbar button')).find(b=>b.textContent.includes('Arrêter')).click()");
 await wait("!document.querySelector('.python-toolbar button').disabled");
 console.log("PASS stop infinite loop");
 const draft="print('brouillon conservé')\n";
 await fill(draft);
 await evaluate("Array.from(document.querySelectorAll('.reader-tabs button')).find(b=>b.textContent==='Cours').click()");
 await wait("!!document.querySelector('.ml-supplied-course')");
 await evaluate("Array.from(document.querySelectorAll('.reader-tabs button')).find(b=>b.textContent==='Exercices').click()");
 await wait("document.querySelector('.python-editor textarea')?.value==="+JSON.stringify(draft));
 await new Promise(r=>setTimeout(r,1200));
 await navigate("/cours/ml-explainable-ai?section=Exercices");
 await wait("document.querySelector('.python-editor textarea')?.value==="+JSON.stringify(draft));
 console.log("PASS draft saved across tabs and reload");
 for(const id of ["ml-conformal-prediction","ml-fairness-responsible-ml","ml-model-diagnostics","ml-drift-monitoring","ml-validation-backtesting","ml-evaluation-governance"]){
  await navigate("/cours/"+id);
  await wait("document.querySelectorAll('.interactive-lab').length>0&&!!document.querySelector('.python-workshop')");
  assert.equal(await evaluate("document.querySelectorAll('.python-editor textarea').length"),1);
  console.log("PASS course and interactive widget",id);
 }
 await navigate("/cours/ml-conformal-prediction");
 await wait("!!document.querySelector('.interactive-lab select')");
 await evaluate("(()=>{const t=document.querySelector('.interactive-lab select');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(t,'95');t.dispatchEvent(new Event('change',{bubbles:true}));})()");
 await wait("document.querySelector('.interactive-lab').innerText.includes('k = 11')");
 assert.equal(await evaluate("document.querySelector('.interactive-lab').innerText.includes('+∞')"),true);
 console.log("PASS conformal infinite quantile at 95%");
 await navigate("/cours/ml-evaluation-governance");
 await wait("!!document.querySelector('.interactive-lab input[type=checkbox]')");
 await evaluate("document.querySelector('.interactive-lab input[type=checkbox]').click()");
 await wait("document.querySelector('.interactive-lab').innerText.includes('Rouge : critères bloquants')");
 await evaluate("document.querySelector('.interactive-lab').scrollIntoView({block:'start'})");
 fs.mkdirSync("outputs",{recursive:true});
 const shot=await cdp("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});
 fs.writeFileSync("outputs/qa-python-desktop.png",Buffer.from(shot.data,"base64"));
 await cdp("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await evaluate("document.querySelector('.python-workshop').scrollIntoView({block:'start'})");
 await new Promise(r=>setTimeout(r,400));
 assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth+1"),true,"mobile layout must not overflow");
 const mobile=await cdp("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});
 fs.writeFileSync("outputs/qa-python-mobile.png",Buffer.from(mobile.data,"base64"));
 assert.deepEqual(exceptions,[]);
 console.log("PASS governance missing metric, responsive layout, no page exceptions");
 console.log("PASS browser QA");
 socket.close();
})().catch(error=>{console.error(error);process.exit(1);});
