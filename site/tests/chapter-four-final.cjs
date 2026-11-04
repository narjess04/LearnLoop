const fs=require("node:fs"),assert=require("node:assert/strict"),crypto=require("node:crypto");
const ts=require("typescript"),Module=require("node:module"),pathModule=require("node:path");
const resolveFile=Module._resolveFilename;
Module._resolveFilename=function(id,...args){return resolveFile.call(this,id.startsWith("@/")?pathModule.join(__dirname,"..",id.slice(2)):id,...args);};
require.extensions[".ts"]=(m,n)=>m._compile(ts.transpileModule(fs.readFileSync(n,"utf8"),{compilerOptions:{esModuleInterop:true,module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,n);
const {lessons}=require("../lib/curriculum.ts"),{richCourses}=require("../lib/ml-rich-courses.ts"),{mlChapters}=require("../lib/ml-roadmap.ts");
const base="http://127.0.0.1:5173";
async function request(path,body,cookie="",user=""){
 const response=await fetch(base+path,{method:body?"POST":"GET",headers:{...(cookie?{Cookie:cookie}:{}),...(body?{"Content-Type":"application/json",Origin:base,"X-Recap-User":user}:{})},...(body?{body:JSON.stringify(body)}:{})});
 const data=await response.json();return {response,data};
}
(async()=>{
 const suffix=crypto.randomUUID(),password="Test-"+suffix;
 const registration=await request("/api/account/auth/register",{name:"Test LearnLoop",email:"ui-"+suffix+"@example.invalid",password});
 assert.equal(registration.response.status,200,JSON.stringify(registration.data));
 const cookie=registration.response.headers.get("set-cookie").split(";")[0];
 const account=await request("/api/account",undefined,cookie);
 const first=await request("/api/account/quiz",{lessonId:"ml-introduction",answers:[1]},cookie,account.data.user.id);
 assert.equal(first.response.status,200,JSON.stringify(first.data));
 const blockedQuiz=await request("/api/account/quiz",{lessonId:"ml-learning-optimization",answers:[0]},cookie,account.data.user.id);
 assert.equal(blockedQuiz.response.status,403);
 const page=await (await fetch("http://127.0.0.1:9333/json/new?about:blank",{method:"PUT"})).json();
 const socket=new WebSocket(page.webSocketDebuggerUrl),pending=new Map();let nextId=0;
 await new Promise((resolve,reject)=>{socket.addEventListener("open",resolve,{once:true});socket.addEventListener("error",reject,{once:true});});
 const exceptions=[];
 socket.addEventListener("message",event=>{const message=JSON.parse(event.data);if(message.id){const p=pending.get(message.id);pending.delete(message.id);message.error?p.reject(Error(JSON.stringify(message.error))):p.resolve(message.result);}if(message.method==="Runtime.exceptionThrown")exceptions.push(message.params.exceptionDetails.text);});
 function cdp(method,params={}){return new Promise((resolve,reject)=>{const id=++nextId;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});}
 async function evaluate(expression){const r=await cdp("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
 async function wait(expression){const start=Date.now();while(Date.now()-start<30000){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,250));}throw Error("Timeout: "+expression+"; page="+await evaluate("document.body.innerText.slice(0,1200)"));}
 await cdp("Page.enable");await cdp("Runtime.enable");await cdp("Network.enable");
 await cdp("Network.setCookie",{name:"recap_session",value:cookie.split("=")[1],url:base,httpOnly:true,sameSite:"Lax"});
 await cdp("Emulation.setDeviceMetricsOverride",{width:1440,height:960,deviceScaleFactor:1,mobile:false});
 const sequence=mlChapters.flatMap(c=>c.lessonIds);
 async function complete(id){const rich=richCourses[id],lesson=lessons.find(l=>l.id===id);const result=await request("/api/account/quiz",{lessonId:id,answers:rich?rich.quiz.map(q=>q.answer):[lesson.answer]},cookie,account.data.user.id);assert.equal(result.response.status,200,id+JSON.stringify(result.data));}
 for(const id of sequence.slice(0,sequence.indexOf("ml-pca")))await complete(id);
 async function navigate(id){await cdp("Page.navigate",{url:base+"/cours/"+id});await wait("!!document.querySelector('.reader-main')&&document.querySelectorAll('.ml-scenarios fieldset').length===5");}
 await navigate("ml-pca");assert.equal(await evaluate("document.querySelectorAll('.interactive-lab input[type=range]').length"),2);
 await evaluate("document.querySelector('input[aria-label=\"Angle de projection en degrés\"]').focus()");
 await cdp("Input.dispatchKeyEvent",{type:"keyDown",key:"ArrowRight",code:"ArrowRight",windowsVirtualKeyCode:39});await cdp("Input.dispatchKeyEvent",{type:"keyUp",key:"ArrowRight",code:"ArrowRight",windowsVirtualKeyCode:39});
 await wait("document.querySelector('input[aria-label=\"Angle de projection en degrés\"]').value==='1'");
 const choices=richCourses["ml-pca"].quiz.map(q=>q.answer);
 await evaluate("(()=>{const a="+JSON.stringify(choices)+";document.querySelectorAll('.ml-scenarios fieldset').forEach((f,i)=>f.querySelectorAll('input')[a[i]].click());document.querySelector('.ml-scenarios button').click();})()");
 await wait("document.querySelector('.ml-scenarios [role=status]')?.textContent.includes('5/5')");
 async function saved(key,expression){for(let i=0;i<50;i++){const a=await request('/api/account',undefined,cookie);const value=JSON.parse(a.data.values[key]||'null');if(expression(value))return;await new Promise(r=>setTimeout(r,100));}throw Error('Sauvegarde non trouvée : '+key);}
 await saved('recap-learning-v1',v=>v?.passed.includes('ml-pca'));
 await navigate("ml-tsne-umap");assert.equal(await evaluate("document.querySelectorAll('.interactive-lab input[type=range]').length"),2);
 await evaluate("document.querySelector('input[aria-label=n_neighbors]').focus()");await cdp("Input.dispatchKeyEvent",{type:"keyDown",key:"ArrowRight",code:"ArrowRight",windowsVirtualKeyCode:39});await cdp("Input.dispatchKeyEvent",{type:"keyUp",key:"ArrowRight",code:"ArrowRight",windowsVirtualKeyCode:39});await wait("document.querySelector('.interactive-lab pre').textContent.includes('n_neighbors=16')");
 await complete("ml-tsne-umap");await navigate("ml-anomaly-detection");assert.equal(await evaluate("document.querySelectorAll('.interactive-lab input[type=range]').length"),1);
 await complete("ml-anomaly-detection");await navigate("ml-unsupervised-pipelines");
 assert.equal(await evaluate("document.querySelectorAll('.interactive-lab input[type=range]').length"),1);
 assert.equal(await evaluate("document.querySelector('.ml-assessment').querySelectorAll('input[type=checkbox]').length"),15);
 await evaluate("document.querySelector('.ml-assessment input[type=checkbox]').click()");
 await wait("document.querySelector('.ml-assessment p').textContent.includes('1/15')");
 await saved('recap-ch04-project-final-v1',v=>v?.[0]===true);
 await cdp("Page.reload");await wait("document.querySelector('.ml-assessment input[type=checkbox]')?.checked===true");
 for(const id of ["ml-pca","ml-tsne-umap","ml-anomaly-detection","ml-unsupervised-pipelines"]){
  await cdp("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});await navigate(id);
  assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth+1"),true,id+" mobile overflow");
  assert.equal(await evaluate("document.querySelectorAll('.katex-error').length"),0,id+" KaTeX");
  assert.equal(await evaluate("document.querySelectorAll('.course-downloads a[download]').length"),3);
 }
 await complete("ml-unsupervised-pipelines");const repeat=await request("/api/account/quiz",{lessonId:"ml-unsupervised-pipelines",answers:richCourses["ml-unsupervised-pipelines"].quiz.map(q=>q.answer)},cookie,account.data.user.id);assert.equal(repeat.data.awarded,0);
 for(let n=7;n<=10;n++)for(const suffix of ['lab','starter','solution']){const filename='ch04_cours'+String(n).padStart(2,'0')+'_'+suffix+'.py';const response=await fetch(base+'/downloads/'+filename);assert.equal(response.status,200);assert.equal(await response.text(),fs.readFileSync(pathModule.join(__dirname,'../public/downloads',filename),'utf8'));}
 assert.equal(exceptions.length,0,exceptions.join("\n"));
 const shot=await cdp("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});fs.writeFileSync("outputs/qa-ch4-final-mobile.png",Buffer.from(shot.data,"base64"));
 await cdp("Browser.close").catch(()=>{});socket.close();
 console.log("Browser verified: 4 new course routes, ordered access, PCA/UMAP sliders, quiz UI/server, 15-item checklist persistence, 12 downloads, math/mobile layouts; no runtime exceptions.");
})().catch(error=>{console.error(error);process.exit(1);});