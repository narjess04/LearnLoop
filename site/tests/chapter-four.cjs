const fs=require("node:fs"),ts=require("typescript"),assert=require("node:assert/strict"),Module=require("node:module"),path=require("node:path"),katex=require("katex"),cp=require("node:child_process");
const resolve=Module._resolveFilename;
Module._resolveFilename=function(id,...args){return resolve.call(this,id.startsWith("@/")?path.join(__dirname,"..",id.slice(2)):id,...args);};
require.extensions[".ts"]=(m,n)=>m._compile(ts.transpileModule(fs.readFileSync(n,"utf8"),{compilerOptions:{esModuleInterop:true,module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,n);
const {lessons,projects}=require("../lib/curriculum.ts");
const {mlChapters}=require("../lib/ml-roadmap.ts");
const {chapterCourses,blockingLesson,blockingChapter}=require("../lib/learning-paths.ts");
const data=require("../lib/ml-chapter-four.json");
assert.equal(data.program.length,10);
assert.equal(data.program.reduce((s,p)=>s+p.minutes,0),690);
assert.equal(data.lessons.length,10);assert.equal(new Set(lessons.map(l=>l.id)).size,lessons.length);
assert.deepEqual(mlChapters.find(c=>c.number===4).lessonIds,data.lessons.map(l=>l.id));
assert.equal(chapterCourses(4).filter(c=>c.id).length,10);
assert(projects.some(p=>p.id==="customer-segmentation-anomaly"));
const ordered=mlChapters.flatMap(c=>c.lessonIds);
assert.equal(blockingLesson(ordered[0],[]),undefined);
assert.equal(blockingLesson(ordered[1],[]).id,ordered[0]);
assert.equal(blockingLesson(ordered[1],[ordered[0]]),undefined);
assert.equal(blockingLesson(data.lessons[0].id,[]).id,ordered[0]);
const prefix=ordered.slice(0,ordered.indexOf(data.lessons[0].id));
assert.equal(blockingLesson(data.lessons[0].id,prefix),undefined);
assert.equal(blockingLesson(data.lessons[1].id,prefix).id,data.lessons[0].id);
assert.equal(blockingLesson(data.lessons[1].id,[...prefix,data.lessons[0].id]),undefined);
assert.equal(blockingLesson(data.lessons[2].id,[data.lessons[2].id]),undefined);
assert.equal(blockingChapter(1,[]),undefined);
assert.equal(blockingChapter(2,[]).number,1);
assert.equal(blockingChapter(2,mlChapters[0].lessonIds),undefined);
assert.equal(blockingChapter(4,ordered.slice(0,ordered.indexOf(data.lessons[0].id))),undefined);
assert.equal(blockingChapter(5,ordered),undefined);
assert.equal(blockingChapter(6,ordered).number,5);
const blocks=[];let formulas=0;
for(const c of Object.values(data.courses)){
 assert(c.text.includes("[[quiz]]")&&c.text.includes("[[assessment]]"));
 assert(c.objectives.length>=6);
 assert(c.quiz.every(q=>q.answer>=0&&q.answer<q.options.length));
 assert(!/pltfrom|npclass|0250500750|0 \/ 4 réponses/.test(c.text));
 for(const m of c.text.matchAll(/\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g)){katex.renderToString(m[1]||m[2],{throwOnError:true,trust:false});formulas++;}
 for(const m of c.text.matchAll(/~~~python\n([\s\S]*?)~~~/g))blocks.push(m[1]);
 for(const suffix of ["lab","starter","solution"]){const f="ch04_cours"+String(c.number).padStart(2,"0")+"_"+suffix+".py";assert.equal(fs.readFileSync(path.join(__dirname,"../public/downloads",f),"utf8"),fs.readFileSync(path.join(__dirname,"../../03_Machine_learning/CHAPITRE_04",f),"utf8"));}
}
const python=path.join(__dirname,"../.venv-learning/Scripts/python.exe");
const result=cp.spawnSync(python,["-B","-c","import ast,json,sys; [ast.parse(s) for s in json.load(sys.stdin)]"],{input:JSON.stringify(blocks),encoding:"utf8"});
assert.equal(result.status,0,result.stderr);
console.log("Verified: 10-course roadmap, 690 minutes, 10 available courses, sequential access including chapter boundaries, "+formulas+" formulas, "+blocks.length+" Python blocks, downloads and quizzes.");
