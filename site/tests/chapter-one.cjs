const fs=require("node:fs"),ts=require("typescript"),assert=require("node:assert/strict"),katex=require("katex");
require.extensions[".ts"]=(m,n)=>m._compile(ts.transpileModule(fs.readFileSync(n,"utf8"),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,n);
const {lessons,projects}=require("../lib/curriculum.ts");
const {mlChapters}=require("../lib/ml-roadmap.ts");
const data=require("../lib/ml-chapter-one.json");
assert.deepEqual(lessons.filter(l=>l.module==="ml").map(l=>l.id),["ml-introduction","ml-first-model","ml-learning-optimization","ml-generalization","ml-validation-strategies","ml-data-leakage","ml-hyperparameter-tuning","ml-project-methodology"]);
for(const id of ["linear","validation","metrics"])assert(!lessons.some(l=>l.id===id));
assert.deepEqual(mlChapters.map(c=>c.number),Array.from({length:12},(_,i)=>i+1));
assert.deepEqual(mlChapters[0].lessonIds,["ml-introduction","ml-first-model","ml-learning-optimization","ml-generalization","ml-validation-strategies","ml-data-leakage","ml-hyperparameter-tuning","ml-project-methodology"]);
for(const chapter of mlChapters)for(const id of chapter.lessonIds)assert(lessons.some(l=>l.id===id));
assert(projects.some(p=>p.id==="capstone-chapter-one"));assert(projects.some(p=>p.id==="wine"));assert(projects.some(p=>p.id==="numpy-multivariate"));
let formulas=0;
for(const text of Object.values(data.courses)){
 for(const m of text.matchAll(/\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g)){katex.renderToString(m[1]||m[2],{throwOnError:true,trust:false});formulas++;}
 const blocks=Array.from(text.matchAll(/~~~python\n[\s\S]*?~~~/g));
 assert(blocks.length>0);
 assert(!text.includes("import numpy as nprng"));assert(!text.includes("import pandas as pdimport"));
}
console.log("Catalog, 12 chapters, two projects and "+formulas+" mathematical expressions verified.");
