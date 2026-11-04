import { lessons, type Lesson } from "@/lib/curriculum";
import { mlChapters } from "@/lib/ml-roadmap";
import chapterTwo from "@/lib/ml-chapter-two.json";
import chapterThree from "@/lib/ml-chapter-three.json";
import chapterFour from "@/lib/ml-chapter-four.json";
export type CourseNode={id?:string;title:string;summary:string;minutes:number};
export function chapterCourses(number:number):CourseNode[]{
 if(number===4)return chapterFour.program.map(p=>({id:p.id,title:p.title,summary:p.summary,minutes:p.minutes}));
 const chapter=mlChapters.find(c=>c.number===number);if(!chapter)return [];
 const titles=number===2?chapterTwo.titles:number===3?chapterThree.titles:null;
 if(titles)return titles.map((title,i)=>{const lesson=lessons.find(l=>l.id===chapter.lessonIds[i]);return {id:lesson?.id,title,summary:lesson?.goal||"Support détaillé à venir.",minutes:lesson?.minutes||0};});
 return chapter.lessonIds.map(id=>lessons.find(l=>l.id===id)).filter((l):l is Lesson=>!!l).map(l=>({id:l.id,title:l.title.replace(/^Cours \d+ — /,""),summary:l.goal,minutes:l.minutes}));
}
export function blockingLesson(courseId:string,passed:readonly string[]):Lesson|undefined{
 if(passed.includes(courseId))return undefined;
 const lesson=lessons.find(l=>l.id===courseId);if(!lesson)return undefined;
 const sequence=lesson.module==="ml"?mlChapters.flatMap(c=>c.lessonIds):lessons.filter(l=>l.module===lesson.module).map(l=>l.id);
 const index=sequence.indexOf(courseId);
 if(index<0)return undefined;
 const missing=sequence.slice(0,index).find(id=>!passed.includes(id));
 return lessons.find(l=>l.id===missing);
}
export function courseUnlocked(courseId:string,passed:readonly string[]){return !blockingLesson(courseId,passed);}
export function chapterNumber(courseId:string){return mlChapters.find(c=>c.lessonIds.some(id=>id===courseId))?.number;}

export function blockingChapter(number:number,passed:readonly string[]){
 return mlChapters.filter(c=>c.number<number).find(chapter=>{
  const nodes=chapterCourses(chapter.number);
  return !nodes.length||nodes.some(node=>!node.id||!passed.includes(node.id));
 });
}
