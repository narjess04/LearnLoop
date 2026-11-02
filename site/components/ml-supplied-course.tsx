"use client";
import CourseText from "@/components/course-text";
import MLCourseWidget from "@/components/ml-course-widgets";
import chapterOne from "@/lib/ml-chapter-one.json";
export type SuppliedCourseId=keyof typeof chapterOne.courses;
export default function MLSuppliedCourse({courseId}:{courseId:SuppliedCourseId}){
 const text=chapterOne.courses[courseId];const config=chapterOne.config[courseId];
 return <article className="ml-supplied-course"><div className="ml-course-intro"><span className="eyebrow">MACHINE LEARNING · CHAPITRE 01 · COURS {String(config.number).padStart(2,"0")}</span><p>{chapterOne.lessons.find(l=>l.id===courseId)?.goal}</p><div><span>{config.duration} indicatives</span><span>Théorie + Python</span><span>Entretien technique</span></div></div>{text.split(/(\[\[[a-z]+\]\])/g).map((part,i)=>part.startsWith("[[")?<MLCourseWidget key={i} kind={part.slice(2,-2)} courseId={courseId}/>:<CourseText key={i} text={part}/>)}<div className="course-downloads"><strong>Scripts de travail</strong><a href={config.lab} download>Télécharger le Coding Lab Python</a><a href={config.starter} download>Télécharger le squelette du projet</a></div></article>;
}
