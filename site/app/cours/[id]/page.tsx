import { notFound } from "next/navigation";
import { headers } from "next/headers";
import { env } from "cloudflare:workers";
import Link from "next/link";
import { LockKeyhole } from "lucide-react";
import { lessons } from "@/lib/curriculum";
import { blockingLesson, chapterNumber, blockingChapter } from "@/lib/learning-paths";
import { sessionUser } from "@/lib/password-auth";
import LessonReader from "@/components/lesson-reader";
export const dynamic="force-dynamic";
export default async function CoursePage({params}:{params:Promise<{id:string}>}){
 const {id}=await params;const lesson=lessons.find(l=>l.id===id);if(!lesson)notFound();
 const requestHeaders=await headers();
 const user=env.DB?await sessionUser(new Request("https://learnloop.local",{headers:requestHeaders}),env.DB):null;
 if(!user)return null;
 const completed=await env.DB!.prepare("SELECT lesson_id FROM quiz_completions WHERE user_id=?").bind(user.id).all<{lesson_id:string}>();
 const passed=completed.results.map(r=>r.lesson_id);
 const chapterBlock=blockingChapter(chapterNumber(id)||1,passed);
 if(chapterBlock)return <main className="course-access-lock"><LockKeyhole size={30}/><h1>Ce chapitre est encore verrouillé.</h1><p>Termine tous les cours du chapitre {chapterBlock.number} — <strong>{chapterBlock.title}</strong> — avant d’accéder à celui-ci.</p><Link className="button-dark" href={"/parcours/machine-learning?chapitre="+chapterBlock.number}>Reprendre le chapitre à terminer →</Link></main>;
 const blocker=blockingLesson(id,passed);
 if(blocker)return <main className="course-access-lock"><LockKeyhole size={30}/><h1>Ce cours est encore verrouillé.</h1><p>Termine d’abord le quiz de <strong>{blocker.title}</strong>. Le cours suivant sera débloqué automatiquement.</p><Link className="button-dark" href={"/cours/"+blocker.id}>Reprendre le cours à terminer →</Link><Link className="text-button" href={"/parcours/machine-learning?chapitre="+(chapterNumber(id)||1)}>Voir la roadmap</Link></main>;
 return <LessonReader key={lesson.id} lesson={lesson}/>;
}
