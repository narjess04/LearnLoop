import { notFound } from "next/navigation";
import { lessons } from "@/lib/curriculum";
import LessonReader from "@/components/lesson-reader";
export default async function CoursePage({params}:{params:Promise<{id:string}>}){const {id}=await params;const lesson=lessons.find(l=>l.id===id);if(!lesson)notFound();return <LessonReader key={lesson.id} lesson={lesson}/>;}
