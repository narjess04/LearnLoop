import LearnLoopLogo from "@/components/learnloop-logo";
import Link from "next/link";
import Tutor from "@/components/tutor";
export default function TutorPage(){return <div className="reader-shell tutor-page"><header className="reader-top"><Link href="/" className="reader-brand"><LearnLoopLogo/></Link><Link href="/">← Mon espace</Link></header><main className="tutor-main"><p className="eyebrow">APPRENDRE EN POSANT DES QUESTIONS</p><h1>Ton tuteur <span>Python & IA.</span></h1><p className="reader-intro">Demande une explication, un indice ou une revue de ton raisonnement. Pour une aide liée au cours, ouvre le tuteur depuis la leçon.</p><Tutor/></main></div>;}
