import AccountProvider from "@/components/account-provider";
import "./accounts.css";
import type { Metadata } from "next";
import "./globals.css";
import "./reader.css";
import "./assistant.css";
import "./learning-paths.css";
import "./ml-learning.css";
import "./ml-labs.css";
import "./ml-advanced-labs.css";
import "katex/dist/katex.min.css";
export const metadata: Metadata = { title: "LearnLoop — Comprendre. Pratiquer. Maîtriser.", description: "Ton espace personnel pour apprendre le machine learning, le deep learning et l’IA générative en profondeur.", icons: {icon:"/favicon.svg?v=learnloop", shortcut:"/favicon.svg?v=learnloop"} };
export default function RootLayout({children}:Readonly<{children:React.ReactNode}>){return <html lang="fr"><body><AccountProvider>{children}</AccountProvider></body></html>;}
