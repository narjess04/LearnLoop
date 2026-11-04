import { sqliteTable, text, integer, primaryKey } from "drizzle-orm/sqlite-core";
export const learners = sqliteTable("learners", { id:text("id").primaryKey(), email:text("email").notNull(), name:text("name").notNull() });
export const learningRecords = sqliteTable("learning_records", { userId:text("user_id").notNull().references(()=>learners.id), key:text("key").notNull(), value:text("value").notNull() }, t=>[primaryKey({columns:[t.userId,t.key]})]);
export const quizCompletions = sqliteTable("quiz_completions", { userId:text("user_id").notNull().references(()=>learners.id), lessonId:text("lesson_id").notNull(), date:text("date").notNull() }, t=>[primaryKey({columns:[t.userId,t.lessonId]})]);
export const accountCredentials=sqliteTable("account_credentials",{userId:text("user_id").primaryKey().references(()=>learners.id),email:text("email").notNull().unique(),passwordHash:text("password_hash").notNull()});
export const accountSessions=sqliteTable("account_sessions",{hash:text("hash").primaryKey(),userId:text("user_id").notNull().references(()=>learners.id),expiresAt:integer("expires_at").notNull()});
export const authLimits=sqliteTable("auth_limits",{key:text("key").primaryKey(),count:integer("count").notNull(),startedAt:integer("started_at").notNull()});
