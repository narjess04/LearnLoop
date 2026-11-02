import chapterTwo from "@/lib/ml-chapter-two.json";
import chapterThree from "@/lib/ml-chapter-three.json";

export const richCourses = {...chapterTwo.courses, ...chapterThree.courses};
export type RichCourseId = keyof typeof richCourses;
export function richCourseChapter(courseId:RichCourseId):2|3 {
  return courseId in chapterThree.courses ? 3 : 2;
}
