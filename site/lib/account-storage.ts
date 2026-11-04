type User = {id:string;name:string;email:string};
let values: Record<string,string> = {};
let currentUserId="";
let queue = Promise.resolve();
let pending = 0;
let failed = false;
export function initializeAccount(data:Record<string,string>,userId="") { currentUserId=userId;values={...data}; pending=0; failed=false; }
function status(){window.dispatchEvent(new CustomEvent("account-save",{detail:{pending,failed}}));}
export async function accountRequest(path:string,body:unknown){
  const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-Recap-User":currentUserId},body:JSON.stringify(body),keepalive:true});
  const data=await response.json() as {error?:string;ok?:boolean;awarded?:number;date:string};
  if(!response.ok)throw new Error(data.error||"La sauvegarde a échoué.");
  return data;
}
export const accountStorage={
  getItem(key:string){return values[key]??null;},
  setItem(key:string,value:string){
    if(values[key]===value)return;
    values[key]=value;
    pending++;status();
    queue=queue.then(async()=>{
      try{await accountRequest("/api/account/record",{key,value});}catch{failed=true;}
      finally{pending--;status();}
    });
  }
};
export async function completeQuiz(lessonId:string,answers:number[]){
  return accountRequest("/api/account/quiz",{lessonId,answers});
}
export type {User};

export async function flushAccount(){await queue;if(failed)throw new Error("La sauvegarde a échoué. Copie tes notes avant de quitter.");}
