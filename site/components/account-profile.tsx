"use client";
import {useContext} from "react";
import {UserRound,LogOut} from "lucide-react";
import {AccountActionsContext,useAccount} from "@/lib/account-context";
export function AccountProfile(){
 const user=useAccount(),actions=useContext(AccountActionsContext);
 if(!user||!actions)return null;
 return <div className="profile"><button className="profile-main" onClick={actions.edit} aria-label={"Modifier le profil de "+user.name}><span className="avatar"><UserRound size={19}/></span><span className="profile-identity"><strong>{user.name}</strong><small>Espace personnel</small></span></button><button className="profile-logout" onClick={actions.logout} disabled={actions.disabled} title="Se déconnecter" aria-label="Se déconnecter"><LogOut size={18}/></button></div>;
}
export function AccountSaveStatus(){
 const actions=useContext(AccountActionsContext);
 return <span className="local-tag" role="status"><span/>{actions?.failed?"Sauvegarde en échec":actions?.saving?"Enregistrement…":"Progression sauvegardée"}</span>;
}
