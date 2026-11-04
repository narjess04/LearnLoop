"use client";
import {createContext,useContext} from "react";
import type {User} from "@/lib/account-storage";
export type AccountActions={edit:()=>void;logout:()=>void;disabled:boolean;saving:boolean;failed:boolean};
export const AccountContext=createContext<User|null>(null);
export const AccountActionsContext=createContext<AccountActions|null>(null);
export function useAccount(){return useContext(AccountContext);}
