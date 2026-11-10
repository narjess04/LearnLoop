// Python runs in a dedicated module worker so a loop can be stopped.
const INDEX = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
let python;
let ready;
let output = "";
let figures = [];
function append(text) { if (output.length < 60000) output += text.slice(0, 60000-output.length) + "\n"; }
async function engine() {
 if (!ready) ready = (async () => {
  const { loadPyodide } = await import(INDEX + "pyodide.mjs");
  python = await loadPyodide({ indexURL: INDEX, stdout: append, stderr: append });
  return python;
 })();
 return ready;
}
function moduleNames(text) {
 return [...text.matchAll(/^\s*(?:from|import)\s+(ch\d{2}_cours\d{2}_(?:lab|transformers|solution))\b/gm)].map(m=>m[1]);
}
async function dependencies(source) {
 const files = new Map(), pending = moduleNames(source);
 while (pending.length) {
  const name = pending.shift();
  if (files.has(name)) continue;
  if (files.size >= 16) throw Error("Trop de fichiers auxiliaires.");
  const response = await fetch("/downloads/"+name+".py");
  if (!response.ok) throw Error("Fichier auxiliaire introuvable : "+name+".py");
  const text = await response.text();
  files.set(name, text); pending.push(...moduleNames(text).filter(n=>!files.has(n)));
 }
 return files;
}
const syntaxCode = String.raw`
import ast, json
try:
    ast.parse(__source, filename="mon_code.py")
    __syntax = {"error": None}
except SyntaxError as e:
    __syntax = {"error": e.msg, "line": e.lineno}
json.dumps(__syntax)
`;
const executeCode = String.raw`
import sys, json, traceback, os, io, base64
os.chdir("/home/pyodide")
sys.argv = ["mon_code.py"]
for _name in list(sys.modules):
    if _name.startswith("ch") and "_cours" in _name:
        del sys.modules[_name]
_namespace = {"__name__": "__main__", "__file__": "mon_code.py"}
_result = {"error": None, "checks": [], "line": None}
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.close("all")
    def _capture(*args, **kwargs):
        for _number in plt.get_fignums()[:12]:
            _buffer = io.BytesIO()
            plt.figure(_number).savefig(_buffer, format="png", bbox_inches="tight", dpi=100)
            _figure_callback("data:image/png;base64," + base64.b64encode(_buffer.getvalue()).decode())
        plt.close("all")
    plt.show = _capture
    exec(compile(__source, "mon_code.py", "exec"), _namespace)
    _capture()
except BaseException as _error:
    _result["error"] = "".join(traceback.format_exception(type(_error), _error, _error.__traceback__, limit=6))
    _frames = traceback.extract_tb(_error.__traceback__)
    _own = [f for f in _frames if f.filename == "mon_code.py"]
    _result["line"] = getattr(_error, "lineno", None) or (_own[-1].lineno if _own else None)
_check_scope = {"namespace": _namespace, "exercise": __exercise}
try:
    exec(__checks_source, _check_scope)
    _result["checks"] = _check_scope["results"]
except BaseException as _check_error:
    _result["checks"] = [{"name": "Chargement des tests", "passed": False, "detail": str(_check_error)}]
json.dumps(_result)
`;
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
 queue = queue.then(()=>handle(data)).catch(error=>{
  self.postMessage({type:data.type,id:data.id,error:String(error),engineReady:!!python});
 });
};
async function handle(data) {
 const {type,id,code} = data;
 if(type!=="run"&&type!=="syntax")return;
 const py = await engine();
 py.globals.set("__source",code);
 if(type==="syntax"){
  const result=JSON.parse(py.runPython(syntaxCode));
  self.postMessage({type,id,...result});return;
 }
 output="";figures=[];
 self.postMessage({type:"status",id,message:"Préparation des données et des bibliothèques…"});
 const files=await dependencies(code);
 const all=code+"\n"+[...files.values()].join("\n");
 for(const [name,text] of files)py.FS.writeFile("/home/pyodide/"+name+".py",text);
 await py.loadPackage("matplotlib");
 await py.loadPackagesFromImports(all);
 const response=await fetch("/python/checks.py");
 if(!response.ok)throw Error("Les tests de l’atelier ne sont pas disponibles.");
 py.globals.set("__checks_source",await response.text());
 py.globals.set("__exercise",data.starterUrl?.split("/").pop()?.replace("_starter.py","")||"");
 py.globals.set("_figure_callback",src=>{if(figures.length<12)figures.push(src);});
 self.postMessage({type:"status",id,message:"Exécution du code et vérification…"});
 const result=JSON.parse(await py.runPythonAsync(executeCode));
 self.postMessage({type,id,...result,output,figures,engineReady:true});
}
