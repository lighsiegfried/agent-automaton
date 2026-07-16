"""Activity Center API + localhost UI (Phase 5D).

    GET  /activity                     (the localhost web UI — no CDNs, no Electron)
    GET  /activity/overview            GET  /activity/events
    GET  /activity/events/{id}         GET  /activity/pending
    GET  /activity/tasks               GET  /activity/schedules
    GET  /activity/services            POST /activity/actions/confirm
    POST /activity/actions/cancel      POST /activity/services/restart
    POST /activity/services/unload     GET  /activity/export

Read endpoints aggregate safe, redacted data. Control endpoints route through the
existing services and the one global broker: confirm dispatches the domain's exact
phrase (there is no generic "confirm everything"), cancel uses the broker's own
cancel, and restart/unload touch OWNED services only. Task/schedule controls are the
existing /tasks/* and /schedules/* endpoints, which the UI calls directly.
"""

import ipaddress
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.activity.service import get_activity_service
from app.config import get_settings

router = APIRouter(prefix="/activity", tags=["activity"])


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return str(host).lower() in ("localhost",)


class PhraseRequest(BaseModel):
    phrase: str = Field(default="")


class ServiceRequest(BaseModel):
    service: str = Field(default="")


def _svc():
    return get_activity_service()


def _enabled_guard():
    if not get_settings().enable_activity_center:
        return {"error_code": "ACTIVITY_DISABLED"}
    return None


@router.get("", response_class=HTMLResponse)
def activity_ui() -> str:
    return ACTIVITY_HTML


@router.get("/overview")
def overview() -> dict[str, Any]:
    return _enabled_guard() or _svc().overview()


@router.get("/events")
def events(domain: str = "", severity: str = "", status: str = "", since: str = "",
           until: str = "", related_id: str = "", page: int = 1, page_size: int = 0) -> dict[str, Any]:
    guard = _enabled_guard()
    if guard:
        return guard
    return _svc().events(domain=domain or None, severity=severity or None, status=status or None,
                         since=since or None, until=until or None, related_id=related_id or None,
                         page=page, page_size=page_size or None)


@router.get("/events/{event_id}")
def event(event_id: str) -> dict[str, Any]:
    data = _svc().event(event_id)
    return {"event": data} if data else {"event": None, "error_code": "EVENT_NOT_FOUND"}


@router.get("/pending")
def pending() -> dict[str, Any]:
    return _enabled_guard() or _svc().pending()


@router.get("/tasks")
def tasks() -> dict[str, Any]:
    return _enabled_guard() or _svc().tasks()


@router.get("/schedules")
def schedules() -> dict[str, Any]:
    return _enabled_guard() or _svc().schedules()


@router.get("/services")
def services() -> dict[str, Any]:
    return _enabled_guard() or _svc().services()


@router.post("/actions/confirm")
def confirm(request: PhraseRequest) -> dict[str, Any]:
    return _svc().confirm_action(request.phrase)


@router.post("/actions/cancel")
def cancel() -> dict[str, Any]:
    return _svc().cancel_action()


@router.post("/services/restart")
def restart(request: ServiceRequest) -> dict[str, Any]:
    return _svc().restart_service(request.service)


@router.post("/services/unload")
def unload() -> dict[str, Any]:
    return _svc().unload()


@router.get("/export")
def export(limit: int = 0) -> dict[str, Any]:
    return _svc().export(limit=limit or None)


# --- localhost UI (self-contained: inline CSS + vanilla JS, no external resources) --

ACTIVITY_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fifi Activity Center</title>
<style>
:root{--bg:#0f1115;--card:#1a1e26;--fg:#e6e8ec;--mut:#8a92a3;--acc:#4da3ff;--err:#e0574d;--warn:#e0a83b;--ok:#3fa860}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
header{padding:.8rem 1.2rem;background:#12151b;border-bottom:1px solid #232833;display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
h1{font-size:1.05rem;margin:0}.pill{padding:.15rem .55rem;border-radius:1rem;font-size:.78rem;background:#232833}
.pill.ready{background:#16351f;color:#7fe0a0}.pill.degraded{background:#3a2a12;color:#ffcf7a}
main{padding:1rem;display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1rem}
section{background:var(--card);border:1px solid #232833;border-radius:.6rem;padding:.8rem 1rem;overflow:auto;max-height:70vh}
section h2{font-size:.9rem;margin:.1rem 0 .6rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.row{padding:.4rem 0;border-bottom:1px solid #21262f}.row:last-child{border:0}
.muted{color:var(--mut)}.err{color:var(--err)}.warn{color:var(--warn)}.ok{color:var(--ok)}
button{background:#232833;color:var(--fg);border:1px solid #313847;border-radius:.4rem;padding:.25rem .6rem;cursor:pointer;font-size:.8rem;margin:.15rem .2rem 0 0}
button:hover{border-color:var(--acc)}button.primary{border-color:var(--acc);color:var(--acc)}
input,select{background:#12151b;color:var(--fg);border:1px solid #313847;border-radius:.35rem;padding:.2rem .4rem;font-size:.8rem}
code{background:#12151b;padding:.05rem .3rem;border-radius:.3rem;color:#9fd0ff}
.tag{font-size:.72rem;padding:.05rem .35rem;border-radius:.3rem;background:#232833;color:var(--mut)}
.filters{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.5rem}
</style></head><body>
<header><h1>Fifi Activity Center</h1><span id="fifi" class="pill">…</span>
<span class="muted" id="modeline"></span><span style="flex:1"></span>
<span class="muted" id="clock"></span><button onclick="refreshAll()">Refresh</button></header>
<main>
<section><h2>Overview</h2><div id="overview">…</div></section>
<section><h2>Pending confirmations</h2><div id="pending">…</div></section>
<section><h2>Active tasks</h2><div id="tasks">…</div></section>
<section><h2>Upcoming schedules</h2><div id="schedules">…</div></section>
<section><h2>Runtime / services</h2><div id="services">…</div></section>
<section><h2>Errors</h2><div id="errors">…</div></section>
<section><h2>Memory activity</h2><div id="memory">…</div></section>
<section style="grid-column:1/-1"><h2>Recent activity</h2>
<div class="filters">
<select id="fdomain" onchange="loadEvents(1)"><option value="">all domains</option></select>
<select id="fseverity" onchange="loadEvents(1)"><option value="">all severity</option>
<option>info</option><option>warning</option><option>error</option></select>
<button onclick="loadEvents(page-1)">‹ prev</button><span id="pageinfo" class="muted"></span>
<button onclick="loadEvents(page+1)">next ›</button></div>
<div id="events">…</div></section>
</main>
<script>
let page=1;
const esc=s=>String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const j=async(u,o)=>{try{const r=await fetch(u,o);return await r.json()}catch(e){return{error:String(e)}}};
const post=(u,b)=>j(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b||{})});
function sev(s){return s==="error"?"err":s==="warning"?"warn":"muted"}
async function loadOverview(){const o=await j("/activity/overview");if(o.error){document.getElementById("overview").innerHTML='<span class="err">unavailable</span>';return}
const f=document.getElementById("fifi");f.textContent="Fifi: "+(o.fifi||"?");f.className="pill "+(o.fifi||"");
document.getElementById("modeline").textContent=`mode: ${o.interaction_mode} · voice: ${o.active_voice} · wake: ${o.wake}`;
const svc=o.services||{};const ram=(o.resources||{}).ram;
document.getElementById("overview").innerHTML=
`<div class="row">Active task: ${o.active_task?esc(o.active_task.title)+" <span class='tag'>"+esc(o.active_task.progress)+"</span>":"<span class='muted'>none</span>"}</div>`+
`<div class="row">Next schedule: ${o.next_schedule?esc(o.next_schedule.title)+" <span class='muted'>"+esc((o.next_schedule.next_run_at||"").slice(0,16))+"</span>":"<span class='muted'>none</span>"}</div>`+
`<div class="row">Pending: ${o.pending?"<span class='warn'>"+esc(o.pending.domain)+"</span>":"<span class='muted'>none</span>"}</div>`+
`<div class="row">Recent issues (24h): <span class="${o.recent_errors?'err':'muted'}">${o.recent_errors}</span></div>`+
`<div class="row">RAM: ${ram?ram.used_pct+"% of "+ram.total_gb+"GB":"<span class='muted'>n/a</span>"}</div>`;
renderErrors(o.errors||[]);}
async function loadPending(){const p=(await j("/activity/pending")).pending;const el=document.getElementById("pending");
if(!p){el.innerHTML='<span class="muted">Nothing awaiting confirmation.</span>';return}
const rec=p.recipient?` · recipient: <b>${esc(p.recipient)}</b>`:"";
const cp=(p.confirm_phrase||"").replace(/'/g,"\\'");
el.innerHTML=`<div class="row"><b>${esc(p.domain)}</b>${rec}<br><span class="muted">target:</span> ${esc(p.target||"")}<br>
say <code>${esc(p.required_confirmation_phrase)}</code>
<div><button class="primary" onclick="confirmPending('${cp}')">Confirm this action</button>
<button onclick="cancelPending()">Cancel</button></div></div>`;}
async function confirmPending(ph){await post("/activity/actions/confirm",{phrase:ph});refreshAll()}
async function cancelPending(){await post("/activity/actions/cancel",{});refreshAll()}
async function loadTasks(){const t=await j("/activity/tasks");const el=document.getElementById("tasks");
if(t.status==="disabled"){el.innerHTML='<span class="muted">Tasks disabled.</span>';return}
const a=t.active;if(!a){el.innerHTML='<span class="muted">No active task.</span>';return}
el.innerHTML=`<div class="row"><b>${esc(a.title)}</b> <span class="tag">${esc(a.status)}</span><br>
progress ${esc(a.progress)} · step ${esc(a.current_step)}${a.blocked_reason?" · blocked: <code>"+esc(a.blocked_reason)+"</code>":""}<br>
<span class="muted">${(a.audit||[]).join(" · ")}</span>
<div><button onclick="taskCtl('${esc(a.task_id)}','resume')">Resume</button>
<button onclick="taskCtl('${esc(a.task_id)}','cancel')">Cancel</button></div></div>`;}
async function taskCtl(id,act){if(act==="cancel")await post(`/tasks/${id}/cancel`,{});else await post(`/tasks/${id}/resume`,{});refreshAll()}
async function loadSchedules(){const s=await j("/activity/schedules");const el=document.getElementById("schedules");
if(s.status==="disabled"){el.innerHTML='<span class="muted">Schedules disabled.</span>';return}
const list=s.schedules||[];if(!list.length){el.innerHTML='<span class="muted">No schedules.</span>';return}
el.innerHTML=list.slice(0,8).map(x=>`<div class="row"><b>${esc(x.title)}</b> <span class="tag">${esc(x.status)}</span><br>
next ${esc((x.next_run_at||"").slice(0,16))} · ${x.recurrence?"recurring":"once"}<br>
<span class="muted">${esc(x.last_result||"")}</span>
<div>${x.status==="active"?`<button onclick="schedCtl('${esc(x.schedule_id)}','pause')">Pause</button>`:`<button onclick="schedCtl('${esc(x.schedule_id)}','resume')">Resume</button>`}
<button onclick="schedCtl('${esc(x.schedule_id)}','cancel')">Cancel</button></div></div>`).join("");}
async function schedCtl(id,act){await post(`/schedules/${id}/${act}`,{});refreshAll()}
async function loadServices(){const s=(await j("/activity/services")).services||{};
document.getElementById("services").innerHTML=Object.entries(s).map(([k,v])=>
`<div class="row">${esc(k)}: <span class="${v.status==='ok'||v.status==='enabled'?'ok':v.status==='disabled'?'muted':'err'}">${esc(v.status)}</span>
${v.status==='unavailable'?`<button onclick="restart('${esc(k)}')">Restart</button>`:""}</div>`).join("")+
`<div class="row"><button onclick="unload()">Release VRAM</button></div>`;}
async function restart(s){await post("/activity/services/restart",{service:s});refreshAll()}
async function unload(){await post("/activity/services/unload",{});refreshAll()}
function renderErrors(items){const el=document.getElementById("errors");
if(!items.length){el.innerHTML='<span class="muted">No recent errors.</span>';return}
el.innerHTML=items.map(e=>`<div class="row"><span class="${sev(e.severity)}">${esc(e.label)}</span> · ${esc(e.domain)}<br>
<span class="muted">${esc(e.title)} — ${esc(e.summary)}</span></div>`).join("");}
async function loadEvents(p){page=Math.max(1,p||1);const d=document.getElementById("fdomain").value,sv=document.getElementById("fseverity").value;
const r=await j(`/activity/events?page=${page}&domain=${d}&severity=${sv}`);const ev=r.events||[];
document.getElementById("pageinfo").textContent=`page ${r.page||1} · ${r.total||0} events`;
document.getElementById("events").innerHTML=ev.length?ev.map(e=>`<div class="row">
<span class="muted">${esc(e.ts.slice(11,19))}</span> <span class="tag">${esc(e.domain)}</span>
<span class="${sev(e.severity)}">${esc(e.title)}</span> — ${esc(e.summary)}${e.related_id?` <code>${esc(e.related_id)}</code>`:""}</div>`).join(""):'<span class="muted">No events.</span>';
const mem=ev.filter(e=>e.domain==="memory");document.getElementById("memory").innerHTML=mem.length?mem.map(e=>`<div class="row"><span class="muted">${esc(e.ts.slice(11,19))}</span> ${esc(e.summary)}</div>`).join(""):'<span class="muted">No memory activity.</span>';
const sel=document.getElementById("fdomain");if(sel.options.length<=1){["command","pending","memory","tasks","schedules","browser","text","whatsapp","email","wake","voice","runtime"].forEach(x=>{const o=document.createElement("option");o.value=o.textContent=x;sel.appendChild(o)})}}
async function loadMemory(){const r=await j("/activity/events?domain=memory&page=1");const ev=r.events||[];
document.getElementById("memory").innerHTML=ev.length?ev.map(e=>`<div class="row"><span class="muted">${esc(e.ts.slice(11,19))}</span> ${esc(e.summary)}</div>`).join(""):'<span class="muted">No memory activity.</span>';}
function refreshAll(){loadOverview();loadPending();loadTasks();loadSchedules();loadServices();loadEvents(page);loadMemory();
document.getElementById("clock").textContent=new Date().toLocaleTimeString()}
refreshAll();setInterval(refreshAll,5000);
</script></body></html>"""
