/* SalesGod product studio (agent "studio"): facts with sources and confirmation, decision tree with AI recommendations,
   simulated-customer tests with fix-with-AI, readiness and Publish. Everything edits a DRAFT (/api/studio/*); live calls change only on Publish.
   platform.js routes #/studio and #/studio/<tenant>.<product> here: window.SGStudio(SG, el, ctx). No external libraries. */
window.SGStudio = function (SG, el, ctx) {
'use strict';
const { h, sv, ic, api, modal, notice, head, S } = SG;
const toast = (m) => { const t = h('div', { class: 'toast', role: 'status' }, m); document.body.append(t); setTimeout(() => t.remove(), 2800); };
const tabs = (items, cur, on) => h('div', { class: 'ptabs', role: 'tablist' }, items.map(([id, label]) => h('button', { role: 'tab', 'aria-selected': id === cur ? 'true' : 'false', class: id === cur ? 'on' : '', onclick: () => on(id) }, label)));
const T3 = (v) => (v && typeof v === 'object' && !Array.isArray(v)) ? (v.en || v.hinglish || v.hi || '') : (v == null ? '' : String(v));
const ST = window.__studio = window.__studio || { tab: 'facts', sel: null, recs: null, run: null, filter: 'open' };
const css = `
.st-pill{display:inline-flex;gap:6px;align-items:center}
.st-facts{display:flex;flex-direction:column;gap:18px}
.st-row{display:grid;grid-template-columns:180px minmax(0,1fr) auto;gap:14px;align-items:start;padding:11px 16px;border-top:1px solid hsl(var(--rule))}
.st-row:first-of-type{border-top:0}
.st-row .lab{font-weight:500;color:hsl(var(--body))}
.st-row .val{white-space:pre-wrap;word-break:break-word}
.st-row .val .alt{display:block;color:hsl(var(--muted));font-size:12.5px;margin-top:2px}
.st-row .meta{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.st-row .acts{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.st-row.crit-open{background:hsl(var(--amber)/.05)}
.st-src{font-size:11.5px;color:hsl(var(--tertiary))}
.st-banner{display:flex;align-items:center;gap:12px;padding:12px 16px;border:1px solid hsl(var(--amber)/.35);background:hsl(var(--amber)/.07);border-radius:var(--r-lg);margin-bottom:16px}
.st-banner.ok{border-color:hsl(var(--green)/.35);background:hsl(var(--green)/.07)}
.st-tree{display:grid;grid-template-columns:minmax(0,1fr) 380px;min-height:560px;height:calc(100vh - 230px);border:1px solid hsl(var(--rule));border-radius:var(--r-lg);overflow:hidden}
.st-canvas{position:relative;overflow:auto;background-color:hsl(var(--bg));background-image:radial-gradient(hsl(var(--rule-strong)) 1px,transparent 1px);background-size:20px 20px}
.st-canvas .layer{position:relative}
.st-canvas svg{position:absolute;left:0;top:0;pointer-events:none}
.st-canvas .edge{fill:none;stroke:hsl(var(--rule-heavy));stroke-width:1.4}
.st-canvas .nd{position:absolute}
.st-canvas .nd.k-new{border-style:dashed;border-color:hsl(var(--violet))}
.st-dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.st-side{border-left:1px solid hsl(var(--rule));background:hsl(var(--panel));overflow-y:auto;display:flex;flex-direction:column}
.st-side .sh{padding:14px 16px;border-bottom:1px solid hsl(var(--rule))}
.st-side .sb{padding:12px 16px;display:flex;flex-direction:column;gap:10px}
.st-rec{border:1px solid hsl(var(--rule));border-radius:var(--r);padding:10px 12px;display:flex;flex-direction:column;gap:6px;background:hsl(var(--bg))}
.st-rec b{font-weight:500}
.st-rec .hint{margin:0}
.st-grid td,.st-grid th{text-align:center}
.st-grid td:first-child,.st-grid th:first-child{text-align:left}
.st-cell{cursor:pointer}
.st-tx{display:flex;flex-direction:column;gap:8px}
.st-tx .t{padding:8px 12px;border-radius:var(--r);border:1px solid hsl(var(--rule));max-width:88%}
.st-tx .t.agent{background:hsl(var(--panel))}
.st-tx .t.customer{align-self:flex-end;background:hsl(var(--elevated))}
.st-tx .t .m{font-size:11.5px;color:hsl(var(--tertiary));margin-top:3px}
.st-diff{display:grid;grid-template-columns:110px 1fr 1fr;gap:8px 12px;font-size:12.5px}
.st-diff .o{color:hsl(var(--red));text-decoration:line-through;white-space:pre-wrap}
.st-diff .n{color:hsl(var(--green));white-space:pre-wrap}
.st-chk{display:flex;gap:10px;align-items:flex-start;padding:10px 16px;border-top:1px solid hsl(var(--rule))}
.st-chk:first-child{border-top:0}
@media (max-width:760px){.st-row{grid-template-columns:1fr}.st-row .acts{justify-content:flex-start}.st-tree{grid-template-columns:1fr;height:auto}.st-canvas{height:420px}.st-diff{grid-template-columns:1fr}}`;
if (!document.getElementById('studio-css')) document.head.append(h('style', { id: 'studio-css' }, css));

const KIND_GROUP = [['Offer & price', ['price', 'offer', 'validity', 'terms', 'link', 'floor']], ['What it is', ['identity', 'what', 'close']], ['Why buy it', ['usp', 'proof', 'competitor']],
  ['Questions customers ask', ['faq']], ['Pronunciation', ['pronunciation']], ['Other', ['other', 'guarantee']]];
const SRC = { import: 'Live product sheet', ai: 'AI draft', user: 'You', user_text: 'Your pasted text', public: 'Public' };
const STATUS = { confirmed: ['green', 'Confirmed'], needs_confirmation: ['amber', 'Needs confirmation'], ai_draft: ['blue', 'AI draft'], rejected: ['red', 'Rejected'] };
const VERD = { pass: ['green', 'Pass'], fail: ['red', 'Fail'], not_reached: ['', 'Not reached'] };

const route = ctx.id ? ctx.id.split('.') : null;
if (!route) return listPage();
const [tenant, pid] = route;
const base = `/api/studio/${encodeURIComponent(tenant)}/${encodeURIComponent(pid)}`;
let V = null; // {draft, readiness, versions, published}

async function load() { V = await api(base); }
function ready() { return V.readiness; }
function nodesFlat(ns, out = []) { (ns || []).forEach((n) => { out.push(n); nodesFlat(n.children, out); }); return out; }
async function saveTree() { V = await api(base, { method: 'PUT', body: JSON.stringify({ tree: V.draft.tree }) }); ST.recs = null; }
async function poll(id, onStep) {
  for (;;) {
    const j = await api('/api/studio-jobs/' + id);
    if (onStep) onStep(j.step || '');
    if (j.status !== 'running') { if (j.status === 'failed') throw new Error(j.error || 'failed'); return j.result; }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

/* ------------------------------------------------------------------ list + new product */
async function listPage() {
  el.replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'empty' }, 'Loading…')));
  let list = [];
  try { list = (await api('/api/studio')).products; } catch (e) { el.replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'empty' }, h('b', {}, 'Studio is not available'), e.message))); return; }
  if (!ctx.alive()) return;
  const mine = list.filter((p) => p.tenant === S.tenant || !S.tenant);
  const nm = h('input', { placeholder: 'e.g. ZenBrew Coffee Subscription' }), txt = h('textarea', { rows: 4, placeholder: 'One line, or paste your brochure / website text. Facts the agent must not invent (price, offer, guarantees) stay marked until you confirm them.' });
  const lp = h('input', { inputmode: 'decimal', placeholder: 'e.g. 1200' }), op = h('input', { inputmode: 'decimal', placeholder: 'e.g. 999' }), unit = h('input', { placeholder: 'e.g. per month' });
  const err = h('div', { class: 'err' }), step = h('div', { class: 'hint' });
  const btn = h('button', { class: 'btn pri', onclick: async () => {
    err.textContent = ''; if (nm.value.trim().length < 2) { err.textContent = 'Give the product a name.'; return; }
    btn.disabled = true; step.textContent = 'Starting…';
    try {
      const j = await api('/api/studio/generate', { method: 'POST', body: JSON.stringify({ tenant: S.tenant || 'default', name: nm.value.trim(), text: txt.value, list_price: lp.value || null, offer_price: op.value || null, unit: unit.value }) });
      const r = await poll(j.id, (s) => { step.textContent = 'Drafting: ' + s; });
      toast('Draft ready. Review the facts, then publish.'); location.hash = `#/studio/${r.tenant}.${r.product_id}`;
    } catch (e) { err.textContent = e.message; btn.disabled = false; step.textContent = ''; }
  } }, ic('spark', 'sm'), 'Draft it with AI');
  el.replaceChildren(h('div', { class: 'pad' },
    head('Product studio', 'Everything the agent needs to sell a product: facts with sources, the decision tree, and tests with simulated customers. You edit a draft; calls change only when you publish.'),
    h('div', { class: 'cols2', style: 'align-items:start' },
      h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Products')),
        mine.length ? h('div', {}, mine.map((p) => h('a', { class: 'item2', href: `#/studio/${p.tenant}.${p.product_id}` },
          h('div', { style: 'flex:1;min-width:0' }, h('b', {}, p.name), h('div', { class: 'hint' }, p.has_draft ? (p.open_facts ? p.open_facts + ' facts need confirmation' : 'Ready to publish') : 'Not opened in the studio yet')),
          p.published ? h('span', { class: 'tag green' }, 'Live v' + p.published) : h('span', { class: 'tag' }, p.live ? 'Live' : 'Draft only'))))
          : h('div', { class: 'empty' }, h('b', {}, 'No products in this project'), 'Draft one with AI on the right.')),
      h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'New product')), h('div', { class: 'panel-body' },
        h('label', { class: 'lbl' }, 'Product name'), nm, h('label', { class: 'lbl' }, 'What is it? ', h('span', { class: 'opt' }, '· one line or pasted text')), txt,
        h('div', { class: 'grid3' }, h('div', {}, h('label', { class: 'lbl' }, 'Normal price ', h('span', { class: 'opt' }, '· optional')), lp), h('div', {}, h('label', { class: 'lbl' }, 'Offer price ', h('span', { class: 'opt' }, '· optional')), op), h('div', {}, h('label', { class: 'lbl' }, 'Price covers'), unit)),
        h('div', { class: 'row', style: 'margin-top:14px;gap:12px' }, btn, step), err)))));
}

/* ------------------------------------------------------------------ product page */
async function draw() {
  if (!V) { el.replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'empty' }, 'Loading the draft…'))); try { await load(); } catch (e) { el.replaceChildren(h('div', { class: 'pad' }, h('div', { class: 'empty' }, h('b', {}, 'Could not open this product'), e.message, h('div', {}, h('a', { class: 'btn', href: '#/studio' }, 'Back to the studio'))))); return; } }
  if (!ctx.alive()) return;
  const d = V.draft, r = ready();
  const open = (r.items.find((i) => i.id === 'facts') || {}).items || [];
  const body = h('div');
  el.replaceChildren(h('div', { class: 'pad' },
    head(d.name || pid, null,
      h('span', { class: 'st-pill' }, h('span', { class: 'tag blue' }, 'Draft'), V.published ? h('span', { class: 'tag green' }, 'Live v' + V.published) : h('span', { class: 'tag' }, 'Not live')),
      h('span', { class: 'tag ' + (r.ready ? 'green' : 'amber') }, 'Ready ', h('b', {}, r.score + '/' + r.total)),
      h('button', { class: 'btn', onclick: () => SG.newCall && SG.newCall({ product: pid }) }, ic('phone', 'sm'), 'Call me'),
      h('button', { class: 'btn pri', onclick: () => { ST.tab = 'publish'; draw(); } }, 'Publish')),
    tabs([['facts', 'Facts' + (open.length ? ` · ${open.length} to confirm` : '')], ['tree', 'Decision tree'], ['tests', 'Tests'], ['say', 'Pronunciation'], ['publish', 'Publish']], ST.tab, (t) => { ST.tab = t; draw(); }),
    body));
  ({ facts: drawFacts, tree: drawTree, tests: drawTests, say: drawSay, publish: drawPublish }[ST.tab] || drawFacts)(body);
}

/* ---- facts */
function factValue(f) {
  const v = f.value;
  if (v && typeof v === 'object' && !Array.isArray(v)) return h('div', { class: 'val' }, v.en || '—', v.hi ? h('span', { class: 'alt' }, v.hi) : null);
  return h('div', { class: 'val' + (v === '' || v == null ? ' hint' : '') }, v === '' || v == null ? 'Not set: the agent will say it does not have this detail' : String(v));
}
async function factAct(f, path, body) {
  try { const r = await api(`${base}/facts/${encodeURIComponent(f.id)}${path}`, { method: path ? 'POST' : 'PUT', body: JSON.stringify(body || {}) }); V.readiness = r.readiness; (r.facts || [r.fact]).forEach((nf) => { const i = V.draft.facts.findIndex((x) => x.id === nf.id); if (i >= 0) V.draft.facts[i] = nf; }); ST.recs = null; draw(); }
  catch (e) { notice(e.message); }
}
function editFact(f) {
  modal('Edit: ' + f.label, f.critical ? 'This fact is critical: after you change it, confirm it again.' : null, (b, foot, close) => {
    const obj = f.value && typeof f.value === 'object' && !Array.isArray(f.value);
    const F = obj ? ['en', 'hi', 'hinglish'].map((l) => [l, h('textarea', { rows: 2 }, f.value[l] || '')]) : [['v', h('textarea', { rows: 3 }, f.value == null ? '' : String(f.value))]];
    F.forEach(([l, e]) => b.append(h('label', { class: 'lbl' }, obj ? { en: 'English', hi: 'Hindi', hinglish: 'Hinglish' }[l] : 'Value'), e));
    const save = async (confirmIt) => { const val = obj ? Object.fromEntries(F.map(([l, e]) => [l, e.value])) : F[0][1].value; close(); await factAct(f, '', { value: val, ...(confirmIt ? { status: 'confirmed' } : {}) }); };
    foot.append(h('button', { class: 'btn pri', onclick: () => save(true) }, ic('check', 'sm'), 'Save and confirm'), h('button', { class: 'btn', onclick: () => save(false) }, 'Save'), h('span', { style: 'flex:1' }), h('button', { class: 'btn ghost', onclick: close }, 'Cancel'));
  });
}
function addFact(kind) {
  modal('Add a fact', 'Something the agent may say: a proof (reviews, years in business), a competitor comparison, a USP. You typed it, so it counts as confirmed.', (b, foot, close) => {
    const k = h('select', {}, [['proof', 'Proof'], ['competitor', 'Competitor comparison'], ['usp', 'USP'], ['faq', 'FAQ answer'], ['other', 'Other']].map(([v, n]) => h('option', { value: v, selected: v === (kind || 'proof') }, n)));
    const lab = h('input', { placeholder: 'e.g. Members' }), val = h('textarea', { rows: 3, placeholder: 'e.g. 4,000+ members across 6 branches since 2016' }), src = h('input', { placeholder: 'e.g. our member register, Nov 2026' }), err = h('div', { class: 'err' });
    b.append(h('label', { class: 'lbl' }, 'Kind'), k, h('label', { class: 'lbl' }, 'Label'), lab, h('label', { class: 'lbl' }, 'Fact'), val, h('label', { class: 'lbl' }, 'Source ', h('span', { class: 'opt' }, '· optional')), src, err);
    foot.append(h('button', { class: 'btn pri', onclick: async () => { if (!val.value.trim()) { err.textContent = 'Write the fact.'; return; } try { const r = await api(base + '/facts', { method: 'POST', body: JSON.stringify({ kind: k.value, label: lab.value || undefined, value: val.value.trim(), source: src.value }) }); V.draft.facts.push(r.fact); V.readiness = r.readiness; ST.recs = null; close(); draw(); } catch (e) { err.textContent = e.message; } } }, ic('plus', 'sm'), 'Add fact'),
      h('span', { style: 'flex:1' }), h('button', { class: 'btn ghost', onclick: close }, 'Cancel'));
  });
}
function drawFacts(body) {
  const facts = V.draft.facts || [], open = facts.filter((f) => f.critical && !['confirmed', 'rejected'].includes(f.status));
  const show = ST.filter === 'open' && open.length ? open : facts;
  body.append(open.length
    ? h('div', { class: 'st-banner' }, ic('alert', 'sm'), h('div', { style: 'flex:1' }, h('b', {}, `${open.length} critical facts need a person to confirm them`), h('div', { class: 'hint' }, 'Prices, offers, validity, terms and anything with a number. Publish stays locked until each one is confirmed, edited or rejected.')),
      h('div', { class: 'seg' }, [['open', 'To confirm'], ['all', 'All facts']].map(([v, n]) => h('button', { class: ST.filter === v ? 'on' : '', onclick: () => { ST.filter = v; draw(); } }, n))),
      h('button', { class: 'btn', onclick: () => modal(`Confirm ${open.length} facts?`, 'You confirm that each of these is true and the agent may say it on calls.', (b, foot, close) => {
        b.append(h('ul', { class: 'pts' }, open.map((f) => h('li', {}, h('b', {}, f.label + ': '), T3(f.value) || '(empty)'))));
        foot.append(h('button', { class: 'btn pri', onclick: async () => { close(); try { const r = await api(base + '/facts/bulk/confirm', { method: 'POST', body: JSON.stringify({ ids: open.map((f) => f.id) }) }); V.readiness = r.readiness; r.facts.forEach((nf) => { const i = V.draft.facts.findIndex((x) => x.id === nf.id); V.draft.facts[i] = nf; }); ST.filter = 'all'; draw(); } catch (e) { notice(e.message); } } }, ic('check', 'sm'), 'Confirm all'), h('span', { style: 'flex:1' }), h('button', { class: 'btn ghost', onclick: close }, 'Cancel'));
      }, { wide: true }) }, 'Confirm all…'))
    : h('div', { class: 'st-banner ok' }, ic('check', 'sm'), h('div', { style: 'flex:1' }, h('b', {}, 'Every critical fact is confirmed'), h('div', { class: 'hint' }, 'Add proofs and competitor lines so the agent has something to point to.')), h('button', { class: 'btn', onclick: () => addFact('proof') }, ic('plus', 'sm'), 'Add fact')));
  const groups = h('div', { class: 'st-facts' });
  for (const [g, kinds] of KIND_GROUP) {
    const fs = show.filter((f) => kinds.includes(f.kind) || (g === 'Other' && !KIND_GROUP.some(([, ks]) => ks.includes(f.kind))));
    if (!fs.length) continue;
    groups.append(h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, g), h('span', { class: 'tag' }, fs.length)),
      h('div', {}, fs.map((f) => {
        const [tone, label] = STATUS[f.status] || ['', f.status], src = f.source || {};
        const needs = f.status !== 'confirmed' && f.status !== 'rejected';
        return h('div', { class: 'st-row' + (f.critical && needs ? ' crit-open' : '') },
          h('div', {}, h('div', { class: 'lab' }, f.label), f.critical ? h('div', { class: 'hint' }, 'critical') : null),
          h('div', {}, factValue(f), h('div', { class: 'meta' }, h('span', { class: 'tag ' + tone }, label), h('span', { class: 'st-src' }, 'Source: ' + (SRC[src.type] || src.type || '—') + (src.ref && src.type !== 'import' ? ' · ' + src.ref : '') + (src.quote ? ' · “' + src.quote.slice(0, 60) + '”' : '')))),
          h('div', { class: 'acts' }, needs ? h('button', { class: 'btn sm pri', onclick: () => factAct(f, '/confirm') }, ic('check', 'sm'), 'Confirm') : null,
            h('button', { class: 'btn sm', onclick: () => editFact(f) }, 'Edit'),
            f.status !== 'rejected' ? h('button', { class: 'btn sm ghost', title: 'The agent will never say this', onclick: () => factAct(f, '', { status: 'rejected' }) }, 'Reject') : h('button', { class: 'btn sm ghost', onclick: () => factAct(f, '', { status: 'needs_confirmation' }) }, 'Restore')));
      }))));
  }
  body.append(groups, h('div', { class: 'row', style: 'margin-top:14px' }, h('button', { class: 'btn', onclick: () => addFact() }, ic('plus', 'sm'), 'Add fact')));
}

/* ---- pronunciation (agent "phonetics"): /api/pronunciations. Edits the LIVE lexicon (applied right before TTS on every call). */
const PL = [['hi', 'Hindi voice (Devanagari)'], ['hinglish', 'Hinglish (Roman)'], ['en', 'English voice']];
async function sayAudio(text, lang) {
  try {
    const res = await fetch('/api/pronunciations/preview', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, lang, product: pid }) });
    if (!res.ok) { let m = res.statusText; try { m = (await res.json()).detail || m; } catch {} return notice('Preview failed: ' + m); }
    const url = URL.createObjectURL(await res.blob()); const au = new Audio(url); au.onended = () => URL.revokeObjectURL(url); au.play();
  } catch (e) { notice('Preview failed: ' + e.message); }
}
async function drawSay(body) {
  body.append(h('div', { class: 'empty' }, 'Loading…'));
  let P; try { P = await api('/api/pronunciations?product=' + encodeURIComponent(pid)); } catch (e) { body.replaceChildren(h('div', { class: 'empty' }, e.message)); return; }
  if (!ctx.alive()) return;
  const lex = P.product.lexicon || {}, words = Object.keys(lex).sort();
  const rows = words.map((w) => ({ w, I: Object.fromEntries(PL.map(([l]) => [l, h('input', { value: (lex[w] || {})[l] || '', placeholder: l === 'en' ? 'e.g. Ouj-laa (or ph:AW1 JH L AA0)' : '' })])), score: h('span', { class: 'hint' }) }));
  const add = h('input', { placeholder: 'Add a word, e.g. Aujla' });
  const collect = () => Object.fromEntries(rows.map((r) => [r.w, Object.fromEntries(PL.map(([l]) => [l, r.I[l].value.trim()]).filter(([, v]) => v))]).filter(([, e]) => Object.keys(e).length));
  const save = async () => { try { await api('/api/pronunciations/product/' + encodeURIComponent(pid), { method: 'PUT', body: JSON.stringify({ lexicon: collect() }) }); toast('Pronunciations saved: live calls use them now.'); draw(); } catch (e) { notice(e.message); } };
  const check = async (r, btn) => {
    btn.disabled = true; r.score.textContent = 'checking…'; const out = [];
    try { for (const [l] of PL) { const v = r.I[l].value.trim(); const x = await api('/api/pronunciations/verify', { method: 'POST', body: JSON.stringify({ product: pid, word: r.w, lang: l, candidates: v ? [v] : [] }) }); const mine = (x.results || []).find((y) => y.value === v) || x.best || {}; const plain = (x.results || []).find((y) => y.value === '') || {}; out.push(`${l}: ${mine.score != null ? Math.round(mine.score * 100) + '%' : '—'}${v && plain.score != null ? ' (plain ' + Math.round(plain.score * 100) + '%)' : ''}`); } r.score.textContent = 'Sounds right? ' + out.join(' · '); }
    catch (e) { r.score.textContent = e.message; } btn.disabled = false;
  };
  const sugg = (P.suggestions || []);
  body.replaceChildren(
    h('div', { class: 'st-banner ok' }, ic('check', 'sm'), h('div', { style: 'flex:1' }, h('b', {}, 'How the agent says names, brands and short forms'), h('div', { class: 'hint' }, 'Applied to every line right before the voice speaks it (global → tenant → this product → agent). ₹ amounts and numbers are read out automatically. Changes here are live immediately (no Publish needed).'))),
    h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'This product'), h('span', { class: 'tag' }, words.length), h('span', { style: 'flex:1' }),
      h('button', { class: 'btn sm', onclick: async (ev) => { const b = ev.currentTarget; b.disabled = true; b.textContent = 'Finding words…'; try { await api('/api/pronunciations/suggest', { method: 'POST', body: JSON.stringify({ product: pid }) }); draw(); } catch (e) { notice(e.message); b.disabled = false; b.textContent = 'Suggest words'; } } }, 'Suggest words'),
      h('button', { class: 'btn sm pri', onclick: save }, ic('check', 'sm'), 'Save')),
      h('div', { class: 'tablewrap' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Word'), PL.map(([, n]) => h('th', {}, n)), h('th', {}, ''))),
        h('tbody', {}, rows.map((r) => h('tr', {}, h('td', {}, h('b', {}, r.w), h('div', {}, r.score)), PL.map(([l]) => h('td', {}, r.I[l])),
          h('td', {}, h('div', { class: 'row' }, h('button', { class: 'btn sm ghost', title: 'Hear it in a Hindi and an English line', onclick: () => { sayAudio(`मैं आपको ${r.w} के बारे में बताता हूँ।`, 'hi'); setTimeout(() => sayAudio(`Let me tell you about ${r.w} today.`, 'en'), 3500); } }, 'Play'),
            h('button', { class: 'btn sm ghost', title: 'Synthesise, transcribe with Whisper and score', onclick: (ev) => check(r, ev.currentTarget) }, 'Check'),
            h('button', { class: 'btn sm ghost', onclick: () => { delete lex[r.w]; rows.splice(rows.indexOf(r), 1); save(); } }, 'Remove')))))))),
      h('div', { class: 'row', style: 'padding:10px 16px' }, add, h('button', { class: 'btn sm', onclick: () => { const w = add.value.trim(); if (!w) return; rows.push({ w, I: Object.fromEntries(PL.map(([l]) => [l, h('input', { value: '' })])) }); lex[w] = {}; const c = collect(); c[w] = c[w] || { hi: w }; api('/api/pronunciations/product/' + encodeURIComponent(pid), { method: 'PUT', body: JSON.stringify({ lexicon: c }) }).then(draw, (e) => notice(e.message)); } }, ic('plus', 'sm'), 'Add'))),
    sugg.length ? h('div', { class: 'panel', style: 'margin-top:16px' }, h('div', { class: 'panel-head' }, h('b', {}, 'Suggestions (not applied until you accept)'), h('span', { class: 'tag amber' }, sugg.length), h('span', { style: 'flex:1' }),
      h('button', { class: 'btn sm', onclick: async () => { try { await api('/api/pronunciations/accept', { method: 'POST', body: JSON.stringify({ product: pid, words: sugg.map((x) => x.word) }) }); draw(); } catch (e) { notice(e.message); } } }, 'Accept all')),
      h('div', {}, sugg.map((x) => h('div', { class: 'st-row' }, h('div', {}, h('div', { class: 'lab' }, x.word), h('div', { class: 'hint' }, x.kind)),
        h('div', { class: 'val' }, PL.map(([l, n]) => (x.values || {})[l] ? h('div', {}, h('span', { class: 'hint' }, n + ': '), x.values[l], x.checks && x.checks[l] ? h('span', { class: 'hint' }, ` · heard “${x.checks[l].heard}” ${Math.round(x.checks[l].score * 100)}%`) : null) : null)),
        h('div', { class: 'acts' }, h('button', { class: 'btn sm pri', onclick: async () => { try { await api('/api/pronunciations/accept', { method: 'POST', body: JSON.stringify({ product: pid, words: [x.word] }) }); draw(); } catch (e) { notice(e.message); } } }, ic('check', 'sm'), 'Accept')))))) : null,
    h('div', { class: 'panel', style: 'margin-top:16px' }, h('div', { class: 'panel-head' }, h('b', {}, 'Try a line')), (() => {
      const t = h('textarea', { rows: 2, placeholder: 'e.g. Karan Aujla के VIP टिकट सिर्फ़ ₹4,999 में' }), l = h('select', {}, PL.map(([v, n]) => h('option', { value: v }, n)));
      return h('div', { class: 'row', style: 'padding:10px 16px;align-items:flex-start' }, t, l, h('button', { class: 'btn sm', onclick: () => t.value.trim() && sayAudio(t.value.trim(), l.value) }, 'Play as the agent'));
    })()));
}

/* ---- decision tree */
const NW = 212, NH = 58, GX = 260, GY = 74;
function layoutTree(nodes) {
  const L = {}; let row = 0;
  const place = (n, depth) => { const kids = n.children || []; if (!kids.length) { L[n.id] = { x: 30 + (depth + 1) * GX, y: 40 + row * GY }; row++; return L[n.id].y; }
    const ys = kids.map((c) => place(c, depth + 1)); L[n.id] = { x: 30 + (depth + 1) * GX, y: (ys[0] + ys[ys.length - 1]) / 2 }; return L[n.id].y; };
  nodes.forEach((n) => place(n, 0));
  L.__root = { x: 30, y: row ? (40 + (row - 1) * GY / 2) : 40 };
  return { L, w: 30 + (Math.max(1, ...Object.values(L).map((p) => (p.x - 30) / GX)) + 1) * GX, h: 40 + Math.max(1, row) * GY + 20 };
}
function nodeVerdict(id) { const g = ST.run && ST.run.grid && ST.run.grid[id]; if (!g) return null; const v = Object.values(g); return v.includes('fail') ? 'fail' : v.includes('pass') ? 'pass' : 'not_reached'; }
async function drawTree(body) {
  const nodes = V.draft.tree.nodes || [];
  if (!ST.recs) { try { ST.recs = (await api(base + '/recommendations')).recommendations; } catch (e) { ST.recs = []; } }
  if (!ST.run && V.draft.last_test) { try { ST.run = (await api(base + '/tests')).run; } catch (e) { /* no run */ } }
  const { L, w, h: ht } = layoutTree(nodes);
  const svg = sv('svg', { width: w, height: ht });
  const mid = (p, r) => [r ? p.x + NW : p.x, p.y + NH / 2];
  const edge = (a, b) => { const [x1, y1] = mid(a, 1), [x2, y2] = mid(b, 0), dx = Math.max(30, (x2 - x1) / 2); svg.append(sv('path', { class: 'edge', d: `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}` })); };
  const layer = h('div', { class: 'layer', style: `width:${w}px;height:${ht}px` }, svg);
  const walk = (ns, parent) => ns.forEach((n) => { edge(parent, L[n.id]); walk(n.children || [], L[n.id]); });
  walk(nodes, L.__root);
  layer.append(h('div', { class: 'nd k-root', style: `left:${L.__root.x}px;top:${L.__root.y}px;width:${NW - 40}px` }, h('div', { class: 'k' }, 'Call'), h('div', { class: 't' }, 'Opening'), h('div', { class: 'm' }, 'gates run first')));
  nodesFlat(nodes).forEach((n) => {
    const v = nodeVerdict(n.id), p = L[n.id];
    layer.append(h('div', { class: 'nd k-node' + (ST.sel === n.id ? ' sel' : '') + (n.origin ? ' k-new' : ''), style: `left:${p.x}px;top:${p.y}px`, title: n.title, onclick: () => { ST.sel = n.id; draw(); } },
      h('div', { class: 'k' }, n.origin ? 'AI added' : 'Situation', v ? h('span', { class: 'st-dot', style: `margin-left:auto;background:hsl(var(--${v === 'pass' ? 'green' : v === 'fail' ? 'red' : 'slate'}))`, title: 'last test: ' + v }) : null),
      h('div', { class: 't' }, n.title), h('div', { class: 'm' }, `${(n.cites || []).length} facts · ${(n.points || []).length} points${n.points_hi && n.points_hi.length ? '' : ' · no Hindi'}`)));
  });
  const canvas = h('div', { class: 'st-canvas', onclick: (e) => { if (e.target === canvas || e.target === layer) { ST.sel = null; draw(); } } }, layer);
  const side = h('aside', { class: 'st-side' });
  body.append(h('div', { class: 'st-tree' }, canvas, side));
  const sel = ST.sel && nodesFlat(nodes).find((n) => n.id === ST.sel);
  sel ? inspector(side, sel) : recsRail(side);
}
function recsRail(side) {
  const recs = ST.recs || [];
  const add = recs.filter((r) => r.action === 'add'), warn = recs.filter((r) => r.action !== 'add');
  const card = (r) => h('div', { class: 'st-rec' }, h('div', { class: 'row', style: 'gap:8px;align-items:flex-start' }, h('span', { class: 'tag ' + (r.severity === 'warn' ? 'amber' : 'blue') }, r.severity === 'warn' ? 'Fix' : 'Idea'), h('b', { style: 'flex:1' }, r.title)), h('p', { class: 'hint' }, r.detail),
    h('div', { class: 'row', style: 'gap:6px' },
      r.action === 'add' || r.action === 'regenerate' ? h('button', { class: 'btn sm pri', onclick: (e) => addRec(r, e.currentTarget) }, ic('spark', 'sm'), r.action === 'add' ? 'Add with AI' : 'Rewrite with AI') : null,
      r.action === 'add_fact' ? h('button', { class: 'btn sm', onclick: () => addFact(r.kind) }, ic('plus', 'sm'), 'Add fact') : null,
      r.node ? h('button', { class: 'btn sm', onclick: () => { ST.sel = r.node; draw(); } }, 'Open') : null));
  side.append(h('div', { class: 'sh' }, h('b', {}, 'Recommendations'), h('div', { class: 'hint' }, `${recs.length} things to improve. Click a situation on the left to edit it.`)),
    h('div', { class: 'sb' }, h('button', { class: 'btn sm', onclick: () => describeCase() }, ic('spark', 'sm'), 'Describe a new situation'),
      add.length ? h('div', { class: 'sec-lbl' }, 'Missing situations') : null, ...add.map(card), warn.length ? h('div', { class: 'sec-lbl' }, 'Checks') : null, ...warn.map(card)));
}
async function addRec(r, btn, desc) {
  if (btn) { btn.disabled = true; btn.lastChild.textContent = 'Writing…'; }
  try { const out = await api(base + '/recommendations/add', { method: 'POST', body: JSON.stringify(r ? { id: r.id } : { description: desc }) }); await load(); ST.recs = out.recommendations; ST.sel = out.node.id; toast('Added: ' + out.node.title + '. Review it, it is in the draft.'); draw(); }
  catch (e) { notice(e.message); if (btn) { btn.disabled = false; btn.lastChild.textContent = 'Try again'; } }
}
function describeCase() {
  modal('Describe a situation', 'What might a customer say? AI writes the cue words, talking points from your facts, and the question to end on, in English and Hindi.', (b, foot, close) => {
    const t = h('textarea', { rows: 4, placeholder: 'e.g. The customer says they joined a gym before and never went, so it was a waste of money.' });
    b.append(t);
    foot.append(h('button', { class: 'btn pri', onclick: (e) => { if (!t.value.trim()) return; close(); addRec(null, null, t.value.trim()); } }, ic('spark', 'sm'), 'Write it'), h('span', { style: 'flex:1' }), h('button', { class: 'btn ghost', onclick: close }, 'Cancel'));
  });
}
function inspector(side, n) {
  n.when = n.when || {};
  const facts = (V.draft.facts || []).filter((f) => f.status !== 'rejected' && !['identity', 'pronunciation'].includes(f.kind));
  const lines = (a) => (a || []).join('\n'), split = (v) => v.split('\n').map((x) => x.trim()).filter(Boolean);
  const title = h('input', { value: n.title || '' }), goal = h('input', { value: n.goal || '' }), kw = h('textarea', { rows: 2 }, (n.when.keywords || []).join(', '));
  const pts = h('textarea', { rows: 5 }, lines(n.points)), ask = h('input', { value: n.ask || '' }), ptsHi = h('textarea', { rows: 4 }, lines(n.points_hi)), askHi = h('input', { value: n.ask_hi || '' });
  const cites = new Set(n.cites || []);
  const chips = h('div', { class: 'chips' }, facts.map((f) => h('label', { class: 'chip', title: T3(f.value) }, h('input', { type: 'checkbox', checked: cites.has(f.id), onchange: (e) => { e.target.checked ? cites.add(f.id) : cites.delete(f.id); } }), f.label)));
  const save = async () => {
    Object.assign(n, { title: title.value, goal: goal.value, points: split(pts.value), ask: ask.value, points_hi: split(ptsHi.value), ask_hi: askHi.value, cites: [...cites] });
    n.when.keywords = kw.value.split(',').map((x) => x.trim()).filter(Boolean); delete n.origin;
    try { await saveTree(); toast('Saved to the draft.'); draw(); } catch (e) { notice(e.message); }
  };
  const verdicts = ST.run && ST.run.grid && ST.run.grid[n.id];
  side.append(h('div', { class: 'sh' }, h('div', { class: 'row' }, h('span', { class: 'sec-lbl' }, 'Situation · ' + n.id), h('span', { style: 'flex:1' }), h('button', { class: 'btn sm ghost', onclick: () => { ST.sel = null; draw(); } }, 'Back to recommendations')),
    verdicts ? h('div', { class: 'row', style: 'gap:6px;margin-top:8px;flex-wrap:wrap' }, Object.entries(verdicts).map(([p, v]) => h('span', { class: 'tag ' + VERD[v][0] }, p + ': ' + VERD[v][1]))) : null),
    h('div', { class: 'sb' },
      h('label', { class: 'lbl' }, 'When the customer…'), title, h('label', { class: 'lbl' }, 'Goal of the reply'), goal,
      h('label', { class: 'lbl' }, 'Cue words · comma separated'), kw,
      h('label', { class: 'lbl' }, 'Talking points · one per line'), pts, h('label', { class: 'lbl' }, 'Question to end on'), ask,
      h('label', { class: 'lbl' }, 'Hindi talking points'), ptsHi, h('label', { class: 'lbl' }, 'Hindi question'), askHi,
      h('label', { class: 'lbl' }, 'Facts this situation may cite'), chips,
      h('div', { class: 'row', style: 'gap:6px;flex-wrap:wrap;margin-top:6px' }, h('button', { class: 'btn pri sm', onclick: save }, ic('save', 'sm'), 'Save'),
        h('button', { class: 'btn sm', onclick: () => fixNode(n.id, null, true) }, ic('spark', 'sm'), 'Improve with AI'), h('span', { style: 'flex:1' }),
        h('button', { class: 'btn sm danger', onclick: async () => { const rm = (ns) => ns.filter((x) => x.id !== n.id).map((x) => ({ ...x, children: rm(x.children || []) })); V.draft.tree.nodes = rm(V.draft.tree.nodes); ST.sel = null; try { await saveTree(); draw(); } catch (e) { notice(e.message); } } }, ic('trash', 'sm'), 'Delete'))));
}
function fixNode(nid, personaId, ask) {
  modal('Improve with AI', 'AI rewrites this situation from your confirmed facts' + (personaId ? ' and the failed test call.' : '.') + ' Nothing changes until you accept.', (b, foot, close) => {
    const instr = h('input', { placeholder: 'Optional: what should change? e.g. answer the trust question before talking price' }), out = h('div'), err = h('div', { class: 'err' });
    let prop = null;
    const accept = h('button', { class: 'btn pri', style: 'display:none', onclick: async () => { try { await api(`${base}/nodes/${encodeURIComponent(nid)}/apply`, { method: 'POST', body: JSON.stringify({ node: prop.after }) }); close(); await load(); ST.recs = null; toast('Applied to the draft. Run the tests again to check.'); draw(); } catch (e) { err.textContent = e.message; } } }, ic('check', 'sm'), 'Accept');
    const run = async (btn) => {
      btn.disabled = true; btn.lastChild.textContent = 'Writing…'; err.textContent = ''; out.replaceChildren();
      try {
        prop = await api(base + '/fix', { method: 'POST', body: JSON.stringify({ node_id: nid, persona_id: personaId || undefined, instruction: instr.value || undefined }) });
        const ch = Object.entries(prop.changes || {});
        const show = (v) => Array.isArray(v) ? v.join('\n') : (v && typeof v === 'object') ? JSON.stringify(v) : (v || '—');
        out.append(ch.length ? h('div', { class: 'st-diff', style: 'margin-top:14px' }, ...ch.flatMap(([k, c]) => [h('b', {}, k), h('div', { class: 'o' }, show(c.old)), h('div', { class: 'n' }, show(c.new))])) : h('p', { class: 'hint' }, 'No changes suggested.'));
        accept.style.display = ch.length ? '' : 'none';
      } catch (e) { err.textContent = e.message; }
      btn.disabled = false; btn.lastChild.textContent = 'Write again';
    };
    b.append(instr, err, out);
    const go = h('button', { class: 'btn', onclick: (e) => run(e.currentTarget) }, ic('spark', 'sm'), h('span', {}, 'Suggest'));
    foot.append(accept, go, h('span', { style: 'flex:1' }), h('button', { class: 'btn ghost', onclick: close }, 'Cancel'));
    if (!ask) run(go);
  }, { wide: true });
}

/* ---- tests */
async function drawTests(body) {
  if (!ST.run && V.draft.last_test) { try { ST.run = (await api(base + '/tests')).run; } catch (e) { /* none */ } }
  const run = ST.run, step = h('span', { class: 'hint' });
  const btn = h('button', { class: 'btn pri', onclick: async () => {
    btn.disabled = true; step.textContent = 'Starting…';
    try { const j = await api(base + '/tests', { method: 'POST', body: JSON.stringify({ personas: 4, max_turns: 6 }) }); ST.run = await poll(j.id, (s) => { step.textContent = s; }); await load(); toast('Tests finished.'); draw(); }
    catch (e) { notice(e.message); btn.disabled = false; step.textContent = ''; }
  } }, ic('play', 'sm'), run ? 'Run again' : 'Run tests');
  body.append(h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Simulated customers'), run ? h('span', { class: 'tag ' + (run.pass_rate >= 0.75 ? 'green' : 'amber') }, Math.round(run.pass_rate * 100) + '% of calls passed') : null),
    h('div', { class: 'panel-body' }, h('p', { class: 'hint', style: 'margin-top:0' }, 'AI customers (an easy buyer, a haggler, a sceptic, a busy person) call the draft through the real pipeline. Code checks every number against your confirmed facts and that the link is only sent after a clear yes; an AI judge grades each situation. Results warn, they never block publishing.'),
      h('div', { class: 'row', style: 'gap:12px' }, btn, step))));
  if (!run) return;
  const ps = run.personas || [], ids = Object.keys(run.grid || {});
  const resOf = (pid2) => (run.results || []).find((r) => r.persona.id === pid2);
  body.append(h('div', { class: 'panel', style: 'margin-top:16px' }, h('div', { class: 'panel-head' }, h('b', {}, 'Situations × customers'), h('span', { class: 'hint' }, 'click a cell for the call')),
    h('div', { class: 'tablewrap' }, h('table', { class: 'tbl st-grid' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Situation'), ps.map((p) => h('th', {}, p.name || p.id, h('div', { class: 'hint' }, p.language))), h('th', {}, ''))),
      h('tbody', {}, ids.map((nid) => { const n = nodesFlat(V.draft.tree.nodes).find((x) => x.id === nid); const failed = ps.find((p) => run.grid[nid][p.id] === 'fail');
        return h('tr', {}, h('td', {}, h('a', { class: 'lnk', href: 'javascript:void 0', onclick: () => { ST.sel = nid; ST.tab = 'tree'; draw(); } }, n ? n.title : nid)),
          ps.map((p) => { const v = run.grid[nid][p.id]; return h('td', { class: 'st-cell', onclick: () => v && showCall(resOf(p.id), nid) }, v ? h('span', { class: 'tag ' + VERD[v][0] }, VERD[v][1]) : h('span', { class: 'hint' }, '—')); }),
          h('td', {}, failed ? h('button', { class: 'btn sm', onclick: () => fixNode(nid, failed.id) }, ic('spark', 'sm'), 'Fix with AI') : null)); }))))),
    h('div', { class: 'cards', style: 'margin-top:16px' }, (run.results || []).map((r) => h('div', { class: 'pcard', style: 'cursor:pointer', onclick: () => showCall(r) },
      h('div', { class: 'row' }, h('b', { style: 'flex:1' }, r.persona.name || r.persona.id), h('span', { class: 'tag ' + (r.passed ? 'green' : 'red') }, r.passed ? 'Passed' : 'Failed')),
      h('p', { class: 'hint' }, r.persona.brief || ''), r.judge && r.judge.reason ? h('p', {}, r.judge.reason) : null,
      (r.general_issues || []).length ? h('p', { class: 'err', style: 'display:block' }, r.general_issues[0]) : null))));
}
function showCall(r, nid) {
  if (!r) return;
  modal((r.persona.name || r.persona.id) + ' · test call', r.persona.brief || null, (b, foot, close) => {
    const nv = nid && r.nodes[nid];
    if (nv) b.append(h('div', { class: 'st-banner' + (nv.verdict === 'pass' ? ' ok' : '') }, h('div', {}, h('b', {}, (nodesFlat(V.draft.tree.nodes).find((x) => x.id === nid) || {}).title || nid, ': ', VERD[nv.verdict][1]), (nv.reasons || []).map((x) => h('div', { class: 'hint' }, x)))));
    b.append(h('div', { class: 'st-tx' }, r.turns.map((t) => h('div', { class: 't ' + t.role }, t.text, h('div', { class: 'm' }, t.role === 'agent' ? [t.node ? 'situation: ' + t.node : t.intent || '', ...(t.issues || []).map((x) => ' · ⚠ ' + x)].join('') : r.persona.name || 'customer')))));
    foot.append(nv && nv.verdict === 'fail' ? h('button', { class: 'btn pri', onclick: () => { close(); fixNode(nid, r.persona.id); } }, ic('spark', 'sm'), 'Fix with AI') : null, h('span', { style: 'flex:1' }), h('button', { class: 'btn ghost', onclick: close }, 'Close'));
  }, { drawer: true });
}

/* ---- publish */
function drawPublish(body) {
  const r = ready(), block = r.items.filter((i) => i.blocking && !i.ok);
  const pubBtn = h('button', { class: 'btn pri lg', disabled: !r.ready, onclick: async () => {
    pubBtn.disabled = true;
    try { const out = await api(base + '/publish', { method: 'POST' }); await load(); toast(`Published v${out.version}. The next call uses it.`); draw(); }
    catch (e) { notice(e.message); pubBtn.disabled = false; }
  } }, 'Publish v' + (((V.versions || []).slice(-1)[0] || {}).version + 1 || 1));
  body.append(h('div', { class: 'cols2', style: 'align-items:start' },
    h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Readiness'), h('span', { class: 'tag ' + (r.ready ? 'green' : 'amber') }, r.score + ' of ' + r.total)),
      h('div', {}, r.items.map((i) => h('div', { class: 'st-chk' }, h('span', { class: 'st-dot', style: `margin-top:6px;background:hsl(var(--${i.ok ? 'green' : i.blocking ? 'red' : 'amber'}))` }),
        h('div', { style: 'flex:1' }, h('b', { style: 'font-weight:500' }, i.label), h('div', { class: 'hint' }, i.detail + (i.blocking && !i.ok ? ' · blocks publishing' : ''))),
        !i.ok && i.id === 'facts' ? h('button', { class: 'btn sm', onclick: () => { ST.tab = 'facts'; ST.filter = 'open'; draw(); } }, 'Review') : null,
        !i.ok && ['cites', 'hindi', 'grounded', 'asks'].includes(i.id) ? h('button', { class: 'btn sm', onclick: () => { ST.tab = 'tree'; ST.sel = (i.items || [])[0] || null; draw(); } }, 'Open') : null,
        !i.ok && i.id === 'tests' ? h('button', { class: 'btn sm', onclick: () => { ST.tab = 'tests'; draw(); } }, 'Run') : null)))),
    h('div', {}, h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('b', {}, 'Go live')), h('div', { class: 'panel-body' },
      h('p', { class: 'hint', style: 'margin-top:0' }, block.length ? 'Confirm the critical facts first. Everything else is a warning you can publish with.' : 'Publishing saves this draft as a new version and the next call uses it. Edits made in the classic editor since the last publish are kept.'),
      pubBtn)),
      h('div', { class: 'panel', style: 'margin-top:16px' }, h('div', { class: 'panel-head' }, h('b', {}, 'Versions')),
        h('div', {}, (V.versions || []).slice().reverse().map((v) => h('div', { class: 'item2' }, h('div', { style: 'flex:1' }, h('b', { class: 'mono' }, 'v' + v.version), ' ', h('span', { class: 'hint' }, v.note || v.kind)), v.published ? h('span', { class: 'tag green' }, 'Live') : h('span', { class: 'tag' }, v.status))))))));
}

draw();
};
