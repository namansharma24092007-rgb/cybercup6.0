"""UPI Rakshak - real-time UPI fraud detection gateway.

Pay ANY UPI ID. Every payment is analysed BEFORE it is handed to your UPI app:

  1. check    : validate the UPI ID, enrich it (PSP name lookup if configured, network reputation, blocklist),
                score it with an ML model + explainable rules + optional Grok (LLM) reasoning  -> ALLOW / WARN / BLOCK
  2. proceed  : ALLOW / WARN (acknowledged) / analyst-approved payments get a standard NPCI UPI deep link + QR
                (upi://pay?pa=...&am=...) that opens GPay / PhonePe / Paytm / BHIM / any bank app.
                The UPI PIN is entered ONLY inside that app. Rakshak never sees or stores a PIN.
  3. outcome  : the user reports Paid / Failed (optionally with the 12-digit UTR); everything is stored in SQLite.

Why a deep link and not "we debit your account"?  Moving money over UPI requires being a regulated PSP / TPAP
(NPCI + bank partner). Rakshak is the risk layer in front of the user's own regulated UPI app, so real money moves
to real UPI IDs without Rakshak holding funds. See the README notes in this docstring's bottom section.

Single file: DB layer, fraud engine and HTTP API.  Frontend: index.html.

Run:   pip install -r requirements.txt   then   python app.py   ->  http://localhost:5000
Production: gunicorn -w 1 --threads 8 app:app   behind HTTPS (set HTTPS=1 so the session cookie is Secure).

Optional settings (environment variables or a .env file next to app.py):
  XAI_API_KEY, GROK_MODEL        Grok second opinion + discovery of brand-new scam patterns
  RAZORPAY_KEY_ID/_SECRET        real UPI-ID existence + account-holder-name check (Razorpay VPA validation API)
  ANALYST_PHONE, ANALYST_PASSWORD  analyst login (otherwise 9000000001 + a random password printed on first start)
  SECRET_KEY, DB_PATH, PORT, HOST, HTTPS, DAILY_LIMIT_INR (default 100000), EXTRA_HANDLES=comma,separated
"""
import os, re, json, secrets, random, time, sqlite3, difflib, base64, urllib.request, urllib.parse, urllib.error
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from flask import Flask, request, jsonify, session, send_from_directory, g
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, ".env")):  # optional .env file next to app.py
    for line in open(os.path.join(HERE, ".env")):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v.strip("\"'"))

XAI_KEY = os.getenv("XAI_API_KEY"); GROK_MODEL = os.getenv("GROK_MODEL", "grok-3")          # optional: LLM second opinion
RZP_ID, RZP_SECRET = os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")          # optional: real UPI ID + name check
PER_TXN_LIMIT = 100000 * 100                                                                  # paise (UPI P2P limit)
DEFAULT_DAILY = int(os.getenv("DAILY_LIMIT_INR", "100000")) * 100
CHECK_TTL_MIN = 15                                                                            # a verdict expires after this


# ============================================================================
# 1. DATABASE  (SQLite, WAL, explicit transactions)
# ============================================================================
DB_PATH = os.getenv("DB_PATH", os.path.join(HERE, "upi_rakshak.db"))


def db_conn():
    c = sqlite3.connect(DB_PATH, timeout=15, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("pragma journal_mode=wal"); c.execute("pragma foreign_keys=on")
    return c


@contextmanager
def db_tx():
    """BEGIN IMMEDIATE = write lock up-front; any exception rolls the whole unit of work back."""
    c = db_conn()
    try:
        c.execute("begin immediate"); yield c; c.execute("commit")
    except BaseException:
        if c.in_transaction: c.execute("rollback")
        raise
    finally:
        c.close()


SCHEMA = """
create table if not exists users(
  id integer primary key autoincrement, name text not null, phone text unique not null, email text,
  pw_hash text not null, role text not null default 'user', daily_limit integer not null, created text not null);

create table if not exists payments(
  id integer primary key autoincrement, ref text unique, user_id integer not null,
  payee_vpa text not null, payee_name text, display_name text, amount integer not null,   -- paise
  mode text, purpose text, note text,
  status text not null,   -- CHECKED | UNDER_REVIEW | APPROVED | INITIATED | PAID | FAILED | CANCELLED | REJECTED
  risk real, model_risk real, decision text, category text,
  features text, reasons text, contrib text, ai text,
  analyst_label text, user_ack integer default 0, utr text, vpa_verified integer default 0,
  created text, updated text);
create index if not exists ix_pay_user on payments(user_id, status);
create index if not exists ix_pay_payee on payments(payee_vpa, status);

create table if not exists reports(
  id integer primary key autoincrement, reporter_id integer, vpa text, payment_id integer, reason text, ts text,
  unique(reporter_id, vpa));
create table if not exists blocklist(vpa text primary key, reason text, added_by integer, ts text);
create table if not exists vpa_cache(vpa text primary key, valid integer, name text, ts text);
create table if not exists patterns(
  id integer primary key autoincrement, name text unique, description text, keywords text, status text,
  hits integer default 0, created text);
create table if not exists model_meta(id integer primary key autoincrement, ts text, metrics text);
create table if not exists audit(id integer primary key autoincrement, ts text, user_id integer, event text, detail text);
"""


def db_init():
    c = db_conn(); c.executescript(SCHEMA); c.close()


# ============================================================================
# 2. FRAUD ENGINE  (pure functions: UPI-ID analysis, ML model, rules, Grok, learned patterns)
# ============================================================================
VPA_RE = re.compile(r"[a-z0-9][a-z0-9.\-_]{1,255}@[a-z][a-z0-9]{1,63}")
PURPOSES = {"shop": "Shop / merchant", "person": "Friend or family", "bill": "Bill or service", "job_invest": "Job, task or investment",
            "receive_money": "To RECEIVE money (refund, prize, loan)", "other": "Something else"}

# Common live UPI handles (PSP / bank suffixes). Not exhaustive: unknown handle = a flag, never a verdict. Extend with EXTRA_HANDLES=a,b,c
HANDLES = set("""okaxis okhdfcbank okicici oksbi ybl ibl axl paytm pthdfc ptsbi ptyes ptaxis apl yapl rapl upi waicici waaxis wahdfcbank
wasbi axisb freecharge ikwik mbk sbi hdfcbank icici axisbank kotak pnb boi cnrb unionbank uboi barodampay idfcfirst idfcbank indus
yesbank federal fbl rbl aubank bandhan dbs hsbc citi airtel ippb postbank jio tapicici abfspay pz equitas jkb idbi iob mahb sib psb
cbin allbank kbl kvb karb lvb dlb tjsb jsb ucobank nsdl okbizaxis fifederal jupiter slice pingpay ezetap timecosmos kmbl hdfcbankjd
""".split()) | {h.strip() for h in os.getenv("EXTRA_HANDLES", "").split(",") if h.strip()}

FEATS = ["amount_log", "night", "amount_ratio", "new_payee", "net_age", "complaints", "fan_in", "collect", "qr", "link", "refund_word",
         "kyc_word", "urgency_word", "invest_word", "name_mismatch", "screenshare", "remote_app", "handle_unknown", "handle_lookalike",
         "vpa_risky_words", "vpa_random", "receive_purpose"]
FL = {"amount_log": "Large amount", "night": "Unusual hour (night)", "amount_ratio": "Amount far above your usual",
      "new_payee": "First payment to this UPI ID", "net_age": "UPI ID recently seen on the network",
      "complaints": "Fraud reports on this UPI ID", "fan_in": "Receives from many different senders (mule pattern)",
      "collect": "Request that takes money OUT", "qr": "Paying via QR", "link": "Paying via link",
      "refund_word": "'Refund / prize / cashback' wording", "kyc_word": "KYC / account-block wording",
      "urgency_word": "Urgency pressure", "invest_word": "Job / task / investment wording",
      "name_mismatch": "Name shown differs from account holder", "screenshare": "Someone is guiding you on a call",
      "remote_app": "Remote-access app running", "handle_unknown": "Unrecognised UPI handle after the @",
      "handle_lookalike": "Handle looks like a real bank/app handle", "vpa_risky_words": "Official / support / refund words in UPI ID",
      "vpa_random": "UPI ID looks auto-generated", "receive_purpose": "Paying in order to 'receive' money"}
CATS = {"collect_refund": "Refund / collect-request scam", "qr_swap": "QR / name mismatch", "remote_access": "Remote-access or coached payment",
        "fake_kyc": "Fake KYC / bank alert", "mule_account": "Mule-account pattern", "fake_job_investment": "Fake job / task / investment",
        "impersonation": "Impersonation / look-alike UPI ID", "blocklisted": "Reported / blocklisted UPI ID",
        "suspicious_other": "Other suspicious", "none": "Legitimate"}
M = {}

REF = re.compile(r"refund|cashback|reversal|money back|reward|prize|lottery|\bwon\b|claim|scratch card|gift ?(card|voucher)", re.I)
KYC = re.compile(r"kyc|account.{0,20}(block|suspend|freez|deactiv)|update.{0,15}(pan|aadhaar|aadhar)|verification fee|sim.{0,10}block|card.{0,12}(block|expire)", re.I)
URG = re.compile(r"urgent|immediately|right now|today|tonight|last chance|expire|within \d+|final notice|disconnect|power cut|cut off|legal action|arrest", re.I)
INV = re.compile(r"invest|trading|crypto|bitcoin|forex|stock tip|guaranteed|profit|returns?\b|part.?time|work from home|\bwfh\b|\btask\b|commission|per day|telegram|registration fee|processing fee|withdraw", re.I)
RISKY_VPA = re.compile(r"refund|cashback|kyc|support|help.?desk|helpline|care|customer|lottery|reward|prize|loan|task|earn|invest|trad(e|ing)|crypto|"
                       r"official|verif|update|block|police|cbi|rbi|npci|income.?tax|electricity|bsnl|airtel|jio|amazon|flipkart|paytm|phonepe|gpay|"
                       r"\bsbi|hdfc|icici|axis|gov", re.I)


def vpa_traits(vpa):
    """Everything that can be said about a UPI ID from its text alone."""
    user, _, handle = vpa.partition("@")
    unknown = handle not in HANDLES
    look = unknown and bool(difflib.get_close_matches(handle, list(HANDLES), n=1, cutoff=.78))
    rnd = bool(re.search(r"\d{5,}", user)) and not re.fullmatch(r"(91)?\d{10}", user)
    return dict(user=user, handle=handle, handle_known=not unknown, handle_unknown=float(unknown), handle_lookalike=float(look),
                vpa_risky_words=float(bool(RISKY_VPA.search(user))), vpa_random=float(rnd), phone_based=bool(re.fullmatch(r"(91)?\d{10}", user)))


# ---------------------------------------------------------------- training data (typologies of Indian UPI scams)
def synth(n, seed):
    r = np.random.default_rng(seed); X, y, C = [], [], []; U = r.uniform
    P = lambda q: float(r.random() < q)
    classes = ["legit", "collect_refund", "qr_swap", "remote_access", "fake_kyc", "mule_account", "fake_job_investment", "impersonation"]
    for c in r.choice(classes, n, p=[.72, .06, .05, .04, .04, .03, .04, .02]):
        amt = float(np.clip(r.lognormal(6.3, 1.1), 10, 100000)); mode = r.choice(["pay", "qr", "link", "collect"], p=[.5, .35, .07, .08])
        new = P(.25); seen = P(.3)
        f = dict(night=P(.06), new_payee=new, net_age=U(1, 400) if seen else 0., complaints=float(r.poisson(.02)),
                 fan_in=float(r.integers(1, 60)) if seen else 0., refund_word=P(.005), kyc_word=P(.005), urgency_word=P(.03), invest_word=P(.01),
                 name_mismatch=(float(np.clip(abs(r.normal(0, .06)), 0, 1)) if P(.9) else U(.1, .4)) if P(.45) else 0.,
                 screenshare=P(.005), remote_app=P(.02), handle_unknown=P(.03), handle_lookalike=P(.002), vpa_risky_words=P(.03),
                 vpa_random=P(.04), receive_purpose=P(.002))
        ratio = float(np.clip(r.lognormal(0, .6), .05, 30))
        if c == "collect_refund":
            mode = "collect"; amt = U(300, 20000); ratio = U(.5, 6)
            f.update(refund_word=P(.9), new_payee=P(.9), urgency_word=P(.4), complaints=float(r.poisson(2)), vpa_risky_words=P(.5), receive_purpose=P(.6), net_age=U(1, 30) if P(.5) else 0.)
        elif c == "qr_swap":
            mode = "qr"; f.update(name_mismatch=U(.45, .95) if P(.6) else 0., new_payee=P(.8), complaints=float(r.poisson(1)), vpa_random=P(.4), handle_lookalike=P(.1))
        elif c == "remote_access":
            amt = U(5000, 100000); ratio = U(2, 25); f.update(screenshare=P(.85), remote_app=P(.9), new_payee=P(.8), urgency_word=P(.5))
        elif c == "fake_kyc":
            mode = r.choice(["link", "collect", "pay"]); amt = U(1, 10) if P(.5) else U(500, 15000)
            f.update(kyc_word=P(.9), urgency_word=P(.8), new_payee=P(.9), vpa_risky_words=P(.6), handle_unknown=P(.15), handle_lookalike=P(.1))
        elif c == "mule_account":
            amt = U(2000, 40000); f.update(fan_in=U(15, 80), net_age=U(1, 14), complaints=float(r.poisson(3) + 1), new_payee=P(.7), night=P(.4))
        elif c == "fake_job_investment":
            amt = U(500, 50000); f.update(invest_word=P(.9), urgency_word=P(.5), vpa_random=P(.6), new_payee=P(.9), vpa_risky_words=P(.3), complaints=float(r.poisson(1)))
        elif c == "impersonation":
            f.update(vpa_risky_words=P(.9), handle_lookalike=P(.4), handle_unknown=P(.4), name_mismatch=U(.4, .9) if P(.5) else 0., new_payee=P(.8), urgency_word=P(.3), complaints=float(r.poisson(1)))
        f.update(amount_log=float(np.log1p(amt)), amount_ratio=ratio, collect=float(mode == "collect"), qr=float(mode == "qr"), link=float(mode == "link"))
        X.append([float(f[k]) for k in FEATS]); y.append(0 if c == "legit" else 1); C.append(c)
    return np.array(X), np.array(y), np.array(C)


def mets(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return dict(precision=round(tp / max(tp + fp, 1), 4), recall=round(tp / max(tp + fn, 1), 4), fpr=round(fp / max(fp + tn, 1), 4),
                accuracy=round((tp + tn) / len(y), 4), tp=tp, fp=fp, fn=fn, tn=tn)


def train(feedback=()):
    """feedback: iterable of (feature_dict, 'fraud'|'legit') from analysts and user scam reports; weighted x15."""
    feedback = list(feedback)
    X, y, C = synth(9000, 1)
    Xa, Xt, ya, yt, Ca, Ct = train_test_split(X, y, C, test_size=.25, stratify=y, random_state=7)
    w = np.ones(len(ya))
    if feedback:
        Xa = np.vstack([Xa, [[float(f.get(k, 0)) for k in FEATS] for f, _ in feedback]])
        ya = np.concatenate([ya, [1 if l == "fraud" else 0 for _, l in feedback]]); w = np.concatenate([w, np.full(len(feedback), 15.)])
    clf = RandomForestClassifier(200, min_samples_leaf=2, class_weight="balanced", random_state=7, n_jobs=-1).fit(Xa, ya, sample_weight=w)
    p = clf.predict_proba(Xt)[:, 1]; pred = (p >= .4).astype(int)
    ai = {k: i for i, k in enumerate(FEATS)}; amt = np.expm1(Xt[:, ai["amount_log"]])
    base = ((amt > 10000) | ((Xt[:, ai["new_payee"]] == 1) & (amt > 5000))).astype(int)
    m = dict(test_size=len(yt), fraud_in_test=int(yt.sum()), threshold=.4, model=mets(yt, pred), static_rule_baseline=mets(yt, base),
             per_category_recall={k: round(float(pred[Ct == k].mean()), 3) for k in CATS if k in set(Ct)},
             feedback_samples=len(feedback), data="synthetic hold-out set built from UPI scam typologies + your labelled cases",
             trained_at=datetime.now().isoformat(timespec="seconds"))
    M.update(clf=clf, base=np.median(Xa[ya == 0], axis=0), metrics=m)
    return m


def score(f):
    """Risk probability + per-feature attribution (replace each feature by the legit median, measure the drop)."""
    x = np.array([[float(f[k]) for k in FEATS]]); Z = [x[0]]
    for i in range(len(FEATS)):
        z = x[0].copy(); z[i] = M["base"][i]; Z.append(z)
    pr = M["clf"].predict_proba(np.array(Z))[:, 1]; rows = []
    for i, k in enumerate(FEATS):
        d = float(pr[0] - pr[i + 1])
        if d > .01: rows.append(dict(feature=k, label=FL[k], impact=round(d, 3), value=round(float(x[0, i]), 2)))
    return float(pr[0]), sorted(rows, key=lambda r: -r["impact"])[:6]


# ---------------------------------------------------------------- real payment context -> features
def name_gap(a, b):
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b: return 0.
    srt = lambda s: " ".join(sorted(s.split()))
    return round(1 - max(difflib.SequenceMatcher(None, a, b).ratio(), difflib.SequenceMatcher(None, srt(a), srt(b)).ratio()), 3)


def featurize(x):
    """x: amount (INR), mode, purpose, payee_vpa, registered_name (PSP-verified or None), display_name, message, known_payee,
    avg_amount (None if no history), net_age, complaints, fan_in, screenshare, remote_app, hour (optional)."""
    amt = float(x["amount"]); mode = x.get("mode", "pay"); t = vpa_traits(x["payee_vpa"])
    txt = " ".join(filter(None, [x.get("message"), x.get("display_name")]))
    hr = int(x["hour"]) if x.get("hour") is not None else datetime.now().hour
    avg = x.get("avg_amount")
    return dict(amount_log=float(np.log1p(amt)), night=float(hr < 5 or hr >= 23), amount_ratio=min(amt / max(float(avg), 1), 30) if avg else 1.0,
                new_payee=float(not x.get("known_payee")), net_age=float(x.get("net_age", 0)), complaints=float(x.get("complaints", 0)),
                fan_in=float(x.get("fan_in", 0)), collect=float(mode == "collect"), qr=float(mode == "qr"), link=float(mode == "link"),
                refund_word=float(bool(REF.search(txt))), kyc_word=float(bool(KYC.search(txt))), urgency_word=float(bool(URG.search(txt))),
                invest_word=float(bool(INV.search(txt)) or x.get("purpose") == "job_invest"),
                name_mismatch=name_gap(x.get("registered_name"), x.get("display_name")),
                screenshare=float(bool(x.get("screenshare"))), remote_app=float(bool(x.get("remote_app"))),
                handle_unknown=t["handle_unknown"], handle_lookalike=t["handle_lookalike"], vpa_risky_words=t["vpa_risky_words"],
                vpa_random=t["vpa_random"], receive_purpose=float(x.get("purpose") == "receive_money"))


def indicators(f, d):
    I = []; a = lambda t, s: I.append(dict(text=t, severity=s)); amt = d["amount"]; vpa = d.get("payee_vpa") or "this UPI ID"
    if f["collect"]: a(f"This is a REQUEST: approving it and entering your PIN sends ₹{amt:,.0f} OUT of your account.", "high" if f["refund_word"] else "medium")
    if f["receive_purpose"]: a("You are paying in order to 'receive' money. You never pay to receive a refund, prize, loan or payout.", "high")
    if f["refund_word"]: a("'Refund / prize / cashback' wording: real refunds never ask you to approve a debit or scan a QR.", "high")
    if f["name_mismatch"] > .4: a(f"Name given ('{d.get('display_name')}') does not match the account holder ('{d.get('registered_name')}').", "high")
    if f["screenshare"] or f["remote_app"]: a("Someone is guiding this payment or a remote-access app is running. This is the #1 pattern in coached scams.", "high")
    if f["kyc_word"]: a("Message uses KYC / account-block wording typical of fake bank alerts.", "high")
    if f["invest_word"]: a("Task / job / investment wording: these schemes ask for small payments first, then bigger ones, then vanish.", "high" if f["new_payee"] else "medium")
    if f["urgency_word"]: a("The message pressures you to act urgently.", "medium")
    if f["fan_in"] >= 15 and f["net_age"] < 30 and f["net_age"] > 0: a(f"{vpa} is new on our network but has received from {f['fan_in']:.0f} different senders: mule-account pattern.", "high")
    if f["complaints"] >= 1: a(f"{f['complaints']:.0f} fraud report(s) recorded against {vpa}.", "high" if f["complaints"] >= 2 else "medium")
    if f["handle_lookalike"]: a("The part after @ looks like a real bank/app handle but is not (a classic look-alike trick).", "high")
    elif f["handle_unknown"]: a("The part after @ is not a UPI handle we recognise: check for a typo.", "medium")
    if f["vpa_risky_words"]: a("UPI ID uses official-sounding words (support, refund, bank or brand names). Genuine institutions use verified merchant IDs.", "medium")
    if f["vpa_random"]: a("UPI ID is a long auto-generated number string, not a name or shop.", "low")
    if f["new_payee"] and f["amount_ratio"] > 4: a("First payment to this UPI ID and well above your usual amount.", "medium")
    if f["night"]: a("Payment at an unusual hour.", "low")
    return I


def grok(ctx):
    if not XAI_KEY: return None
    sysm = ("You are a UPI fraud analyst for India. Given a payment context, an ML risk score and rule indicators, decide whether it is a scam. "
            "Known categories: collect_refund, qr_swap, remote_access, fake_kyc, mule_account, fake_job_investment, impersonation, none. If it is a scam fitting none of these, "
            "use category 'novel' with new_pattern_name (snake_case) and 3-6 lowercase keywords from the message that characterise it. "
            'Reply ONLY with JSON: {"fraud_confidence":0-1,"category":str,"new_pattern_name":str|null,"keywords":[str],'
            '"user_warning":"one short sentence to the payer starting with the key fact","explanation":"max 2 plain sentences citing concrete indicators"}')
    body_ = {"model": GROK_MODEL, "temperature": .1, "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": json.dumps(ctx)}]}
    rq = urllib.request.Request("https://api.x.ai/v1/chat/completions", json.dumps(body_).encode(),
                                {"Content-Type": "application/json", "Authorization": "Bearer " + XAI_KEY, "User-Agent": "upi-rakshak/3.0"})
    try:
        out = json.load(urllib.request.urlopen(rq, timeout=25))["choices"][0]["message"]["content"]
        return json.loads(re.search(r"\{.*\}", out, re.S).group())
    except Exception as e:
        return {"error": str(e)[:160]}


def psp_verify(vpa):
    """Optional: ask a real PSP whether the UPI ID exists and who owns it (Razorpay VPA validation API, needs RAZORPAY_KEY_ID/SECRET
    and the feature enabled on your Razorpay account). Returns {'valid': bool, 'name': str|None} or None if unavailable."""
    if not (RZP_ID and RZP_SECRET): return None
    rq = urllib.request.Request("https://api.razorpay.com/v1/payments/validate/vpa", json.dumps({"vpa": vpa}).encode(),
                                {"Content-Type": "application/json", "Authorization": "Basic " + base64.b64encode(f"{RZP_ID}:{RZP_SECRET}".encode()).decode()})
    try:
        d = json.load(urllib.request.urlopen(rq, timeout=10))
        return dict(valid=bool(d.get("success")), name=d.get("customer_name") or None)
    except urllib.error.HTTPError as e:
        try: msg = (json.load(e).get("error") or {}).get("description", "")
        except Exception: msg = ""
        return dict(valid=False, name=None) if e.code == 400 and re.search(r"invalid|not.{0,12}(exist|valid|found)", msg, re.I) else None
    except Exception:
        return None


def assess(x, patterns=(), use_ai=True):
    """Full assessment. `patterns` = analyst-approved patterns [{id,name,keywords}]. Result carries `learned_ids` and `novel`
    (a pattern Grok proposes) for the caller to persist."""
    f = featurize(x); p, contrib = score(f); ind = indicators(f, x); txt = " ".join(filter(None, [x.get("message"), x.get("display_name")])).lower()
    learned = []
    for r in patterns:
        kws = r["keywords"]; need = 1 if len(kws) < 2 else 2
        if kws and sum(k in txt for k in kws) >= need:
            learned.append(r); ind.insert(0, dict(text=f"Matches analyst-approved pattern '{r['name']}' learned from earlier cases.", severity="high"))
    g_ = None
    if use_ai and (p >= .2 or txt.strip() or ind):
        g_ = grok(dict(amount_inr=x["amount"], mode=x.get("mode"), purpose=x.get("purpose"), payee_vpa=x.get("payee_vpa"), verified_name=x.get("registered_name"),
                       name_given=x.get("display_name"), message=x.get("message"), someone_guiding_on_call=bool(x.get("screenshare")),
                       remote_app=bool(x.get("remote_app")), ml_risk=round(p, 3), indicators=[i["text"] for i in ind]))
    gc = float(g_["fraud_confidence"]) if g_ and "fraud_confidence" in g_ else None
    risk = p if gc is None else .6 * p + .4 * gc
    rules = []
    if x.get("blocked"): risk = max(risk, .95); rules.append("blocklist"); ind.insert(0, dict(text=f"{x['payee_vpa']} is on the Rakshak blocklist: {x.get('block_reason') or 'reported for fraud'}.", severity="high"))
    elif f["complaints"] >= 3: risk = max(risk, .8); rules.append("complaints>=3")
    if f["receive_purpose"] and not f["collect"]: risk = max(risk, .75); rules.append("pay_to_receive")
    if (f["screenshare"] or f["remote_app"]) and f["new_payee"] and x["amount"] >= 5000: risk = max(risk, .75); rules.append("coached_large_payment")
    if learned: risk = max(risk, .75)
    dec = "BLOCK" if risk >= .7 else "WARN" if risk >= .4 else "ALLOW"
    cat = "none"
    if dec != "ALLOW":
        if x.get("blocked"): cat = "blocklisted"
        elif learned: cat = learned[0]["name"]
        elif f["screenshare"] or (f["remote_app"] and x["amount"] > 5000): cat = "remote_access"
        elif f["collect"] and (f["refund_word"] or f["receive_purpose"] or f["complaints"] >= 2): cat = "collect_refund"
        elif f["receive_purpose"] or f["refund_word"]: cat = "collect_refund"
        elif f["name_mismatch"] > .4 and f["qr"]: cat = "qr_swap"
        elif f["kyc_word"]: cat = "fake_kyc"
        elif f["fan_in"] >= 15 and f["net_age"] < 30: cat = "mule_account"
        elif f["invest_word"]: cat = "fake_job_investment"
        elif f["handle_lookalike"] or (f["vpa_risky_words"] and (f["handle_unknown"] or f["name_mismatch"] > .4)): cat = "impersonation"
        elif g_ and g_.get("category") not in (None, "none", "novel", "error") and "error" not in g_: cat = g_["category"]
        elif g_ and g_.get("category") == "novel" and g_.get("new_pattern_name"): cat = g_["new_pattern_name"]
        else: cat = "suspicious_other"
    novel = None
    if g_ and g_.get("category") == "novel" and gc and gc >= .6 and g_.get("new_pattern_name"):
        novel = dict(name=g_["new_pattern_name"], description=g_.get("explanation", ""), keywords=[k.lower() for k in g_.get("keywords", [])])
    amt = x["amount"]
    head = ("This UPI ID has been reported for fraud" if cat == "blocklisted" else
            f"This is a request for you to PAY ₹{amt:,.0f}, not to receive" if f["collect"] and dec != "ALLOW" else
            "Someone may be coaching or controlling this payment" if cat == "remote_access" else
            "The name on this QR does not match the account holder" if cat == "qr_swap" else
            "This looks like a scam" if dec != "ALLOW" else "No fraud signals found")
    warn = (g_ or {}).get("user_warning") or (ind[0]["text"] if ind and dec != "ALLOW" else "This payment is consistent with normal behaviour.")
    expl = (g_ or {}).get("explanation") or (" ".join(i["text"] for i in ind[:3]) if ind else "Amount, UPI ID and context look normal.")
    return dict(risk=round(risk, 3), model_risk=round(p, 3), ai_confidence=gc, decision=dec, category=cat, category_label=CATS.get(cat, cat.replace("_", " ").title()),
                headline=head, warning=warn, explanation=expl, indicators=ind, contributions=contrib, rules=rules, ai_used=bool(g_ and "error" not in g_),
                ai_error=(g_ or {}).get("error") if g_ else ("Grok is off (no XAI_API_KEY): ML + rules only" if use_ai else None),
                features=f, learned_ids=[r["id"] for r in learned], novel=novel)


# ============================================================================
# 3. API  (auth, UPI ID lookup, payment lifecycle, analyst console)
# ============================================================================
def secret_key():
    if os.getenv("SECRET_KEY"): return os.environ["SECRET_KEY"]
    p = os.path.join(HERE, ".secret_key")
    if not os.path.exists(p): open(p, "w").write(secrets.token_hex(32))
    return open(p).read().strip()


app = Flask(__name__, static_folder=None)  # never serve the project directory (db, source, .env)
app.config.update(SECRET_KEY=secret_key(), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=os.getenv("HTTPS", "0") == "1", PERMANENT_SESSION_LIFETIME=timedelta(hours=8), JSON_SORT_KEYS=False)


class Fail(Exception):
    def __init__(self, msg, code=400, **extra): self.msg, self.code, self.extra = msg, code, extra


@app.errorhandler(Fail)
def _fail(e): return jsonify(error=e.msg, **e.extra), e.code


@app.errorhandler(404)
def _404(e): return jsonify(error="Not found"), 404


@app.errorhandler(500)
def _500(e): return jsonify(error="Something went wrong on our side. Please try again."), 500


@app.before_request
def _guard():  # JSON-only POSTs: a cross-site <form> cannot send application/json, which blocks CSRF without tokens
    if request.method == "POST" and request.path.startswith("/api/") and request.mimetype != "application/json":
        return jsonify(error="Content-Type must be application/json"), 415


@app.after_request
def _headers(r):
    r.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "same-origin",
                      "Permissions-Policy": "camera=(self)", "Cache-Control": "no-store" if request.path.startswith("/api/") else "no-cache"})
    return r


now = lambda: datetime.now().isoformat(timespec="seconds")
rs = lambda p: round((p or 0) / 100, 2)
js = lambda s, d=None: json.loads(s) if s else d
HITS = defaultdict(deque)


def limit(key, n, secs):
    q = HITS[key]; t = time.time()
    while q and q[0] < t - secs: q.popleft()
    if len(q) >= n: raise Fail("Too many attempts. Please wait a moment and try again.", 429)
    q.append(t)


def audit(c, uid, event, detail=""):
    c.execute("insert into audit(ts,user_id,event,detail) values(?,?,?,?)", (now(), uid, event, str(detail)[:300]))


def login_required(role=None):
    def deco(fn):
        def w(*a, **k):
            uid = session.get("uid")
            if not uid: raise Fail("Please sign in to continue.", 401)
            c = db_conn(); u = c.execute("select * from users where id=?", (uid,)).fetchone(); c.close()
            if not u: session.clear(); raise Fail("Please sign in to continue.", 401)
            if role and u["role"] != role: raise Fail("You do not have access to this area.", 403)
            g.user = u; return fn(*a, **k)
        w.__name__ = fn.__name__; return w
    return deco


def body(): return request.get_json(silent=True) or {}


def used_today(c, uid):
    return c.execute("select coalesce(sum(amount),0) s from payments where user_id=? and status in('INITIATED','PAID') and substr(updated,1,10)=?",
                     (uid, datetime.now().strftime("%Y-%m-%d"))).fetchone()["s"]


def user_out(u, c):
    return dict(id=u["id"], name=u["name"], phone=u["phone"], role=u["role"], daily_limit=rs(u["daily_limit"]), daily_used=rs(used_today(c, u["id"])),
                ai_enabled=bool(XAI_KEY), ai_model=GROK_MODEL, psp_verify=bool(RZP_ID and RZP_SECRET), purposes=PURPOSES)


def vpa_info(c, vpa, psp=True):
    """Everything known about a UPI ID: syntax traits, PSP-verified name (cached 24h), and what the Rakshak network has seen."""
    t = vpa_traits(vpa); row = c.execute("select * from vpa_cache where vpa=?", (vpa,)).fetchone()
    if psp and (RZP_ID and RZP_SECRET) and (not row or (datetime.now() - datetime.fromisoformat(row["ts"])).total_seconds() > 86400):
        v = psp_verify(vpa)
        if v:
            c.execute("insert into vpa_cache(vpa,valid,name,ts) values(?,?,?,?) on conflict(vpa) do update set valid=excluded.valid,name=excluded.name,ts=excluded.ts",
                      (vpa, int(v["valid"]), v["name"], now())); row = c.execute("select * from vpa_cache where vpa=?", (vpa,)).fetchone()
    first = c.execute("select min(created) m from payments where payee_vpa=?", (vpa,)).fetchone()["m"]
    senders = c.execute("select count(distinct user_id) n from payments where payee_vpa=? and status in('INITIATED','PAID')", (vpa,)).fetchone()["n"]
    reps = c.execute("select count(*) n from reports where vpa=?", (vpa,)).fetchone()["n"]
    lab = c.execute("select count(*) n from payments where payee_vpa=? and analyst_label='fraud'", (vpa,)).fetchone()["n"]
    bl = c.execute("select reason from blocklist where vpa=?", (vpa,)).fetchone()
    return dict(vpa=vpa, **{k: t[k] for k in ("user", "handle", "handle_known", "phone_based")},
                psp_checked=bool(row), exists=None if not row else bool(row["valid"]), verified_name=row["name"] if row else None,
                net_age=round((datetime.now() - datetime.fromisoformat(first)).total_seconds() / 86400, 1) if first else 0.0,
                seen=bool(first), fan_in=senders, complaints=reps + lab, blocked=bool(bl), block_reason=bl["reason"] if bl else None)


def pay_out(r, detail=False):
    o = dict(id=r["id"], ref=r["ref"], amount=rs(r["amount"]), mode=r["mode"], purpose=r["purpose"], note=r["note"], status=r["status"],
             decision=r["decision"], risk=r["risk"], category=r["category"], category_label=CATS.get(r["category"], (r["category"] or "").replace("_", " ").title()),
             payee_vpa=r["payee_vpa"], payee_name=r["payee_name"], display_name=r["display_name"], utr=r["utr"], created=r["created"], updated=r["updated"])
    if detail: o.update(reasons=js(r["reasons"], []), ai=js(r["ai"], {}), contrib=js(r["contrib"], []))
    return o


def upi_links(p):
    name = p["display_name"] or p["payee_name"] or p["payee_vpa"].split("@")[0]
    qs = urllib.parse.urlencode({"pa": p["payee_vpa"], "pn": name, "am": f"{p['amount'] / 100:.2f}", "cu": "INR"}, quote_via=urllib.parse.quote)
    out = dict(generic=f"upi://pay?{qs}", gpay=f"gpay://upi/pay?{qs}", phonepe=f"phonepe://pay?{qs}", paytm=f"paytmmp://pay?{qs}", qr_svg=None)
    try:  # QR for paying from another phone (needs `pip install qrcode`)
        import qrcode, qrcode.image.svg
        raw = qrcode.make(out["generic"], image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2).to_string()
        raw = raw.decode() if isinstance(raw, bytes) else raw
        out["qr_svg"] = re.sub(r"^<\?xml[^>]*\?>\s*", "", raw)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- pages / auth
@app.get("/")
def home(): return send_from_directory(HERE, "index.html")


@app.get("/healthz")
def healthz(): return jsonify(ok=True, model_trained=bool(M))


@app.post("/api/auth/register")
def register():
    d = body(); name = (d.get("name") or "").strip(); phone = re.sub(r"\D", "", d.get("phone") or "")[-10:]; pw = d.get("password") or ""
    if not 2 <= len(name) <= 60: raise Fail("Enter your full name.")
    if not re.fullmatch(r"[6-9]\d{9}", phone): raise Fail("Enter a valid 10-digit Indian mobile number.")
    if len(pw) < 8: raise Fail("Password must be at least 8 characters.")
    limit("reg:" + request.remote_addr, 10, 600)
    with db_tx() as c:
        if c.execute("select 1 from users where phone=?", (phone,)).fetchone(): raise Fail("This mobile number is already registered. Sign in instead.", 409)
        uid = c.execute("insert into users(name,phone,email,pw_hash,daily_limit,created) values(?,?,?,?,?,?)",
                        (name, phone, (d.get("email") or "").strip() or None, generate_password_hash(pw), DEFAULT_DAILY, now())).lastrowid
        audit(c, uid, "register", phone)
    session.clear(); session["uid"] = uid; session.permanent = True
    c = db_conn(); out = user_out(c.execute("select * from users where id=?", (uid,)).fetchone(), c); c.close()
    return jsonify(user=out), 201


@app.post("/api/auth/login")
def login():
    d = body(); phone = re.sub(r"\D", "", d.get("phone") or "")[-10:]; limit(f"login:{request.remote_addr}:{phone}", 8, 300)
    c = db_conn(); u = c.execute("select * from users where phone=?", (phone,)).fetchone()
    if not u or not check_password_hash(u["pw_hash"], d.get("password") or ""): c.close(); raise Fail("Mobile number or password is incorrect.", 401)
    audit(c, u["id"], "login", request.remote_addr); out = user_out(u, c); c.close()
    session.clear(); session["uid"] = u["id"]; session.permanent = True
    return jsonify(user=out)


@app.post("/api/auth/logout")
def logout(): session.clear(); return jsonify(ok=True)


@app.get("/api/me")
@login_required()
def me():
    c = db_conn(); out = user_out(g.user, c); c.close(); return jsonify(user=out)


# ---------------------------------------------------------------- UPI ID lookup (works for ANY UPI ID)
@app.get("/api/vpa/lookup")
@login_required()
def lookup():
    limit(f"lookup:{g.user['id']}", 40, 60)
    vpa = (request.args.get("vpa") or "").strip().lower()
    if not VPA_RE.fullmatch(vpa): raise Fail("That does not look like a valid UPI ID. It should look like name@bank.")
    c = db_conn(); info = vpa_info(c, vpa)
    info["paid_before"] = c.execute("select count(*) n from payments where user_id=? and payee_vpa=? and status in('INITIATED','PAID')", (g.user["id"], vpa)).fetchone()["n"]
    c.close(); return jsonify(info)


@app.get("/api/payees")
@login_required()
def payees():
    c = db_conn()
    r = [dict(x) for x in c.execute("select payee_vpa vpa, coalesce(payee_name,display_name) name, max(updated) last from payments where user_id=? and status in('INITIATED','PAID') "
                                    "group by payee_vpa order by last desc limit 6", (g.user["id"],))]
    c.close(); return jsonify(r)


# ---------------------------------------------------------------- payment lifecycle
@app.post("/api/payments/check")
@login_required()
def pay_check():
    d = body(); u = g.user; limit(f"check:{u['id']}", 30, 60)
    mode = d.get("mode", "pay")
    if mode not in ("pay", "qr", "link", "collect"): raise Fail("Unknown payment type.")
    vpa = (d.get("payee_vpa") or "").strip().lower()
    if not VPA_RE.fullmatch(vpa): raise Fail("That does not look like a valid UPI ID. It should look like name@bank.")
    try: paise = round(float(d.get("amount")) * 100)
    except Exception: raise Fail("Enter a valid amount.")
    if paise <= 0: raise Fail("Enter an amount greater than ₹0.")
    if paise > PER_TXN_LIMIT: raise Fail("UPI allows up to ₹1,00,000 per payment.")
    purpose = d.get("purpose") if d.get("purpose") in PURPOSES else "other"
    note = (d.get("message") or "").strip()[:400]; disp = (d.get("display_name") or "").strip()[:80]
    c = db_conn()
    try:
        info = vpa_info(c, vpa)
        if info["exists"] is False: raise Fail("This UPI ID is not registered with any bank or app. Check the spelling and try again.", 404)
        known = c.execute("select count(*) n from payments where user_id=? and payee_vpa=? and status in('INITIATED','PAID')", (u["id"], vpa)).fetchone()["n"]
        h = c.execute("select count(*) n, avg(amount)/100.0 a from payments where user_id=? and status in('INITIATED','PAID')", (u["id"],)).fetchone()
        pats = [dict(id=r["id"], name=r["name"], keywords=json.loads(r["keywords"])) for r in c.execute("select * from patterns where status='approved'")]
        ctx = dict(amount=paise / 100, mode=mode, purpose=purpose, payee_vpa=vpa, registered_name=info["verified_name"], display_name=disp or info["verified_name"],
                   message=note, known_payee=known > 0, avg_amount=h["a"] if h["n"] >= 3 else None, net_age=info["net_age"], complaints=info["complaints"],
                   fan_in=info["fan_in"], blocked=info["blocked"], block_reason=info["block_reason"], screenshare=bool(d.get("screenshare")), remote_app=bool(d.get("remote_app")))
        res = assess(ctx, pats)
    finally:
        c.close()
    with db_tx() as c:
        for i in res["learned_ids"]: c.execute("update patterns set hits=hits+1 where id=?", (i,))
        n = res["novel"]
        if n and not c.execute("select 1 from patterns where name=?", (n["name"],)).fetchone():
            c.execute("insert into patterns(name,description,keywords,status,created) values(?,?,?,?,?)", (n["name"], n["description"], json.dumps(n["keywords"]), "pending", now()))
        else: res["novel"] = None
        ref = "RK" + "".join(random.choices("0123456789", k=10)); t = now()
        cur = c.execute("""insert into payments(ref,user_id,payee_vpa,payee_name,display_name,amount,mode,purpose,note,status,risk,model_risk,decision,category,features,reasons,contrib,ai,vpa_verified,created,updated)
                           values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (ref, u["id"], vpa, info["verified_name"], disp or None, paise, mode, purpose, note, "CHECKED", res["risk"], res["model_risk"], res["decision"], res["category"],
                         json.dumps(res["features"]), json.dumps(res["indicators"]), json.dumps(res["contributions"]),
                         json.dumps(dict(warning=res["warning"], explanation=res["explanation"], ai_used=res["ai_used"], rules=res["rules"])), int(bool(info["verified_name"])), t, t))
        audit(c, u["id"], "payment_checked", f"#{cur.lastrowid} {vpa} {res['decision']} {res['category']}")
    for k in ("features", "learned_ids"): res.pop(k)
    res.update(payment_id=cur.lastrowid, ref=ref, payee_vpa=vpa, payee_name=info["verified_name"], name_verified=bool(info["verified_name"]), amount=paise / 100,
               mode=mode, handle_known=info["handle_known"], network_seen=info["seen"], new_pattern=(res.pop("novel") or {}).get("name"))
    return jsonify(res)


@app.post("/api/payments/<int:pid>/proceed")
@login_required()
def proceed(pid):
    """Hand the (already analysed) payment to the user's own UPI app. Only ALLOW / acknowledged WARN / analyst-APPROVED get through."""
    d = body(); u = g.user
    with db_tx() as c:
        p = c.execute("select * from payments where id=? and user_id=?", (pid, u["id"])).fetchone()
        if not p: raise Fail("Payment not found.", 404)
        st = p["status"]
        if st == "CHECKED":
            if p["decision"] == "BLOCK": raise Fail("This payment was blocked as high risk. You can send it for review, report it, or cancel it.", 403, hold=True)
            if p["decision"] == "WARN" and not d.get("acknowledge"): raise Fail("Please confirm that you understand the warning.", 400)
            if (datetime.now() - datetime.fromisoformat(p["created"])).total_seconds() > CHECK_TTL_MIN * 60:
                raise Fail("This safety check has expired. Please check the payment again.", 409, expired=True)
        elif st not in ("APPROVED", "INITIATED"): raise Fail("This payment can no longer be started.", 409)
        if st != "INITIATED":
            if used_today(c, u["id"]) + p["amount"] > u["daily_limit"]: raise Fail(f"This would exceed your daily limit of ₹{rs(u['daily_limit']):,.0f}.", 400)
            c.execute("update payments set status='INITIATED', user_ack=?, updated=? where id=?", (1 if d.get("acknowledge") else 0, now(), pid))
            audit(c, u["id"], "payment_handed_off", f"#{pid} {p['payee_vpa']} {rs(p['amount'])}")
        p = c.execute("select * from payments where id=?", (pid,)).fetchone()
    return jsonify(payment=pay_out(p), links=None if p["mode"] == "collect" else upi_links(p), expected_name=p["payee_name"] or p["display_name"])


@app.post("/api/payments/<int:pid>/outcome")
@login_required()
def outcome(pid):
    d = body(); res = d.get("result"); utr = re.sub(r"\s", "", str(d.get("utr") or ""))
    if res not in ("paid", "failed"): raise Fail("Unknown result.")
    if utr and not re.fullmatch(r"\d{12}", utr): raise Fail("A UPI reference (UTR) is 12 digits. Leave it empty if you do not have it.")
    with db_tx() as c:
        n = c.execute("update payments set status=?, utr=?, updated=? where id=? and user_id=? and status='INITIATED'",
                      ("PAID" if res == "paid" else "FAILED", utr or None, now(), pid, g.user["id"])).rowcount
        if not n: raise Fail("This payment is not waiting for a result.", 409)
        audit(c, g.user["id"], f"outcome_{res}", f"#{pid} utr={utr or '-'}")
    return jsonify(ok=True)


@app.post("/api/payments/<int:pid>/cancel")
@login_required()
def cancel(pid):
    with db_tx() as c:
        n = c.execute("update payments set status='CANCELLED', updated=? where id=? and user_id=? and status in('CHECKED','UNDER_REVIEW','APPROVED','INITIATED')", (now(), pid, g.user["id"])).rowcount
        if not n: raise Fail("This payment cannot be cancelled.", 409)
        audit(c, g.user["id"], "payment_cancelled", pid)
    return jsonify(ok=True)


@app.post("/api/payments/<int:pid>/review")
@login_required()
def send_review(pid):
    with db_tx() as c:
        n = c.execute("update payments set status='UNDER_REVIEW', updated=? where id=? and user_id=? and status='CHECKED' and decision='BLOCK'", (now(), pid, g.user["id"])).rowcount
        if not n: raise Fail("Only blocked payments can be sent for review.", 409)
        audit(c, g.user["id"], "sent_for_review", pid)
    return jsonify(ok=True)


@app.post("/api/payments/<int:pid>/report")
@login_required()
def report(pid):
    with db_tx() as c:
        p = c.execute("select * from payments where id=? and user_id=?", (pid, g.user["id"])).fetchone()
        if not p: raise Fail("Payment not found.", 404)
        if c.execute("select 1 from reports where reporter_id=? and vpa=?", (g.user["id"], p["payee_vpa"])).fetchone(): return jsonify(ok=True, duplicate=True)
        c.execute("insert into reports(reporter_id,vpa,payment_id,reason,ts) values(?,?,?,?,?)", (g.user["id"], p["payee_vpa"], pid, (body().get("reason") or "scam")[:200], now()))
        audit(c, g.user["id"], "reported_vpa", p["payee_vpa"])
    return jsonify(ok=True)


@app.get("/api/payments")
@login_required()
def history():
    c = db_conn(); r = [pay_out(x) for x in c.execute("select * from payments where user_id=? and status!='CHECKED' order by updated desc limit 80", (g.user["id"],))]
    c.close(); return jsonify(r)


@app.get("/api/payments/<int:pid>")
@login_required()
def one(pid):
    c = db_conn(); r = c.execute("select * from payments where id=? and user_id=?", (pid, g.user["id"])).fetchone(); c.close()
    if not r: raise Fail("Payment not found.", 404)
    return jsonify(pay_out(r, True))


# ---------------------------------------------------------------- analyst console
PAY_Q = "select p.*, s.name sender_name, s.phone sender_phone from payments p left join users s on s.id=p.user_id"


@app.get("/api/admin/stats")
@login_required("analyst")
def stats():
    c = db_conn(); q = lambda s, *a: c.execute(s, a).fetchall()
    t = q("select count(*) n,sum(decision!='ALLOW') fl,sum(decision='BLOCK') bl,sum(case when decision!='ALLOW' then amount else 0 end) val from payments")[0]
    held = q("select coalesce(sum(amount),0) s from payments where decision!='ALLOW' and status in('CANCELLED','REJECTED','UNDER_REVIEW','FAILED')")[0]["s"]
    cats = [dict(category=r["category"], label=CATS.get(r["category"], r["category"].replace("_", " ").title()), n=r["n"]) for r in
            q("select category,count(*) n from payments where decision!='ALLOW' group by category order by n desc")]
    tr = [dict(day=r["d"], total=r["n"], flagged=r["fl"] or 0) for r in q("select substr(created,1,10) d,count(*) n,sum(decision!='ALLOW') fl from payments group by d order by d desc limit 7")][::-1]
    out = dict(total=t["n"], flagged=t["fl"] or 0, blocked=t["bl"] or 0, value_at_risk=rs(t["val"]), value_held_back=rs(held), categories=cats, trend=tr,
               pending_patterns=q("select count(*) n from patterns where status='pending'")[0]["n"], under_review=q("select count(*) n from payments where status='UNDER_REVIEW'")[0]["n"],
               users=q("select count(*) n from users where role='user'")[0]["n"], paid=rs(q("select coalesce(sum(amount),0) s from payments where status='PAID'")[0]["s"]),
               blocklisted=q("select count(*) n from blocklist")[0]["n"], ai_enabled=bool(XAI_KEY), ai_model=GROK_MODEL, psp_verify=bool(RZP_ID and RZP_SECRET))
    c.close(); return jsonify(out)


@app.get("/api/admin/transactions")
@login_required("analyst")
def admin_txns():
    cond = {"flagged": " where p.decision!='ALLOW'", "review": " where p.status='UNDER_REVIEW'"}.get(request.args.get("view", "flagged"), "")
    c = db_conn(); rows = []
    for r in c.execute(PAY_Q + cond + " order by p.created desc limit 150").fetchall():
        o = pay_out(r, True); o.update(sender=r["sender_name"] or "Unknown", sender_phone=r["sender_phone"], analyst_label=r["analyst_label"]); rows.append(o)
    c.close(); return jsonify(rows)


@app.post("/api/admin/payments/<int:pid>/resolve")
@login_required("analyst")
def admin_resolve(pid):
    act = body().get("action")
    if act not in ("release", "reject"): raise Fail("Unknown action.")
    with db_tx() as c:
        n = c.execute("update payments set status=?, updated=? where id=? and status='UNDER_REVIEW'", ("APPROVED" if act == "release" else "REJECTED", now(), pid)).rowcount
        if not n: raise Fail("This payment is not waiting for review.", 409)
        audit(c, g.user["id"], f"review_{act}", pid)
    return jsonify(ok=True)


@app.post("/api/admin/payments/<int:pid>/label")
@login_required("analyst")
def admin_label(pid):
    l = body().get("label")
    if l not in ("fraud", "legit"): raise Fail("Label must be fraud or legit.")
    with db_tx() as c: c.execute("update payments set analyst_label=? where id=?", (l, pid))
    return jsonify(ok=True)


@app.get("/api/admin/metrics")
@login_required("analyst")
def metrics(): return jsonify(M["metrics"])


def retrain():
    c = db_conn()
    fb = [(json.loads(r["features"]), r["lbl"]) for r in c.execute(
        "select features, coalesce(analyst_label,'fraud') lbl from payments where analyst_label is not null or id in(select payment_id from reports where payment_id is not null)")]
    m = train(fb); c.execute("insert into model_meta(ts,metrics) values(?,?)", (m["trained_at"], json.dumps(m))); c.close(); return m


@app.post("/api/admin/retrain")
@login_required("analyst")
def admin_retrain(): return jsonify(retrain())


@app.get("/api/admin/patterns")
@login_required("analyst")
def patterns():
    c = db_conn(); r = [dict(x, keywords=json.loads(x["keywords"])) for x in c.execute("select * from patterns order by id desc")]; c.close(); return jsonify(r)


@app.post("/api/admin/patterns/<int:i>")
@login_required("analyst")
def pattern_set(i):
    s = body().get("status")
    if s not in ("approved", "rejected"): raise Fail("Unknown status.")
    with db_tx() as c: c.execute("update patterns set status=? where id=?", (s, i))
    return jsonify(ok=True)


@app.get("/api/admin/blocklist")
@login_required("analyst")
def bl_list():
    c = db_conn(); r = [dict(x) for x in c.execute("select * from blocklist order by ts desc")]; c.close(); return jsonify(r)


@app.post("/api/admin/blocklist")
@login_required("analyst")
def bl_edit():
    d = body(); vpa = (d.get("vpa") or "").strip().lower()
    if not VPA_RE.fullmatch(vpa): raise Fail("Enter a valid UPI ID.")
    with db_tx() as c:
        if d.get("action") == "remove": c.execute("delete from blocklist where vpa=?", (vpa,))
        else: c.execute("insert into blocklist(vpa,reason,added_by,ts) values(?,?,?,?) on conflict(vpa) do update set reason=excluded.reason", (vpa, (d.get("reason") or "reported for fraud")[:200], g.user["id"], now()))
        audit(c, g.user["id"], f"blocklist_{d.get('action', 'add')}", vpa)
    return jsonify(ok=True)


@app.get("/api/admin/users")
@login_required("analyst")
def admin_users():
    c = db_conn()
    r = [dict(id=x["id"], name=x["name"], phone=x["phone"], role=x["role"], created=x["created"][:10], payments=x["n"], flagged=x["fl"] or 0)
         for x in c.execute("select u.*, (select count(*) from payments where user_id=u.id) n, (select sum(decision!='ALLOW') from payments where user_id=u.id) fl from users u order by u.id desc limit 100")]
    c.close(); return jsonify(r)


# ---------------------------------------------------------------- startup
def bootstrap():
    """Create tables and the analyst login. Set ANALYST_PHONE / ANALYST_PASSWORD in .env to choose them (the password is re-applied on every start);
    otherwise a random password is generated once and printed."""
    db_init(); c = db_conn(); phone = os.getenv("ANALYST_PHONE", "9000000001"); pw = os.getenv("ANALYST_PASSWORD")
    a = c.execute("select id from users where role='analyst' limit 1").fetchone()
    if a and pw: c.execute("update users set pw_hash=? where id=?", (generate_password_hash(pw), a["id"]))
    elif not a:
        gen = pw or secrets.token_urlsafe(9)
        c.execute("insert into users(name,phone,pw_hash,role,daily_limit,created) values(?,?,?,?,?,?) on conflict(phone) do update set role='analyst', pw_hash=excluded.pw_hash",
                  ("Fraud Analyst", phone, generate_password_hash(gen), "analyst", 0, now()))
        print(f"\n  Analyst login  ->  mobile {phone}   password {gen}" + ("" if pw else "   (generated once: set ANALYST_PASSWORD in .env to choose your own)") + "\n")
    c.close()


bootstrap(); retrain()
if __name__ == "__main__":
    print("UPI Rakshak  |  Grok:", f"ON ({GROK_MODEL})" if XAI_KEY else "off (set XAI_API_KEY)", " |  PSP name check:", "ON (Razorpay)" if RZP_ID and RZP_SECRET else "off (set RAZORPAY_KEY_ID/SECRET)")
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 5000)))
