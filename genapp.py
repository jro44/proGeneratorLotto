# genapp.py
# Lotto 6/49 Quantum Learning AI v3 CLEAN
# Stabilna wersja Streamlit + Firebase:
# - jedna baza historii Firebase: lotto649_ai_history
# - brak dublujących się widgetów, bo aplikacja używa menu zamiast renderowania wszystkich zakładek naraz
# - parser PDF 999 losowań
# - generator Anty-Błąd PRO, Elitarny, Test A/B, FINAL TOP, Eksperymenty
# - Strażnik AI, Anty-Powtórka 999, jakość 0–100
# - uczenie na Twoich realnych kuponach
# - rekomendowana suma, balans parzyste/nieparzyste, niskie/wysokie

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:
    firebase_admin = None
    credentials = None
    firestore = None

APP_NAME = "Lotto 6/49 Quantum Learning AI v3 CLEAN"
DEFAULT_PDF_NAME = "Wyniki060626.pdf"
NUMBER_MIN = 1
NUMBER_MAX = 49
DRAW_SIZE = 6
EXPECTED_DRAW_COUNT = 999
FIREBASE_COLLECTION = "lotto649_ai_history"
LOCAL_HISTORY = Path("lotto649_ai_history_local.csv")


@dataclass
class Draw:
    draw_id: int
    numbers: Tuple[int, ...]


@dataclass
class Settings:
    count: int
    window: int
    sum_min: int
    sum_max: int
    even_min: int
    even_max: int
    low_min: int
    low_max: int
    max_chain: int
    max_sector: int
    max_latest: int
    risk: str
    profile: str
    attempts: int


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_nums(text: str) -> Tuple[int, ...]:
    values = [int(x) for x in re.findall(r"\d+", str(text))]
    values = [n for n in values if NUMBER_MIN <= n <= NUMBER_MAX]
    return tuple(sorted(set(values)))


def doc_id(prefix: str) -> str:
    raw = f"{prefix}_{time.time_ns()}_{random.randint(100000,999999)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def resolve_file(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    target = p.name.lower()
    for f in Path(".").iterdir():
        if f.name.lower() == target:
            return f
    return p


def file_sig(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(1024 * 512))
    return f"{path.name}_{stat.st_size}_{stat.st_mtime_ns}_{h.hexdigest()[:12]}"


class FirebaseClean:
    def __init__(self):
        self.enabled = False
        self.error = ""
        self.db = None
        self.init()

    def init(self):
        if firebase_admin is None:
            self.error = "firebase-admin nie jest zainstalowane."
            return
        try:
            data = None
            if "firebase_service_account" in st.secrets:
                raw = st.secrets["firebase_service_account"]
                data = json.loads(str(raw))
            elif "firebase" in st.secrets:
                data = dict(st.secrets["firebase"])
                data["private_key"] = str(data["private_key"]).replace("\\n", "\n")
            else:
                self.error = "Brak Firebase w Secrets."
                return

            if not firebase_admin._apps:
                cred = credentials.Certificate(data)
                firebase_admin.initialize_app(cred)

            self.db = firestore.client()
            self.enabled = True
            self.error = ""
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)

    def write(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        if not self.enabled or self.db is None:
            self.save_local(payload)
            return False, "Firebase nieaktywny — zapis lokalny CSV."

        try:
            did = doc_id("history")
            payload = dict(payload)
            payload["_doc_id"] = did
            payload["_saved_at"] = now_str()
            self.db.collection(FIREBASE_COLLECTION).document(did).set(payload)
            return True, f"Zapisano do Firebase: {FIREBASE_COLLECTION}/{did}"
        except Exception as exc:
            self.error = str(exc)
            self.save_local(payload)
            return False, f"Błąd Firebase, zapis lokalny CSV: {exc}"

    def save_local(self, payload: Dict[str, Any]) -> None:
        old = pd.read_csv(LOCAL_HISTORY) if LOCAL_HISTORY.exists() else pd.DataFrame()
        new = pd.concat([old, pd.DataFrame([payload])], ignore_index=True)
        new.to_csv(LOCAL_HISTORY, index=False, encoding="utf-8-sig")

    def read(self, limit: int = 5000) -> pd.DataFrame:
        if self.enabled and self.db is not None:
            try:
                rows = []
                for d in self.db.collection(FIREBASE_COLLECTION).limit(limit).stream():
                    item = d.to_dict()
                    item["_doc_id"] = d.id
                    rows.append(item)
                return pd.DataFrame(rows)
            except Exception as exc:
                self.error = str(exc)

        if LOCAL_HISTORY.exists():
            return pd.read_csv(LOCAL_HISTORY)
        return pd.DataFrame()


class Parser:
    @staticmethod
    def parse_result_line(line: str) -> Optional[Tuple[int, ...]]:
        digits = re.sub(r"\D", "", line)
        if len(digits) != DRAW_SIZE * 2:
            return None
        nums = [int(digits[i:i+2]) for i in range(0, len(digits), 2)]
        if len(nums) == DRAW_SIZE and len(set(nums)) == DRAW_SIZE and nums == sorted(nums) and all(1 <= n <= 49 for n in nums):
            return tuple(nums)
        return None

    @staticmethod
    def parse_id_line(line: str) -> Optional[int]:
        tokens = re.findall(r"\d+", line)
        if len(tokens) == 1 and len(tokens[0]) == 4:
            return int(tokens[0])
        return None

    def parse_page(self, text: str) -> List[Draw]:
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if "lotto" in line.lower():
                continue
            if re.search(r"\d", line):
                lines.append(line)

        best_score = -10**9
        best_results = []
        best_ids = []

        for split in range(1, len(lines)):
            results = [x for x in (self.parse_result_line(l) for l in lines[:split]) if x is not None]
            ids = [x for x in (self.parse_id_line(l) for l in lines[split:]) if x is not None]
            if not results or not ids:
                continue
            pairs = min(len(results), len(ids))
            score = pairs * 100 - abs(len(results) - len(ids)) * 20
            if len(results) == len(ids):
                score += 80
            if len(ids) >= 2 and all(ids[i] > ids[i+1] for i in range(len(ids)-1)):
                score += 30
            if score > best_score:
                best_score = score
                best_results = results
                best_ids = ids

        return [Draw(i, n) for i, n in zip(best_ids, best_results)]

    def parse_pdf(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku {path.name}.")
        reader = PdfReader(str(path))
        found: Dict[int, Draw] = {}
        for page in reader.pages:
            text = page.extract_text() or ""
            for draw in self.parse_page(text):
                found[draw.draw_id] = draw
        if not found:
            raise ValueError("Nie odczytano losowań z PDF.")
        rows = []
        for d in found.values():
            row = {"Losowanie": d.draw_id}
            for i, n in enumerate(d.numbers, start=1):
                row[f"N{i}"] = n
            rows.append(row)
        df = pd.DataFrame(rows).drop_duplicates("Losowanie").sort_values("Losowanie", ascending=False).reset_index(drop=True)
        cols = [f"N{i}" for i in range(1, 7)]
        df["Liczby"] = df[cols].apply(lambda r: tuple(int(x) for x in r), axis=1)
        df["Suma"] = df[cols].sum(axis=1)
        df["Parzyste"] = df[cols].apply(lambda r: sum(int(x) % 2 == 0 for x in r), axis=1)
        df["Niskie"] = df[cols].apply(lambda r: sum(int(x) <= 24 for x in r), axis=1)
        return df


@st.cache_data(show_spinner="Czytam PDF...")
def load_pdf_cached(name: str, sig: str) -> pd.DataFrame:
    return Parser().parse_pdf(resolve_file(name))


class Analytics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.cols = [f"N{i}" for i in range(1, 7)]

    def all_numbers(self, window: Optional[int] = None) -> np.ndarray:
        src = self.df.head(window) if window else self.df
        return src[self.cols].to_numpy().flatten().astype(int)

    def freq(self, window: Optional[int] = None) -> pd.DataFrame:
        vals = self.all_numbers(window)
        counts = pd.Series(vals).value_counts().reindex(range(1, 50), fill_value=0)
        draws = len(self.df.head(window)) if window else len(self.df)
        out = pd.DataFrame({
            "Liczba": counts.index.astype(int),
            "Wystąpienia": counts.values.astype(int),
            "% losowań": np.round(counts.values / max(1, draws) * 100, 3)
        })
        q75 = out["Wystąpienia"].quantile(0.75)
        q25 = out["Wystąpienia"].quantile(0.25)
        out["Stan"] = "Neutralna"
        out.loc[out["Wystąpienia"] >= q75, "Stan"] = "Gorąca"
        out.loc[out["Wystąpienia"] <= q25, "Stan"] = "Zimna"
        return out.sort_values(["Wystąpienia", "Liczba"], ascending=[False, True]).reset_index(drop=True)

    def delays(self) -> Dict[int, int]:
        rows = [set(map(int, r)) for r in self.df[self.cols].to_numpy()]
        out = {}
        for n in range(1, 50):
            delay = len(rows)
            for i, row in enumerate(rows):
                if n in row:
                    delay = i
                    break
            out[n] = delay
        return out

    def pair_matrix(self, window: int) -> np.ndarray:
        src = self.df.head(window)
        m = np.zeros((49, 49), dtype=float)
        for row in src[self.cols].to_numpy():
            nums = sorted(map(int, row))
            for a, b in itertools.combinations(nums, 2):
                m[a-1, b-1] += 1
                m[b-1, a-1] += 1
        return m

    @staticmethod
    def sectors(nums: Sequence[int]) -> List[int]:
        return [
            sum(1 <= n <= 10 for n in nums),
            sum(11 <= n <= 20 for n in nums),
            sum(21 <= n <= 30 for n in nums),
            sum(31 <= n <= 40 for n in nums),
            sum(41 <= n <= 49 for n in nums),
        ]

    @staticmethod
    def chain(nums: Sequence[int]) -> int:
        nums = sorted(nums)
        if not nums:
            return 0
        best = cur = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    def history_profile(self, ticket: Sequence[int]) -> Dict[str, Any]:
        s = set(ticket)
        hits = [len(s.intersection(set(map(int, r)))) for r in self.df[self.cols].to_numpy()]
        arr = np.array(hits)
        return {"hist_avg": round(float(arr.mean()), 4), "hist_max": int(arr.max()), "hist_2plus": int((arr >= 2).sum()), "hist_3plus": int((arr >= 3).sum()), "hist_4plus": int((arr >= 4).sum())}

    def recommended_sum_from_pdf(self, window: int = 160) -> Dict[str, Any]:
        src = self.df.head(window).copy()
        sums = src["Suma"].astype(int)
        recent = self.df.head(min(40, len(self.df)))["Suma"].astype(int)
        return {"source": "PDF", "window": window, "sum_min": int(max(21, sums.quantile(0.25))), "sum_max": int(min(294, sums.quantile(0.75))), "sum_center": int(round(float(sums.median()))), "recent_center": int(round(float(recent.median())))}

    def build_weights(self, window: int) -> Dict[str, np.ndarray]:
        global_counts = self.freq(None).sort_values("Liczba")["Wystąpienia"].to_numpy(float) + 1
        recent_counts = self.freq(window).sort_values("Liczba")["Wystąpienia"].to_numpy(float) + 1
        dmap = self.delays()
        delays = np.array([dmap[n] + 1 for n in range(1, 50)], dtype=float)
        global_w = global_counts / global_counts.sum()
        recent_w = recent_counts / recent_counts.sum()
        delay_w = delays / delays.sum()
        uniform = np.ones(49) / 49
        anti = 0.35 * uniform + 0.25 * recent_w + 0.20 * global_w + 0.20 * delay_w
        anti = anti / anti.sum()
        return {"global": global_w, "recent": recent_w, "delay": delay_w, "anti": anti, "uniform": uniform}

    def accept(self, nums: Sequence[int], settings: Settings) -> bool:
        nums = sorted(nums)
        if len(nums) != 6 or len(set(nums)) != 6:
            return False
        total = sum(nums)
        if not (settings.sum_min <= total <= settings.sum_max):
            return False
        even = sum(n % 2 == 0 for n in nums)
        low = sum(n <= 24 for n in nums)
        if not (settings.even_min <= even <= settings.even_max):
            return False
        if not (settings.low_min <= low <= settings.low_max):
            return False
        if self.chain(nums) > settings.max_chain:
            return False
        if max(self.sectors(nums)) > settings.max_sector:
            return False
        latest = set(self.df.iloc[0]["Liczby"])
        if len(set(nums).intersection(latest)) > settings.max_latest:
            return False
        return True

    def score(self, nums: Sequence[int], settings: Settings, weights: Dict[str, np.ndarray], pair: np.ndarray) -> Dict[str, Any]:
        nums = sorted(nums)
        idx = np.array([n-1 for n in nums], dtype=int)
        pair_score = float(pair[np.ix_(idx, idx)].sum() / 2)
        total = sum(nums)
        even = sum(n % 2 == 0 for n in nums)
        low = sum(n <= 24 for n in nums)
        sectors = self.sectors(nums)
        center = (settings.sum_min + settings.sum_max) / 2
        width = max(1, (settings.sum_max - settings.sum_min) / 2)
        sum_score = max(0, 1 - abs(total - center) / width)
        hist = self.history_profile(nums)
        quality = weights["anti"][idx].mean() * 2400 + weights["recent"][idx].mean() * 800 + weights["global"][idx].mean() * 800 + weights["delay"][idx].mean() * 500 + min(pair_score, 25) * 0.7 + sum_score * 18 + hist["hist_avg"] * 12 + hist["hist_3plus"] * 0.04
        return {"Jakość": round(float(quality), 2), "Suma": total, "Parzyste": even, "Nieparzyste": 6-even, "Niskie": low, "Wysokie": 6-low, "Sektory": "-".join(map(str,sectors)), "Łańcuch": self.chain(nums), "Para_bonus": int(pair_score), **hist}


class Guard:
    def __init__(self, analytics: Analytics):
        self.a = analytics

    def profile(self, risk: str) -> Dict[str, Any]:
        if risk == "Bezpieczny": return {"sim": 3, "hot": 4, "cold": 4, "floor": 72}
        if risk == "Agresywny": return {"sim": 4, "hot": 5, "cold": 5, "floor": 58}
        return {"sim": 3, "hot": 4, "cold": 4, "floor": 65}

    def similarity(self, nums: Sequence[int]) -> int:
        s = set(nums)
        best = 0
        for row in self.a.df[self.a.cols].to_numpy():
            best = max(best, len(s.intersection(set(map(int, row)))))
        return best

    def inspect(self, nums: Sequence[int], settings: Settings, freq: pd.DataFrame, base_score: float) -> Dict[str, Any]:
        p = self.profile(settings.risk)
        nums = sorted(nums)
        state = freq.set_index("Liczba")["Stan"].to_dict()
        hot = sum(state.get(n) == "Gorąca" for n in nums)
        cold = sum(state.get(n) == "Zimna" for n in nums)
        sim = self.similarity(nums)
        penalty = 0
        reasons = []
        if sim > p["sim"]: penalty += 12; reasons.append(f"podobieństwo {sim}/6")
        if hot > p["hot"]: penalty += 8; reasons.append("za dużo gorących")
        if cold > p["cold"]: penalty += 8; reasons.append("za dużo zimnych")
        if len(set([n % 10 for n in nums])) <= 3: penalty += 4; reasons.append("podobne końcówki")
        if max(self.a.sectors(nums)) >= 4: penalty += 8; reasons.append("za dużo z sektora")
        final = max(0, min(100, 52 + base_score / 5 - penalty))
        return {"Jakość_FINAL_0_100": round(float(final), 2), "Ocena_FINAL": self.label(final), "Podobieństwo_999": sim, "Gorące": hot, "Zimne": cold, "Strażnik": "OK" if final >= p["floor"] else "RYZYKO", "Uwagi": "; ".join(reasons) if reasons else "OK"}

    @staticmethod
    def label(x: float) -> str:
        if x >= 90: return "wybitny"
        if x >= 80: return "bardzo mocny"
        if x >= 70: return "mocny"
        if x >= 60: return "dobry"
        if x >= 50: return "ryzykowny"
        return "odradzany"


class Generator:
    def __init__(self, analytics: Analytics):
        self.a = analytics

    def weights_for_style(self, weights: Dict[str, np.ndarray], style: str) -> np.ndarray:
        if style == "Elitarny":
            w = 0.45 * weights["anti"] + 0.25 * weights["recent"] + 0.20 * weights["global"] + 0.10 * weights["delay"]
        elif "Kontrtrend" in style:
            w = 0.45 * weights["delay"] + 0.30 * weights["uniform"] + 0.25 * weights["anti"]
        elif "Gorące" in style:
            w = 0.60 * weights["global"] + 0.40 * weights["recent"]
        elif "Zimne" in style:
            w = 0.70 * weights["delay"] + 0.30 * weights["uniform"]
        elif "Hybryda" in style:
            w = 0.34 * weights["global"] + 0.33 * weights["recent"] + 0.33 * weights["delay"]
        else:
            w = weights["anti"]
        return w / w.sum()

    def generate(self, settings: Settings, style: str, top_mode: bool = False) -> pd.DataFrame:
        rng = np.random.default_rng()
        weights = self.a.build_weights(settings.window)
        pair = self.a.pair_matrix(settings.window)
        freq = self.a.freq(settings.window)
        guard = Guard(self.a)
        w = self.weights_for_style(weights, style)
        attempts = max(settings.attempts, settings.count * 300)
        if top_mode: attempts = max(attempts, 4000)
        rows = []
        seen = set()
        pool = np.arange(1, 50)
        for _ in range(attempts):
            nums = tuple(sorted(rng.choice(pool, size=6, replace=False, p=w).tolist()))
            if nums in seen: continue
            seen.add(nums)
            if not self.a.accept(nums, settings): continue
            meta = self.a.score(nums, settings, weights, pair)
            g = guard.inspect(nums, settings, freq, meta["Jakość"])
            rows.append({"Moduł": style, "Zestaw": " ".join(f"{n:02d}" for n in nums), **meta, **g})
        if not rows:
            for _ in range(800):
                nums = tuple(sorted(rng.choice(pool, size=6, replace=False).tolist()))
                meta = self.a.score(nums, settings, weights, pair)
                g = guard.inspect(nums, settings, freq, meta["Jakość"])
                rows.append({"Moduł": style, "Zestaw": " ".join(f"{n:02d}" for n in nums), **meta, **g})
        out = pd.DataFrame(rows).sort_values(["Jakość_FINAL_0_100", "Jakość"], ascending=False).drop_duplicates("Zestaw")
        return out.head(settings.count).reset_index(drop=True)

    def test_ab(self, settings: Settings) -> pd.DataFrame:
        frames = []
        for style in ["A/B: Bezpieczny balans", "A/B: Kontrtrend", "Elitarny"]:
            s = Settings(**{**asdict(settings), "count": 1})
            frames.append(self.generate(s, style, top_mode=True))
        return pd.concat(frames, ignore_index=True)


def ticket_grid(nums: Sequence[int], title: str) -> str:
    selected = set(nums)
    cells = []
    for n in range(1, 50):
        cls = "sel" if n in selected else ""
        cells.append(f'<div class="cell {cls}">{n:02d}</div>')
    return f"""
    <div class="wrap"><h4>{title}</h4><div class="grid">{''.join(cells)}</div></div>
    <style>.wrap{{background:#111827;border:1px solid #374151;border-radius:16px;padding:14px;margin:12px 0;max-width:560px}}.grid{{display:grid;grid-template-columns:repeat(7,42px);gap:7px}}.cell{{width:42px;height:42px;border-radius:50%;background:#1f2937;color:#d1d5db;display:flex;align-items:center;justify-content:center;font-weight:800;border:1px solid #4b5563}}.sel{{background:radial-gradient(circle at 30% 30%,#fff7ed,#fbbf24 40%,#dc2626);color:#111827;border:2px solid #fde68a;box-shadow:0 0 16px rgba(251,191,36,.9)}}</style>
    """


class App:
    def __init__(self):
        st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")
        self.fb = FirebaseClean()
        self.df = self.load_data()
        self.a = Analytics(self.df)
        self.g = Generator(self.a)

    def load_data(self) -> pd.DataFrame:
        with st.sidebar:
            st.header("📄 Plik")
            name = st.text_input("Nazwa PDF", DEFAULT_PDF_NAME, key="pdf_name")
            if st.button("🔄 Wyczyść cache", width="stretch", key="clear_cache"):
                st.cache_data.clear(); st.rerun()
            st.header("☁️ Firebase")
            if self.fb.enabled:
                st.success("Firebase aktywny")
                st.caption(f"Kolekcja: {FIREBASE_COLLECTION}")
            else:
                st.warning("Firebase nieaktywny")
                st.caption(self.fb.error)
        path = resolve_file(name)
        try:
            return load_pdf_cached(str(path), file_sig(path))
        except Exception as exc:
            st.error(f"Błąd PDF: {exc}"); st.stop()

    def header(self):
        st.title("🧠 Lotto 6/49 Quantum Learning AI v3 CLEAN")
        st.info("Stabilna wersja: generator + Strażnik AI + rekomendowana suma + uczenie na Twoich realnych kuponach przez jedną kolekcję Firebase.")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Losowania", len(self.df)); c2.metric("Najnowsze", int(self.df.iloc[0]["Losowanie"])); c3.metric("Najstarsze", int(self.df.iloc[-1]["Losowanie"])); c4.metric("Firebase", "OK" if self.fb.enabled else "CSV")
        if len(self.df)==EXPECTED_DRAW_COUNT: st.success("Odczytano dokładnie 999 losowań.")

    def history(self) -> pd.DataFrame:
        return self.fb.read()

    def learning_recommendations(self) -> Dict[str, Any]:
        h = self.history(); pdf_rec = self.a.recommended_sum_from_pdf(160)
        if h.empty or "hits" not in h.columns:
            return {"ready": False, **pdf_rec, "best_module": "brak danych", "confidence": "PDF only"}
        h["hits"] = pd.to_numeric(h["hits"], errors="coerce").fillna(-1).astype(int)
        played = h[h["hits"] >= 0].copy()
        if played.empty: return {"ready": False, **pdf_rec, "best_module": "brak ocenionych", "confidence": "PDF only"}
        for col in ["sum","even","low"]: played[col] = pd.to_numeric(played.get(col,0), errors="coerce").fillna(0)
        good = played[played["hits"] >= max(2, played["hits"].quantile(0.70))]
        if good.empty: good=played
        modules = played.groupby("module")["hits"].agg(["count","mean","max"]).reset_index().sort_values(["mean","max","count"], ascending=False)
        return {"ready": True, "source":"PDF + Twoje wyniki", "samples": int(len(played)), "sum_min": int(max(21,good["sum"].quantile(0.20))), "sum_max": int(min(294,good["sum"].quantile(0.80))), "sum_center": int(round(float(good["sum"].median()))), "even": int(good["even"].mode().iloc[0]) if not good["even"].mode().empty else 3, "low": int(good["low"].mode().iloc[0]) if not good["low"].mode().empty else 3, "best_module": str(modules.iloc[0]["module"]) if not modules.empty else "brak", "best_avg": round(float(modules.iloc[0]["mean"]),3) if not modules.empty else 0, "confidence": "wysoka" if len(played)>=80 else "średnia" if len(played)>=30 else "wstępna"}

    def settings(self, prefix: str) -> Settings:
        rec = self.learning_recommendations(); st.subheader("⚙️ Ustawienia")
        if rec.get("ready"): st.success(f"AI rekomenduje sumę: {rec['sum_min']}–{rec['sum_max']} | środek: {rec['sum_center']} | najlepszy moduł: {rec['best_module']} | pewność: {rec['confidence']}")
        else: st.info(f"Rekomendowana suma z PDF: {rec['sum_min']}–{rec['sum_max']} | środek: {rec['sum_center']}. Po ocenieniu kuponów AI zacznie dostrajać zakres.")
        c1,c2,c3,c4=st.columns(4)
        with c1:
            count=st.slider("Ile kuponów?",1,20,3,key=f"{prefix}_count"); window=st.select_slider("Okno analizy",[80,120,160,220,300,500],value=160,key=f"{prefix}_window")
        with c2:
            smin=st.number_input("Suma od",21,294,int(rec["sum_min"]),key=f"{prefix}_smin"); smax=st.number_input("Suma do",21,294,int(rec["sum_max"]),key=f"{prefix}_smax")
        with c3:
            emin=st.slider("Min parzystych",0,6,2,key=f"{prefix}_emin"); emax=st.slider("Max parzystych",0,6,4,key=f"{prefix}_emax")
        with c4:
            lmin=st.slider("Min niskich 1–24",0,6,2,key=f"{prefix}_lmin"); lmax=st.slider("Max niskich 1–24",0,6,4,key=f"{prefix}_lmax")
        c5,c6,c7,c8=st.columns(4)
        with c5: max_chain=st.slider("Max ciąg",1,4,2,key=f"{prefix}_chain")
        with c6: max_sector=st.slider("Max z sektora",1,6,2,key=f"{prefix}_sector")
        with c7: max_latest=st.slider("Max z ostatniego",0,6,2,key=f"{prefix}_latest")
        with c8:
            risk=st.selectbox("Ryzyko",["Bezpieczny","Zrównoważony","Agresywny"],index=1,key=f"{prefix}_risk"); profile=st.selectbox("Tryb pracy",["Ekspres","PRO","Dokładny"],index=1,key=f"{prefix}_profile")
        attempts={"Ekspres":900,"PRO":2200,"Dokładny":5000}[profile]
        if smin>smax: smin,smax=smax,smin
        if emin>emax: emin,emax=emax,emin
        if lmin>lmax: lmin,lmax=lmax,lmin
        return Settings(count,window,int(smin),int(smax),emin,emax,lmin,lmax,max_chain,max_sector,max_latest,risk,profile,attempts)

    def save_generated(self,result:pd.DataFrame,settings:Settings,source:str):
        st.session_state["last_result"]=result.copy(); st.session_state["last_settings"]=asdict(settings); st.session_state["last_source"]=source; st.session_state["last_generated_at"]=now_str()

    def result_panel(self, prefix: str = 'global'):
        result=st.session_state.get("last_result"); raw_settings=st.session_state.get("last_settings"); source=st.session_state.get("last_source","unknown")
        if not isinstance(result,pd.DataFrame) or result.empty or not isinstance(raw_settings,dict): return
        st.subheader("💾 Ostatni pakiet"); st.caption(f"Źródło: {source} | {st.session_state.get('last_generated_at','')}"); st.dataframe(result,width="stretch",hide_index=True)
        c1,c2=st.columns(2)
        with c1:
            if st.button("💾 Zapisz pakiet do Firebase",width="stretch",key="save_last_package"):
                saved=0; messages=[]
                for _,row in result.iterrows():
                    nums=parse_nums(row["Zestaw"])
                    payload={"type":"generated","created_at":now_str(),"module":str(row.get("Moduł",source)),"source":source,"ticket":" ".join(f"{n:02d}" for n in nums),"draw_id":int(self.df.iloc[0]["Losowanie"]),"draw_numbers":"","hits":-1,"settings":raw_settings,"sum":int(sum(nums)),"even":int(sum(n%2==0 for n in nums)),"low":int(sum(n<=24 for n in nums)),"quality":float(row.get("Jakość",0)),"final_quality":float(row.get("Jakość_FINAL_0_100",0)),"risk":raw_settings.get("risk","")}
                    ok,msg=self.fb.write(payload); messages.append(msg)
                    if ok: saved+=1
                if saved: st.success(f"Zapisano {saved} dokumentów do {FIREBASE_COLLECTION}. Odśwież Firebase.")
                else: st.warning("Nie zapisano do Firebase. " + (messages[-1] if messages else ""))
        with c2:
            if st.button("🧹 Wyczyść pakiet",width="stretch",key="clear_last_package"):
                for k in ["last_result","last_settings","last_source","last_generated_at"]: st.session_state.pop(k,None)
                st.rerun()

    def show_generated(self,df:pd.DataFrame,settings:Settings,source:str):
        self.save_generated(df,settings,source); st.dataframe(df,width="stretch",hide_index=True)
        for _,row in df.iterrows(): st.markdown(ticket_grid(parse_nums(row["Zestaw"]),f"{row['Moduł']}: {row['Zestaw']}"),unsafe_allow_html=True)
        self.result_panel(prefix='show_generated')

    def page_generator(self):
        settings=self.settings("gen"); c1,c2,c3=st.columns(3)
        if c1.button("🛡️ Anty-Błąd PRO",width="stretch",type="primary",key="gen_anti"): self.show_generated(self.g.generate(settings,"Anty-Błąd PRO"),settings,"Anty-Błąd PRO")
        if c2.button("💎 Elitarny",width="stretch",key="gen_elite"): self.show_generated(self.g.generate(Settings(**{**asdict(settings),"count":1,"attempts":max(settings.attempts,4000)}),"Elitarny",top_mode=True),settings,"Elitarny")
        if c3.button("🧪 Test A/B",width="stretch",key="gen_ab"): self.show_generated(self.g.test_ab(settings),settings,"Test A/B")
        self.result_panel(prefix='page_generator')

    def page_final(self):
        settings=self.settings("final"); style=st.selectbox("Styl TOP",["Anty-Błąd PRO","Elitarny","Hybryda","Kontrtrend","Gorące","Zimne"],key="final_style")
        if st.button("🏆 Generuj FINAL TOP",width="stretch",type="primary",key="final_go"): self.show_generated(self.g.generate(settings,style,top_mode=True),settings,"FINAL TOP")
        self.result_panel(prefix='page_final')

    def page_check(self):
        st.header("🎯 Sprawdź mój kupon i ucz AI")
        latest=self.df.iloc[0]; default_draw=int(latest["Losowanie"]); default_result=" ".join(f"{int(latest[f'N{i}']):02d}" for i in range(1,7))
        c1,c2=st.columns(2)
        with c1:
            module=st.selectbox("Strategia",["Anty-Błąd PRO","Elitarny","Test A/B","FINAL TOP","Eksperyment","Ręczny"],key="check_module"); ticket_text=st.text_input("Mój kupon","",key="check_ticket")
        with c2:
            draw_id=st.number_input("Numer losowania",min_value=1,value=default_draw,key="check_draw"); result_text=st.text_input("Wynik losowania",default_result,key="check_result")
        ticket=parse_nums(ticket_text); result=parse_nums(result_text)
        if len(ticket)==6 and len(result)==6:
            hits=len(set(ticket).intersection(result)); st.metric("Trafienia",f"{hits}/6"); st.markdown(ticket_grid(ticket,"Twój kupon"),unsafe_allow_html=True)
            if st.button("💾 Zapisz ocenę do Firebase / ucz AI",width="stretch",type="primary",key="save_check"):
                payload={"type":"evaluated","created_at":now_str(),"module":module,"source":"manual_check","ticket":" ".join(f"{n:02d}" for n in ticket),"draw_id":int(draw_id),"draw_numbers":" ".join(f"{n:02d}" for n in result),"hits":int(hits),"settings":{},"sum":int(sum(ticket)),"even":int(sum(n%2==0 for n in ticket)),"low":int(sum(n<=24 for n in ticket)),"quality":0,"final_quality":0,"risk":""}
                ok,msg=self.fb.write(payload); st.success(msg) if ok else st.warning(msg)
        elif ticket_text: st.warning("Wpisz dokładnie 6 liczb kuponu i 6 liczb wyniku.")

    def page_autopilot(self):
        st.header("🤖 Autopilot AI i rekomendowana suma"); rec=self.learning_recommendations(); c1,c2,c3,c4=st.columns(4)
        c1.metric("Źródło",rec.get("source","PDF")); c2.metric("Suma cel",f"{rec['sum_min']}–{rec['sum_max']}"); c3.metric("Środek sumy",rec["sum_center"]); c4.metric("Pewność",rec.get("confidence","PDF"))
        if rec.get("ready"): st.success(f"Najlepszy moduł: {rec['best_module']} | średnia trafień: {rec['best_avg']} | parzyste około: {rec['even']} | niskie około: {rec['low']}")
        else: st.info("Na razie rekomendacja sumy pochodzi z PDF. Po zapisaniu ocen kuponów AI zacznie wyliczać sumę z Twoich realnych wyników.")
        h=self.history()
        if h.empty: st.warning("Brak historii w Firebase/CSV.")
        else:
            st.subheader("Historia AI"); st.dataframe(h,width="stretch",hide_index=True)
            if "hits" in h.columns:
                hh=h.copy(); hh["hits"]=pd.to_numeric(hh["hits"],errors="coerce").fillna(-1); played=hh[hh["hits"]>=0]
                if not played.empty: st.bar_chart(played["hits"].value_counts().sort_index())

    def page_stats(self):
        st.header("📈 Statystyka PDF"); window=st.select_slider("Okno",[80,120,160,220,300,500],value=160,key="stats_window"); rec=self.a.recommended_sum_from_pdf(window)
        st.success(f"Rekomendowana suma z PDF: {rec['sum_min']}–{rec['sum_max']} | środek: {rec['sum_center']} | świeży środek: {rec['recent_center']}")
        freq=self.a.freq(window); st.dataframe(freq,width="stretch",hide_index=True); st.bar_chart(freq.sort_values("Liczba").set_index("Liczba")["Wystąpienia"])

    def page_firebase_test(self):
        st.header("☁️ Test Firebase"); st.write(f"Kolekcja docelowa: `{FIREBASE_COLLECTION}`")
        st.info("W Firebase 3 kupony zapisują się jako 3 dokumenty w tej samej kolekcji. Kliknij nazwę kolekcji po lewej, żeby zobaczyć listę dokumentów. Na telefonie panel listy dokumentów może być zwinięty lub niewygodny.")
        if st.button("🧪 Zapisz testowy dokument",width="stretch",key="firebase_test"):
            ok,msg=self.fb.write({"type":"debug","created_at":now_str(),"message":"Firebase działa"}); st.success(msg) if ok else st.warning(msg)
        h=self.history(); st.dataframe(h,width="stretch",hide_index=True)

    def run(self):
        self.header(); page=st.sidebar.radio("Menu",["Generator","FINAL TOP","Sprawdź kupon","Autopilot AI","Statystyka PDF","Test Firebase"],key="menu")
        if page=="Generator": self.page_generator()
        elif page=="FINAL TOP": self.page_final()
        elif page=="Sprawdź kupon": self.page_check()
        elif page=="Autopilot AI": self.page_autopilot()
        elif page=="Statystyka PDF": self.page_stats()
        elif page=="Test Firebase": self.page_firebase_test()


if __name__ == "__main__":
    App().run()
