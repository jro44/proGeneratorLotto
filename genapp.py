# generator_lotto_649_antyblad_pro_ai_FINAL.py
# Lotto 6/49 Laboratorium Anty-Błąd PRO AI FINAL
#
# Co robi ta wersja:
# - czyta plik Wyniki060626.pdf / Wyniki060626.PDF z historią Lotto 6/49,
# - generuje kupony w trybie Anty-Błąd PRO,
# - generuje Test A/B: Bezpieczny balans / Kontrtrend / Elitarny,
# - zapisuje kupony i wyniki do Firebase Firestore,
# - ma awaryjny zapis lokalny CSV, gdy Firebase nie działa,
# - buduje DNA Gracza na podstawie Twoich prawdziwych wyników,
# - pokazuje Autopilota AI: rekomendowane ustawienia na podstawie historii skuteczności,
# - nie udaje przewidywania przyszłości — uczy się, które strategie realnie wypadają lepiej u Ciebie.
#
# Wymagane pliki:
#   generator_lotto_649_antyblad_pro_ai_v2.py
#   requirements.txt
#   Wyniki060626.pdf albo Wyniki060626.PDF
#
# Streamlit Secrets:
#   [firebase]
#   type = "service_account"
#   project_id = "..."
#   private_key_id = "..."
#   private_key = """-----BEGIN PRIVATE KEY-----
#   ...
#   -----END PRIVATE KEY-----
#   """
#   client_email = "..."
#   client_id = "..."
#   auth_uri = "https://accounts.google.com/o/oauth2/auth"
#   token_uri = "https://oauth2.googleapis.com/token"
#   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
#   client_x509_cert_url = "..."
#   universe_domain = "googleapis.com"
#
# requirements.txt:
#   streamlit>=1.36.0
#   pypdf>=4.2.0
#   pandas>=2.2.0
#   numpy>=1.26.0
#   firebase-admin>=6.5.0

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


# ============================================================
# KONFIGURACJA
# ============================================================

APP_NAME = "Lotto 6/49 Laboratorium Anty-Błąd PRO AI FINAL"
DEFAULT_PDF_NAME = "Wyniki060626.pdf"

NUMBER_MIN = 1
NUMBER_MAX = 49
DRAW_SIZE = 6
TICKET_SIZE = 6
EXPECTED_DRAW_COUNT = 999

DRAW_ID_MIN = 1000
DRAW_ID_MAX = 9999

LOCAL_GENERATED_LOG = Path("local_generated_tickets.csv")
LOCAL_EVALUATED_LOG = Path("local_evaluated_tickets.csv")
LOCAL_SETTINGS_LOG = Path("local_settings_profile.csv")

FIREBASE_COLLECTION_PREFIX = "lotto649_ai"


# ============================================================
# DATACLASS
# ============================================================

@dataclass(frozen=True)
class ParsedDraw:
    draw_id: int
    numbers: Tuple[int, ...]


@dataclass
class GeneratorSettings:
    module_name: str
    count: int
    rolling_window: int
    sum_min: int
    sum_max: int
    even_min: int
    even_max: int
    low_min: int
    low_max: int
    max_chain: int
    max_one_sector: int
    max_from_latest: int
    candidate_attempts: int
    profile_name: str


@dataclass
class PreparedModel:
    frequency_table: pd.DataFrame
    probability_table: pd.DataFrame
    global_weights: np.ndarray
    rolling_weights: np.ndarray
    delay_weights: np.ndarray
    anti_error_weights: np.ndarray
    pair_matrix_raw: np.ndarray
    pair_matrix_norm: np.ndarray
    latest_draw: Tuple[int, ...]
    ranked_numbers: List[int]


# ============================================================
# NARZĘDZIA
# ============================================================

def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stable_doc_id(prefix: str = "doc") -> str:
    raw = f"{prefix}_{time.time_ns()}_{random.randint(100000, 999999)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def parse_number_list(text: str) -> Tuple[int, ...]:
    nums = [int(x) for x in re.findall(r"\d+", str(text))]
    nums = [n for n in nums if NUMBER_MIN <= n <= NUMBER_MAX]
    return tuple(sorted(set(nums)))


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def resolve_case_insensitive_file(filename: str) -> Path:
    requested = Path(filename)

    if requested.exists():
        return requested

    parent = requested.parent if str(requested.parent) not in ("", ".") else Path(".")
    target = requested.name.lower()

    if parent.exists():
        for candidate in parent.iterdir():
            if candidate.name.lower() == target:
                return candidate

    return requested


def file_signature(path: Path) -> str:
    if not path.exists():
        return "missing"

    stat = path.stat()
    h = hashlib.sha256()

    with path.open("rb") as f:
        h.update(f.read(1024 * 1024))

    return f"{path.name}_{stat.st_size}_{stat.st_mtime_ns}_{h.hexdigest()[:16]}"


def keep_awake_block(minutes: int = 12) -> None:
    interval_ms = max(3, minutes) * 60 * 1000
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {interval_ms});
        </script>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FIREBASE BACKEND
# ============================================================

class FirebaseBackend:
    """
    Warstwa trwałej pamięci.
    Najpierw próbuje Firebase Firestore.
    Gdy Firebase jest niedostępny, aplikacja nie przestaje działać:
    przechodzi na lokalne CSV.
    """

    def __init__(self):
        self.enabled = False
        self.db = None
        self.error_message = ""
        self._init_firebase()

    def _init_firebase(self) -> None:
        if firebase_admin is None or credentials is None or firestore is None:
            self.enabled = False
            self.error_message = "firebase-admin nie jest zainstalowane albo nie udało się go zaimportować."
            return

        try:
            firebase_dict = None

            # Preferowany, stabilny format Streamlit Secrets:
            #
            # [firebase]
            # type = "service_account"
            # project_id = "..."
            # private_key_id = "..."
            # private_key = """-----BEGIN PRIVATE KEY-----
            # ...
            # -----END PRIVATE KEY-----
            # """
            # client_email = "..."
            #
            # Ten format omija problemy z JSON-em i znakami kontrolnymi.
            if "firebase" in st.secrets:
                firebase_dict = dict(st.secrets["firebase"])

            # Kompatybilność wsteczna ze starszym formatem:
            #
            # firebase_service_account = """
            # { ... JSON ... }
            # """
            elif "firebase_service_account" in st.secrets:
                firebase_json = st.secrets["firebase_service_account"]

                if isinstance(firebase_json, dict):
                    firebase_dict = dict(firebase_json)
                else:
                    firebase_dict = json.loads(str(firebase_json))

            else:
                self.enabled = False
                self.error_message = "Brak konfiguracji Firebase w Streamlit Secrets. Użyj sekcji [firebase]."
                return

            required_fields = [
                "type",
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
                "client_id",
                "auth_uri",
                "token_uri",
                "auth_provider_x509_cert_url",
                "client_x509_cert_url",
            ]

            missing = [field for field in required_fields if field not in firebase_dict or not firebase_dict[field]]

            if missing:
                self.enabled = False
                self.error_message = "Brakuje pól w Secrets [firebase]: " + ", ".join(missing)
                return

            # Streamlit TOML z blokiem """...""" przechowuje klucz z prawdziwymi nowymi liniami.
            # Jeśli ktoś jednak wkleił \n jako znaki tekstowe, zamieniamy je na nowe linie.
            firebase_dict["private_key"] = str(firebase_dict["private_key"]).replace("\\n", "\n")

            if not firebase_admin._apps:
                cred = credentials.Certificate(firebase_dict)
                firebase_admin.initialize_app(cred)

            self.db = firestore.client()
            self.enabled = True
            self.error_message = ""

        except Exception as exc:
            self.enabled = False
            self.db = None
            self.error_message = str(exc)

    def status_label(self) -> str:
        if self.enabled:
            return "✅ Firebase aktywny"
        return "⚠️ Tryb lokalny CSV"

    def collection(self, name: str):
        if not self.enabled or self.db is None:
            return None
        return self.db.collection(f"{FIREBASE_COLLECTION_PREFIX}_{name}")

    def add_document(self, collection_name: str, data: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False

        try:
            doc_id = stable_doc_id(collection_name)
            self.collection(collection_name).document(doc_id).set(data)
            return True
        except Exception as exc:
            self.error_message = str(exc)
            self.enabled = False
            return False

    def read_collection(self, collection_name: str, limit: int = 5000) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()

        try:
            docs = self.collection(collection_name).limit(limit).stream()
            rows = []
            for doc in docs:
                row = doc.to_dict()
                row["_doc_id"] = doc.id
                rows.append(row)
            return pd.DataFrame(rows)
        except Exception as exc:
            self.error_message = str(exc)
            return pd.DataFrame()

    def update_document(self, collection_name: str, doc_id: str, data: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False

        try:
            self.collection(collection_name).document(doc_id).update(data)
            return True
        except Exception as exc:
            self.error_message = str(exc)
            return False


# ============================================================
# PARSER PDF
# ============================================================

class LottoPdfParser:
    """
    Parser pliku Lotto 6/49:
    - wynik: 6 liczb dwucyfrowych,
    - numer losowania: 4 cyfry,
    - format mapy PDF: najpierw wyniki, potem numery losowań,
    - kolejność od najnowszego do najstarszego.
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    @staticmethod
    def _parse_result_line(line: str) -> Optional[Tuple[int, ...]]:
        digits = re.sub(r"\D", "", line)

        if len(digits) != DRAW_SIZE * 2:
            return None

        nums = [int(digits[i:i + 2]) for i in range(0, len(digits), 2)]

        if (
            len(nums) == DRAW_SIZE
            and len(set(nums)) == DRAW_SIZE
            and all(NUMBER_MIN <= n <= NUMBER_MAX for n in nums)
            and nums == sorted(nums)
        ):
            return tuple(nums)

        return None

    @staticmethod
    def _parse_draw_id_line(line: str) -> Optional[int]:
        tokens = re.findall(r"\d+", line)

        if len(tokens) != 1:
            return None

        token = tokens[0]

        if len(token) != 4:
            return None

        draw_id = int(token)

        if DRAW_ID_MIN <= draw_id <= DRAW_ID_MAX:
            return draw_id

        return None

    def parse_page(self, text: str) -> List[ParsedDraw]:
        lines: List[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            if "lotto" in line.lower():
                continue

            if re.findall(r"\d+", line):
                lines.append(line)

        best_results: List[Tuple[int, ...]] = []
        best_ids: List[int] = []
        best_score = -10**9

        for split_index in range(1, len(lines)):
            result_rows = [
                parsed
                for parsed in (self._parse_result_line(line) for line in lines[:split_index])
                if parsed is not None
            ]

            draw_ids = [
                parsed
                for parsed in (self._parse_draw_id_line(line) for line in lines[split_index:])
                if parsed is not None
            ]

            if not result_rows or not draw_ids:
                continue

            pair_count = min(len(result_rows), len(draw_ids))
            score = pair_count * 100 - abs(len(result_rows) - len(draw_ids)) * 15

            if len(result_rows) == len(draw_ids):
                score += 60

            if len(draw_ids) >= 2 and all(draw_ids[i] > draw_ids[i + 1] for i in range(len(draw_ids) - 1)):
                score += 30

            if score > best_score:
                best_score = score
                best_results = result_rows
                best_ids = draw_ids

        return [
            ParsedDraw(draw_id=draw_id, numbers=numbers)
            for draw_id, numbers in zip(best_ids, best_results)
        ]

    def parse(self) -> pd.DataFrame:
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f'Nie znaleziono pliku "{self.pdf_path.name}". '
                "Umieść PDF w tym samym folderze co aplikacja."
            )

        reader = PdfReader(str(self.pdf_path))
        all_draws: Dict[int, ParsedDraw] = {}

        for page in reader.pages:
            text = page.extract_text() or ""

            for draw in self.parse_page(text):
                all_draws[draw.draw_id] = draw

        if not all_draws:
            raise ValueError("Nie udało się odczytać losowań z PDF.")

        rows: List[Dict[str, Any]] = []

        for draw in all_draws.values():
            row = {"Losowanie": int(draw.draw_id)}
            for index, number in enumerate(draw.numbers, start=1):
                row[f"N{index}"] = int(number)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Losowanie"])
        df = df.sort_values("Losowanie", ascending=False).reset_index(drop=True)

        number_cols = [f"N{i}" for i in range(1, DRAW_SIZE + 1)]
        df["Liczby"] = df[number_cols].apply(lambda r: tuple(int(x) for x in r), axis=1)
        df["Suma"] = df[number_cols].sum(axis=1)
        df["Parzyste"] = df[number_cols].apply(lambda r: sum(int(x) % 2 == 0 for x in r), axis=1)
        df["Nieparzyste"] = DRAW_SIZE - df["Parzyste"]
        df["Niskie_1_24"] = df[number_cols].apply(lambda r: sum(int(x) <= 24 for x in r), axis=1)
        df["Wysokie_25_49"] = DRAW_SIZE - df["Niskie_1_24"]
        df["Sektor_1_10"] = df[number_cols].apply(lambda r: sum(1 <= int(x) <= 10 for x in r), axis=1)
        df["Sektor_11_20"] = df[number_cols].apply(lambda r: sum(11 <= int(x) <= 20 for x in r), axis=1)
        df["Sektor_21_30"] = df[number_cols].apply(lambda r: sum(21 <= int(x) <= 30 for x in r), axis=1)
        df["Sektor_31_40"] = df[number_cols].apply(lambda r: sum(31 <= int(x) <= 40 for x in r), axis=1)
        df["Sektor_41_49"] = df[number_cols].apply(lambda r: sum(41 <= int(x) <= 49 for x in r), axis=1)

        return df


@st.cache_data(show_spinner="Wczytywanie i parsowanie PDF...")
def cached_load_pdf(pdf_path_text: str, signature: str) -> pd.DataFrame:
    path = resolve_case_insensitive_file(pdf_path_text)
    return LottoPdfParser(path).parse()


def validate_database(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    ok = []
    warn = []

    if len(df) == EXPECTED_DRAW_COUNT:
        ok.append("Odczytano dokładnie 999 losowań.")
    else:
        warn.append(f"Odczytano {len(df)} losowań. Oczekiwano 999.")

    if df.empty:
        warn.append("Baza jest pusta.")
        return ok, warn

    ids = df["Losowanie"].astype(int).tolist()

    if all(ids[i] > ids[i + 1] for i in range(len(ids) - 1)):
        ok.append("Kolejność losowań: od najnowszego do najstarszego.")
    else:
        warn.append("Numery losowań nie są idealnie malejące.")

    diffs = [ids[i] - ids[i + 1] for i in range(len(ids) - 1)]

    if diffs and all(d == 1 for d in diffs):
        ok.append("Numeracja ciągła, bez luk.")
    elif diffs:
        warn.append(f"Wykryto {sum(d != 1 for d in diffs)} przerw w numeracji.")

    if len(ids) == EXPECTED_DRAW_COUNT:
        expected_oldest = ids[0] - (EXPECTED_DRAW_COUNT - 1)
        if ids[-1] == expected_oldest:
            ok.append(f"Zakres logiczny: {ids[0]} → {ids[-1]}.")
        else:
            warn.append(f"Przy najnowszym {ids[0]} najstarszy powinien być {expected_oldest}, a jest {ids[-1]}.")

    return ok, warn


# ============================================================
# ANALITYKA
# ============================================================

class LottoAnalytics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("Losowanie", ascending=False).reset_index(drop=True)
        self.columns = [f"N{i}" for i in range(1, DRAW_SIZE + 1)]

    def all_numbers(self, window: Optional[int] = None) -> np.ndarray:
        source = self.df.head(window) if window else self.df
        return source[self.columns].to_numpy().flatten().astype(int)

    def frequency_table(self, window: Optional[int] = None) -> pd.DataFrame:
        values = self.all_numbers(window)
        counts = pd.Series(values).value_counts().reindex(range(NUMBER_MIN, NUMBER_MAX + 1), fill_value=0)
        source_len = len(self.df.head(window)) if window else len(self.df)

        table = pd.DataFrame(
            {
                "Liczba": counts.index.astype(int),
                "Wystąpienia": counts.values.astype(int),
                "Procent_losowań": np.round(counts.values / max(1, source_len) * 100, 3),
            }
        )

        hot_border = table["Wystąpienia"].quantile(0.75)
        cold_border = table["Wystąpienia"].quantile(0.25)
        table["Stan"] = "Neutralna"
        table.loc[table["Wystąpienia"] >= hot_border, "Stan"] = "Gorąca"
        table.loc[table["Wystąpienia"] <= cold_border, "Stan"] = "Zimna"

        return table.sort_values(["Wystąpienia", "Liczba"], ascending=[False, True]).reset_index(drop=True)

    def delays(self) -> Dict[int, int]:
        rows = [set(map(int, row)) for row in self.df[self.columns].to_numpy()]
        delays = {}

        for n in range(NUMBER_MIN, NUMBER_MAX + 1):
            delay = len(rows)
            for idx, row in enumerate(rows):
                if n in row:
                    delay = idx
                    break
            delays[n] = int(delay)

        return delays

    def pair_matrix_raw(self, window: Optional[int] = None) -> np.ndarray:
        source = self.df.head(window) if window else self.df
        matrix = np.zeros((NUMBER_MAX, NUMBER_MAX), dtype=np.float64)

        for row in source[self.columns].to_numpy():
            nums = sorted(map(int, row))
            for a, b in itertools.combinations(nums, 2):
                ia = a - NUMBER_MIN
                ib = b - NUMBER_MIN
                matrix[ia, ib] += 1.0
                matrix[ib, ia] += 1.0

        return matrix

    @staticmethod
    def sectors(numbers: Sequence[int]) -> List[int]:
        return [
            sum(1 <= n <= 10 for n in numbers),
            sum(11 <= n <= 20 for n in numbers),
            sum(21 <= n <= 30 for n in numbers),
            sum(31 <= n <= 40 for n in numbers),
            sum(41 <= n <= 49 for n in numbers),
        ]

    @staticmethod
    def consecutive_chain(numbers: Sequence[int]) -> int:
        nums = sorted(map(int, numbers))
        if not nums:
            return 0

        longest = 1
        current = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest

    def historical_hit_profile(self, ticket: Sequence[int]) -> Dict[str, Any]:
        ticket_set = set(map(int, ticket))
        hits = []

        for row in self.df[self.columns].to_numpy():
            hits.append(len(ticket_set.intersection(set(map(int, row)))))

        if not hits:
            return {
                "Średnia_trafień": 0.0,
                "Max_trafień": 0,
                "Trafienia_2+": 0,
                "Trafienia_3+": 0,
                "Trafienia_4+": 0,
                "Trafienia_5+": 0,
                "Wynik_historyczny": 0.0,
            }

        arr = np.array(hits, dtype=int)
        hit2 = int(np.sum(arr >= 2))
        hit3 = int(np.sum(arr >= 3))
        hit4 = int(np.sum(arr >= 4))
        hit5 = int(np.sum(arr >= 5))
        avg = float(np.mean(arr))
        max_hit = int(np.max(arr))

        score = avg * 14.0 + hit2 * 0.035 + hit3 * 0.18 + hit4 * 0.85 + hit5 * 3.0 + max_hit * 4.0

        return {
            "Średnia_trafień": round(avg, 4),
            "Max_trafień": max_hit,
            "Trafienia_2+": hit2,
            "Trafienia_3+": hit3,
            "Trafienia_4+": hit4,
            "Trafienia_5+": hit5,
            "Wynik_historyczny": round(float(score), 4),
        }

    def build_model(self, rolling_window: int = 160) -> PreparedModel:
        global_table = self.frequency_table(None)
        rolling_table = self.frequency_table(rolling_window)

        global_counts = global_table.sort_values("Liczba")["Wystąpienia"].to_numpy(dtype=float)
        rolling_counts = rolling_table.sort_values("Liczba")["Wystąpienia"].to_numpy(dtype=float)

        global_weights = global_counts + 1.0
        rolling_weights = rolling_counts + 1.0
        global_weights /= global_weights.sum()
        rolling_weights /= rolling_weights.sum()

        delay_map = self.delays()
        delay_values = np.array([delay_map[n] for n in range(NUMBER_MIN, NUMBER_MAX + 1)], dtype=float)
        delay_weights = delay_values + 1.0
        delay_weights /= delay_weights.sum()

        uniform = np.ones(NUMBER_MAX) / NUMBER_MAX

        anti_error_weights = (
            0.40 * uniform
            + 0.25 * rolling_weights
            + 0.20 * global_weights
            + 0.15 * delay_weights
        )
        anti_error_weights /= anti_error_weights.sum()

        pair_raw = self.pair_matrix_raw(rolling_window)
        max_pair = pair_raw.max()
        pair_norm = pair_raw / max_pair if max_pair > 0 else pair_raw

        probability_table = pd.DataFrame(
            {
                "Liczba": range(NUMBER_MIN, NUMBER_MAX + 1),
                "Waga_globalna": np.round(global_weights, 8),
                "Waga_świeża": np.round(rolling_weights, 8),
                "Waga_opóźnienia": np.round(delay_weights, 8),
                "Waga_antybłąd": np.round(anti_error_weights, 8),
                "Opóźnienie": [delay_map[n] for n in range(NUMBER_MIN, NUMBER_MAX + 1)],
            }
        )

        probability_table["Waga_łączna"] = (
            0.40 * probability_table["Waga_antybłąd"]
            + 0.25 * probability_table["Waga_świeża"]
            + 0.20 * probability_table["Waga_globalna"]
            + 0.15 * probability_table["Waga_opóźnienia"]
        )

        probability_table = probability_table.sort_values("Waga_łączna", ascending=False).reset_index(drop=True)

        latest = tuple(int(x) for x in self.df.loc[0, self.columns].tolist())

        return PreparedModel(
            frequency_table=global_table,
            probability_table=probability_table,
            global_weights=global_weights,
            rolling_weights=rolling_weights,
            delay_weights=delay_weights,
            anti_error_weights=anti_error_weights,
            pair_matrix_raw=pair_raw,
            pair_matrix_norm=pair_norm,
            latest_draw=latest,
            ranked_numbers=probability_table["Liczba"].astype(int).tolist(),
        )

    def accept_ticket(self, ticket: Sequence[int], settings: GeneratorSettings, latest_draw: Sequence[int]) -> bool:
        nums = sorted(map(int, ticket))

        if len(nums) != DRAW_SIZE or len(set(nums)) != DRAW_SIZE:
            return False

        if not all(NUMBER_MIN <= n <= NUMBER_MAX for n in nums):
            return False

        total = sum(nums)

        if total < settings.sum_min or total > settings.sum_max:
            return False

        even = sum(n % 2 == 0 for n in nums)
        low = sum(n <= 24 for n in nums)

        if even < settings.even_min or even > settings.even_max:
            return False

        if low < settings.low_min or low > settings.low_max:
            return False

        if self.consecutive_chain(nums) > settings.max_chain:
            return False

        if max(self.sectors(nums)) > settings.max_one_sector:
            return False

        if len(set(nums).intersection(set(latest_draw))) > settings.max_from_latest:
            return False

        return True

    def quality_score(self, ticket: Sequence[int], model: PreparedModel, settings: GeneratorSettings) -> Dict[str, Any]:
        nums = sorted(map(int, ticket))
        idx = np.array([n - NUMBER_MIN for n in nums], dtype=int)

        avg_anti = float(model.anti_error_weights[idx].mean())
        avg_global = float(model.global_weights[idx].mean())
        avg_rolling = float(model.rolling_weights[idx].mean())
        avg_delay = float(model.delay_weights[idx].mean())

        pair_score = 0.0
        if len(idx) >= 2:
            pair_score = float(model.pair_matrix_raw[np.ix_(idx, idx)].sum() / 2.0)

        even = sum(n % 2 == 0 for n in nums)
        odd = DRAW_SIZE - even
        low = sum(n <= 24 for n in nums)
        high = DRAW_SIZE - low
        sectors = self.sectors(nums)
        chain = self.consecutive_chain(nums)

        sum_center = (settings.sum_min + settings.sum_max) / 2
        sum_width = max(1, (settings.sum_max - settings.sum_min) / 2)
        sum_score = max(0.0, 1.0 - abs(sum(nums) - sum_center) / sum_width)

        balance = 1.0
        balance -= abs(even - odd) * 0.06
        balance -= abs(low - high) * 0.05
        balance -= max(0, max(sectors) - 2) * 0.10
        balance -= max(0, chain - 1) * 0.08
        balance = max(0.0, balance)

        history = self.historical_hit_profile(nums)

        quality = (
            avg_anti * 2400
            + avg_global * 900
            + avg_rolling * 900
            + avg_delay * 600
            + min(pair_score, 20) * 0.55
            + sum_score * 18
            + balance * 22
            + float(history["Wynik_historyczny"]) * 0.25
        )

        return {
            "Jakość": round(float(quality), 2),
            "Suma": int(sum(nums)),
            "Parzyste": int(even),
            "Nieparzyste": int(odd),
            "Niskie": int(low),
            "Wysokie": int(high),
            "Sektory": "-".join(str(x) for x in sectors),
            "Łańcuch": int(chain),
            "Para_bonus": int(pair_score),
            "Średnia_waga_antybłąd": round(avg_anti, 8),
            **history,
        }



# ============================================================
# STRAŻNIK BŁĘDÓW AI + RYZYKO + ANTY-POWTÓRKA 999
# ============================================================

class ErrorKillerAI:
    """
    Strażnik Błędów AI.
    Nie próbuje przewidzieć przyszłości.
    Jego zadanie: eliminować układy, które są słabe jakościowo,
    zbyt skrajne albo zbyt podobne do błędnych schematów.
    """

    @staticmethod
    def risk_profile_settings(profile: str) -> Dict[str, Any]:
        """
        Profile ryzyka:
        - Bezpieczny: mocna kontrola jakości, mniej skrajności.
        - Zrównoważony: najlepszy kompromis.
        - Agresywny: większa zmienność, ale nadal z ochroną przed głupimi układami.
        """
        if profile == "🟢 Bezpieczny":
            return {
                "sum_min": 120,
                "sum_max": 175,
                "even_min": 2,
                "even_max": 4,
                "low_min": 2,
                "low_max": 4,
                "max_chain": 2,
                "max_one_sector": 2,
                "max_from_latest": 1,
                "max_similarity_999": 3,
                "candidate_multiplier": 1.20,
                "quality_floor": 72,
            }

        if profile == "🔴 Agresywny":
            return {
                "sum_min": 95,
                "sum_max": 205,
                "even_min": 1,
                "even_max": 5,
                "low_min": 1,
                "low_max": 5,
                "max_chain": 3,
                "max_one_sector": 3,
                "max_from_latest": 3,
                "max_similarity_999": 4,
                "candidate_multiplier": 1.60,
                "quality_floor": 60,
            }

        return {
            "sum_min": 110,
            "sum_max": 190,
            "even_min": 2,
            "even_max": 4,
            "low_min": 2,
            "low_max": 4,
            "max_chain": 2,
            "max_one_sector": 2,
            "max_from_latest": 2,
            "max_similarity_999": 3,
            "candidate_multiplier": 1.35,
            "quality_floor": 66,
        }

    @staticmethod
    def max_historical_similarity(ticket: Sequence[int], historical_draws: Sequence[Sequence[int]]) -> int:
        ticket_set = set(map(int, ticket))
        best = 0

        for draw in historical_draws:
            best = max(best, len(ticket_set.intersection(set(map(int, draw)))))

        return int(best)

    @staticmethod
    def count_hot_cold(ticket: Sequence[int], frequency_table: pd.DataFrame) -> Tuple[int, int]:
        state_map = frequency_table.set_index("Liczba")["Stan"].to_dict()
        hot = sum(state_map.get(int(n), "Neutralna") == "Gorąca" for n in ticket)
        cold = sum(state_map.get(int(n), "Neutralna") == "Zimna" for n in ticket)
        return int(hot), int(cold)

    @staticmethod
    def human_pattern_penalty(ticket: Sequence[int]) -> float:
        """
        Kara za układy, które ludzie często wybierają:
        - same końcówki,
        - za dużo liczb z jednej dziesiątki,
        - schodki,
        - zbyt estetyczne układy.
        """
        nums = sorted(map(int, ticket))
        penalty = 0.0

        last_digits = [n % 10 for n in nums]
        if len(set(last_digits)) <= 3:
            penalty += 8.0

        decades = [
            sum(1 <= n <= 9 for n in nums),
            sum(10 <= n <= 19 for n in nums),
            sum(20 <= n <= 29 for n in nums),
            sum(30 <= n <= 39 for n in nums),
            sum(40 <= n <= 49 for n in nums),
        ]
        if max(decades) >= 4:
            penalty += 10.0

        diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
        if len(set(diffs)) <= 2:
            penalty += 6.0

        if all(n <= 31 for n in nums):
            penalty += 5.0

        if sum(1 for n in nums if n in {7, 13, 21, 22, 23, 24, 31, 37, 42, 49}) >= 4:
            penalty += 5.0

        return float(penalty)

    def inspect(
        self,
        ticket: Sequence[int],
        analytics: "LottoAnalytics",
        model: PreparedModel,
        settings: GeneratorSettings,
        risk_profile: str,
    ) -> Dict[str, Any]:
        nums = sorted(map(int, ticket))
        historical_draws = [tuple(map(int, row)) for row in analytics.df[analytics.columns].to_numpy()]
        similarity = self.max_historical_similarity(nums, historical_draws)
        hot, cold = self.count_hot_cold(nums, model.frequency_table)
        sectors = analytics.sectors(nums)
        chain = analytics.consecutive_chain(nums)
        even = sum(n % 2 == 0 for n in nums)
        low = sum(n <= 24 for n in nums)
        total = sum(nums)
        latest_overlap = len(set(nums).intersection(set(model.latest_draw)))
        profile = self.risk_profile_settings(risk_profile)

        reasons: List[str] = []
        passed = True

        if total < profile["sum_min"] or total > profile["sum_max"]:
            passed = False
            reasons.append(f"suma poza profilem ryzyka ({total})")

        if even < profile["even_min"] or even > profile["even_max"]:
            passed = False
            reasons.append(f"zły balans parzystych ({even})")

        if low < profile["low_min"] or low > profile["low_max"]:
            passed = False
            reasons.append(f"zły balans niskich ({low})")

        if chain > profile["max_chain"]:
            passed = False
            reasons.append(f"za długi ciąg kolejnych liczb ({chain})")

        if max(sectors) > profile["max_one_sector"]:
            passed = False
            reasons.append(f"za dużo liczb z jednego sektora ({max(sectors)})")

        if latest_overlap > profile["max_from_latest"]:
            passed = False
            reasons.append(f"za dużo powtórek z ostatniego losowania ({latest_overlap})")

        if similarity > profile["max_similarity_999"]:
            passed = False
            reasons.append(f"zbyt podobny do jednego z 999 losowań ({similarity}/6)")

        if hot >= 5:
            passed = False
            reasons.append("za dużo liczb gorących")

        if cold >= 5:
            passed = False
            reasons.append("za dużo liczb zimnych")

        if not reasons:
            reasons.append("kupon przeszedł kontrolę Strażnika Błędów AI")

        human_penalty = self.human_pattern_penalty(nums)

        return {
            "passed": bool(passed),
            "reasons": reasons,
            "max_similarity_999": int(similarity),
            "hot_count": int(hot),
            "cold_count": int(cold),
            "human_pattern_penalty": round(float(human_penalty), 2),
            "risk_profile": risk_profile,
        }


class FinalQualityIndex:
    """
    Inteligentny wskaźnik jakości 0–100.
    Nie oznacza gwarancji trafienia.
    Oznacza jakość kuponu według filtrów, balansu, historii i Strażnika Błędów AI.
    """

    @staticmethod
    def calculate(
        meta: Dict[str, Any],
        guard: Dict[str, Any],
        settings: GeneratorSettings,
    ) -> float:
        base = 58.0

        # Jakość modelu.
        quality = float(meta.get("Jakość", 0))
        base += min(18.0, quality / 6.0)

        # Historia.
        base += min(10.0, float(meta.get("Wynik_historyczny", 0)) / 10.0)
        base += min(4.0, float(meta.get("Trafienia_3+", 0)) / 40.0)

        # Balans sumy.
        total = int(meta.get("Suma", 0))
        center = (settings.sum_min + settings.sum_max) / 2
        width = max(1, (settings.sum_max - settings.sum_min) / 2)
        sum_balance = max(0.0, 1.0 - abs(total - center) / width)
        base += sum_balance * 8.0

        # Kary Strażnika.
        if not guard.get("passed", False):
            base -= 14.0

        base -= float(guard.get("human_pattern_penalty", 0)) * 0.7
        base -= max(0, int(guard.get("max_similarity_999", 0)) - 3) * 7.0
        base -= max(0, int(guard.get("hot_count", 0)) - 4) * 5.0
        base -= max(0, int(guard.get("cold_count", 0)) - 4) * 5.0

        return round(float(max(0.0, min(100.0, base))), 2)

    @staticmethod
    def label(value: float) -> str:
        if value >= 90:
            return "wybitny"
        if value >= 80:
            return "bardzo mocny"
        if value >= 70:
            return "mocny"
        if value >= 60:
            return "dobry"
        if value >= 50:
            return "ryzykowny"
        return "odradzany"


# ============================================================
# GENERATOR
# ============================================================

class LottoGenerator:
    def __init__(self, analytics: LottoAnalytics):
        self.analytics = analytics

    @staticmethod
    def weighted_sample(numbers: Sequence[int], weights: np.ndarray, size: int, rng: np.random.Generator) -> Tuple[int, ...]:
        available = np.array(list(numbers), dtype=int)
        local_weights = np.array([max(0.000001, weights[n - NUMBER_MIN]) for n in available], dtype=float)
        local_weights /= local_weights.sum()
        selected = rng.choice(available, size=size, replace=False, p=local_weights)
        return tuple(sorted(map(int, selected.tolist())))

    def _weights_for_style(self, model: PreparedModel, style: str) -> np.ndarray:
        uniform = np.ones(NUMBER_MAX) / NUMBER_MAX

        if "Bezpieczny balans" in style:
            weights = 0.65 * uniform + 0.35 * model.anti_error_weights
        elif "Kontrtrend" in style:
            weights = 0.42 * model.delay_weights + 0.30 * uniform + 0.28 * model.anti_error_weights
        elif "Elitarny" in style:
            weights = 0.45 * model.anti_error_weights + 0.25 * model.rolling_weights + 0.20 * model.global_weights + 0.10 * model.delay_weights
        elif "Gorące" in style:
            weights = 0.65 * model.global_weights + 0.35 * model.rolling_weights
        elif "Zimne" in style:
            weights = model.delay_weights
        elif "Hybryda" in style:
            weights = 0.35 * model.global_weights + 0.35 * model.rolling_weights + 0.30 * model.delay_weights
        else:
            weights = model.anti_error_weights

        weights /= weights.sum()
        return weights

    def generate(
        self,
        count: int,
        settings: GeneratorSettings,
        model: PreparedModel,
        style: str,
        risk_profile: str = "🟡 Zrównoważony",
        final_mode: bool = True,
    ) -> pd.DataFrame:
        rng = np.random.default_rng()
        all_numbers = list(range(NUMBER_MIN, NUMBER_MAX + 1))
        weights = self._weights_for_style(model, style)

        guard_engine = ErrorKillerAI()
        profile = guard_engine.risk_profile_settings(risk_profile)

        attempts = max(
            int(settings.candidate_attempts * profile["candidate_multiplier"]),
            count * 300,
            1000,
        )

        # W FINAL trybie generator robi dużą pulę kandydatów, a potem wybiera TOP.
        if final_mode:
            attempts = max(attempts, 1000)

        seen: set[Tuple[int, ...]] = set()
        candidates: List[Tuple[float, Tuple[int, ...], Dict[str, Any], Dict[str, Any]]] = []

        for _ in range(attempts):
            ticket = self.weighted_sample(all_numbers, weights, DRAW_SIZE, rng)

            if ticket in seen:
                continue

            seen.add(ticket)

            if not self.analytics.accept_ticket(ticket, settings, model.latest_draw):
                continue

            meta = self.analytics.quality_score(ticket, model, settings)
            guard = guard_engine.inspect(ticket, self.analytics, model, settings, risk_profile)
            final_quality = FinalQualityIndex.calculate(meta, guard, settings)

            # Strażnik nie musi zawsze odrzucać agresywne kupony, ale obniża ich jakość.
            # W profilu bezpiecznym i zrównoważonym pokazujemy tylko te, które przeszły kontrolę.
            if risk_profile in ("🟢 Bezpieczny", "🟡 Zrównoważony") and not guard["passed"]:
                continue

            if final_quality < profile["quality_floor"]:
                continue

            meta["Jakość_FINAL_0_100"] = final_quality
            meta["Ocena_FINAL"] = FinalQualityIndex.label(final_quality)
            meta["Profil_ryzyka"] = risk_profile
            meta["Podobieństwo_999"] = int(guard["max_similarity_999"])
            meta["Gorące_w_kuponie"] = int(guard["hot_count"])
            meta["Zimne_w_kuponie"] = int(guard["cold_count"])
            meta["Kara_wzoru_ludzkiego"] = float(guard["human_pattern_penalty"])
            meta["Strażnik_AI"] = "OK" if guard["passed"] else "RYZYKO"
            meta["Uwagi_strażnika"] = "; ".join(guard["reasons"])

            candidates.append((final_quality, ticket, meta, guard))

        if not candidates:
            # Awaryjnie: jeśli ustawienia są zbyt ciasne, generujemy kandydatów i wybieramy najlepszych bez twardego floor.
            for _ in range(max(1200, count * 250)):
                ticket = self.weighted_sample(all_numbers, weights, DRAW_SIZE, rng)

                if ticket in seen:
                    continue

                seen.add(ticket)
                meta = self.analytics.quality_score(ticket, model, settings)
                guard = guard_engine.inspect(ticket, self.analytics, model, settings, risk_profile)
                final_quality = FinalQualityIndex.calculate(meta, guard, settings)

                meta["Jakość_FINAL_0_100"] = final_quality
                meta["Ocena_FINAL"] = FinalQualityIndex.label(final_quality)
                meta["Profil_ryzyka"] = risk_profile
                meta["Podobieństwo_999"] = int(guard["max_similarity_999"])
                meta["Gorące_w_kuponie"] = int(guard["hot_count"])
                meta["Zimne_w_kuponie"] = int(guard["cold_count"])
                meta["Kara_wzoru_ludzkiego"] = float(guard["human_pattern_penalty"])
                meta["Strażnik_AI"] = "OK" if guard["passed"] else "RYZYKO"
                meta["Uwagi_strażnika"] = "; ".join(guard["reasons"])
                candidates.append((final_quality, ticket, meta, guard))

        candidates.sort(reverse=True, key=lambda x: (x[0], float(x[2].get("Jakość", 0))))

        rows: List[Dict[str, Any]] = []
        used: set[Tuple[int, ...]] = set()

        for _, ticket, meta, _ in candidates:
            if ticket in used:
                continue

            used.add(ticket)
            rows.append(
                {
                    "Moduł": style,
                    "Zestaw": " ".join(f"{n:02d}" for n in ticket),
                    **meta,
                }
            )

            if len(rows) >= count:
                break

        return pd.DataFrame(rows).reset_index(drop=True)

    def generate_ab(self, settings: GeneratorSettings, model: PreparedModel) -> pd.DataFrame:
        frames = [
            self.generate(1, settings, model, "A/B: Bezpieczny balans", "🟢 Bezpieczny"),
            self.generate(1, settings, model, "A/B: Kontrtrend", "🟡 Zrównoważony"),
            self.generate(1, settings, model, "A/B: Elitarny", "🔴 Agresywny"),
        ]
        result = pd.concat(frames, ignore_index=True)
        result.insert(0, "Wariant", ["A", "B", "C"][:len(result)])
        return result


# ============================================================
# PAMIĘĆ, LIGA I DNA GRACZA
# ============================================================

class LearningMemory:
    def __init__(self, backend: FirebaseBackend):
        self.backend = backend

    def save_generated(self, result: pd.DataFrame, settings: GeneratorSettings, source: str) -> int:
        rows = []

        for _, row in result.iterrows():
            item = {
                "created_at": now_string(),
                "module": str(row.get("Moduł", source)),
                "source": source,
                "ticket": str(row["Zestaw"]),
                "settings": asdict(settings),
                "evaluated": False,
                "quality": float(row.get("Jakość", 0)),
                "sum": int(row.get("Suma", 0)),
                "even": int(row.get("Parzyste", 0)),
                "low": int(row.get("Niskie", 0)),
                "sectors": str(row.get("Sektory", "")),
            }
            rows.append(item)

        if not rows:
            return 0

        saved_remote = 0

        if self.backend.enabled:
            for item in rows:
                if self.backend.add_document("generated_tickets", item):
                    saved_remote += 1

        if saved_remote == len(rows):
            return saved_remote

        # Fallback lokalny
        df_old = pd.read_csv(LOCAL_GENERATED_LOG) if LOCAL_GENERATED_LOG.exists() else pd.DataFrame()
        df_new = pd.concat([df_old, pd.DataFrame(rows)], ignore_index=True)
        df_new.to_csv(LOCAL_GENERATED_LOG, index=False, encoding="utf-8-sig")
        return len(rows)

    def load_generated(self) -> pd.DataFrame:
        if self.backend.enabled:
            df = self.backend.read_collection("generated_tickets")
            if not df.empty:
                return df

        if LOCAL_GENERATED_LOG.exists():
            return pd.read_csv(LOCAL_GENERATED_LOG)

        return pd.DataFrame()

    def load_evaluated(self) -> pd.DataFrame:
        if self.backend.enabled:
            df = self.backend.read_collection("evaluated_tickets")
            if not df.empty:
                return df

        if LOCAL_EVALUATED_LOG.exists():
            return pd.read_csv(LOCAL_EVALUATED_LOG)

        return pd.DataFrame()

    @staticmethod
    def ticket_from_text(text: str) -> Tuple[int, ...]:
        return parse_number_list(text)

    def evaluate_pending(self, draw_id: int, draw_numbers: Sequence[int]) -> int:
        generated = self.load_generated()

        if generated.empty:
            return 0

        draw_set = set(map(int, draw_numbers))
        rows = []

        for _, row in generated.iterrows():
            evaluated_flag = str(row.get("evaluated", "False")).lower() in ("true", "1", "yes")
            if evaluated_flag:
                continue

            ticket = self.ticket_from_text(str(row.get("ticket", row.get("Zestaw", ""))))
            if len(ticket) != DRAW_SIZE:
                continue

            hits = len(set(ticket).intersection(draw_set))
            settings_raw = row.get("settings", {})
            if isinstance(settings_raw, str):
                try:
                    settings_obj = json.loads(settings_raw.replace("'", '"'))
                except Exception:
                    settings_obj = {"raw": settings_raw}
            else:
                settings_obj = settings_raw

            item = {
                "evaluated_at": now_string(),
                "module": str(row.get("module", row.get("Moduł", "unknown"))),
                "source": str(row.get("source", "")),
                "ticket": " ".join(f"{n:02d}" for n in ticket),
                "draw_id": int(draw_id),
                "draw_numbers": " ".join(f"{n:02d}" for n in sorted(draw_set)),
                "hits": int(hits),
                "settings": settings_obj,
                "quality": float(row.get("quality", row.get("Jakość", 0)) or 0),
                "sum": int(row.get("sum", row.get("Suma", sum(ticket))) or sum(ticket)),
                "even": int(row.get("even", row.get("Parzyste", sum(n % 2 == 0 for n in ticket))) or 0),
                "low": int(row.get("low", row.get("Niskie", sum(n <= 24 for n in ticket))) or 0),
                "sectors": str(row.get("sectors", row.get("Sektory", ""))),
            }
            rows.append(item)

        if not rows:
            return 0

        saved_remote = 0

        if self.backend.enabled:
            for item in rows:
                if self.backend.add_document("evaluated_tickets", item):
                    saved_remote += 1

        if saved_remote != len(rows):
            old = pd.read_csv(LOCAL_EVALUATED_LOG) if LOCAL_EVALUATED_LOG.exists() else pd.DataFrame()
            new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
            new.to_csv(LOCAL_EVALUATED_LOG, index=False, encoding="utf-8-sig")

        return len(rows)

    def module_summary(self) -> pd.DataFrame:
        df = self.load_evaluated()

        if df.empty:
            return pd.DataFrame()

        df["hits"] = pd.to_numeric(df["hits"], errors="coerce").fillna(0).astype(int)

        grouped = (
            df.groupby("module")
            .agg(
                Próby=("hits", "count"),
                Średnia_trafień=("hits", "mean"),
                Max_trafień=("hits", "max"),
                Trafienia_2_plus=("hits", lambda x: int((x >= 2).sum())),
                Trafienia_3_plus=("hits", lambda x: int((x >= 3).sum())),
                Trafienia_4_plus=("hits", lambda x: int((x >= 4).sum())),
            )
            .reset_index()
        )

        grouped["Średnia_trafień"] = grouped["Średnia_trafień"].round(3)
        grouped["Skuteczność_2_plus_%"] = (grouped["Trafienia_2_plus"] / grouped["Próby"] * 100).round(2)
        grouped["Skuteczność_3_plus_%"] = (grouped["Trafienia_3_plus"] / grouped["Próby"] * 100).round(2)

        return grouped.sort_values(["Średnia_trafień", "Skuteczność_3_plus_%", "Max_trafień"], ascending=[False, False, False]).reset_index(drop=True)

    def dna_player(self) -> Dict[str, Any]:
        df = self.load_evaluated()

        if df.empty:
            return {
                "ready": False,
                "message": "Brak ocenionych kuponów. Zapisuj kupony i oceniaj je po losowaniu.",
            }

        df["hits"] = pd.to_numeric(df["hits"], errors="coerce").fillna(0).astype(int)
        df["sum"] = pd.to_numeric(df.get("sum", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)
        df["even"] = pd.to_numeric(df.get("even", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)
        df["low"] = pd.to_numeric(df.get("low", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)

        best_module = (
            df.groupby("module")["hits"]
            .agg(["count", "mean", "max"])
            .reset_index()
            .sort_values(["mean", "max", "count"], ascending=[False, False, False])
        )

        if best_module.empty:
            return {"ready": False, "message": "Za mało danych."}

        top = best_module.iloc[0]

        good = df[df["hits"] >= max(2, df["hits"].quantile(0.70))].copy()
        if good.empty:
            good = df.copy()

        sum_low = int(max(21, good["sum"].quantile(0.20)))
        sum_high = int(min(294, good["sum"].quantile(0.80)))
        even_mode = int(good["even"].mode().iloc[0]) if not good["even"].mode().empty else 3
        low_mode = int(good["low"].mode().iloc[0]) if not good["low"].mode().empty else 3

        recommendation_strength = "niska"
        if len(df) >= 50:
            recommendation_strength = "wysoka"
        elif len(df) >= 20:
            recommendation_strength = "średnia"

        return {
            "ready": True,
            "samples": int(len(df)),
            "best_module": str(top["module"]),
            "best_module_avg": round(float(top["mean"]), 3),
            "best_module_max": int(top["max"]),
            "recommended_sum_min": sum_low,
            "recommended_sum_max": sum_high,
            "recommended_even": even_mode,
            "recommended_low": low_mode,
            "confidence": recommendation_strength,
            "message": "DNA Gracza zostało zbudowane na podstawie ocenionych kuponów.",
        }



    def save_manual_check(
        self,
        module: str,
        ticket: Sequence[int],
        draw_id: int,
        draw_numbers: Sequence[int],
        settings_note: str = "",
    ) -> bool:
        """
        Ręczny zapis kuponu użytkownika.
        To jest serce Kroku 3: AI uczy się wyłącznie z kuponów,
        które użytkownik faktycznie chce sprawdzić/zapisać.
        """
        ticket_tuple = tuple(sorted(map(int, ticket)))
        draw_tuple = tuple(sorted(map(int, draw_numbers)))
        hits = len(set(ticket_tuple).intersection(set(draw_tuple)))

        even = sum(n % 2 == 0 for n in ticket_tuple)
        low = sum(n <= 24 for n in ticket_tuple)
        sectors = [
            sum(1 <= n <= 10 for n in ticket_tuple),
            sum(11 <= n <= 20 for n in ticket_tuple),
            sum(21 <= n <= 30 for n in ticket_tuple),
            sum(31 <= n <= 40 for n in ticket_tuple),
            sum(41 <= n <= 49 for n in ticket_tuple),
        ]

        item = {
            "evaluated_at": now_string(),
            "module": module,
            "source": "manual_check",
            "ticket": " ".join(f"{n:02d}" for n in ticket_tuple),
            "draw_id": int(draw_id),
            "draw_numbers": " ".join(f"{n:02d}" for n in draw_tuple),
            "hits": int(hits),
            "settings": {"note": settings_note, "source": "manual_check"},
            "quality": 0.0,
            "sum": int(sum(ticket_tuple)),
            "even": int(even),
            "low": int(low),
            "sectors": "-".join(str(x) for x in sectors),
        }

        if self.backend.enabled:
            return self.backend.add_document("evaluated_tickets", item)

        old = pd.read_csv(LOCAL_EVALUATED_LOG) if LOCAL_EVALUATED_LOG.exists() else pd.DataFrame()
        new = pd.concat([old, pd.DataFrame([item])], ignore_index=True)
        new.to_csv(LOCAL_EVALUATED_LOG, index=False, encoding="utf-8-sig")
        return True

    def effectiveness_center(self) -> Dict[str, Any]:
        """
        Centrum Skuteczności:
        podsumowanie całej historii ocenionych kuponów.
        """
        df = self.load_evaluated()

        if df.empty:
            return {
                "ready": False,
                "total": 0,
                "message": "Brak ocenionych kuponów.",
            }

        df["hits"] = pd.to_numeric(df["hits"], errors="coerce").fillna(0).astype(int)

        hit_counts = {f"{i}_trafień": int((df["hits"] == i).sum()) for i in range(0, 7)}
        total = int(len(df))
        avg = round(float(df["hits"].mean()), 3)
        max_hits = int(df["hits"].max())

        hit_2_plus = int((df["hits"] >= 2).sum())
        hit_3_plus = int((df["hits"] >= 3).sum())
        hit_4_plus = int((df["hits"] >= 4).sum())

        # Współczynnik opłacalności nie oznacza finansowej gwarancji.
        # To syntetyczna miara jakości Twoich zapisanych strategii.
        profitability = (
            avg * 22
            + (hit_2_plus / max(1, total)) * 22
            + (hit_3_plus / max(1, total)) * 32
            + (hit_4_plus / max(1, total)) * 60
            + max_hits * 4
        )
        profitability = round(float(max(0, min(100, profitability))), 2)

        return {
            "ready": True,
            "total": total,
            "avg_hits": avg,
            "max_hits": max_hits,
            "hit_counts": hit_counts,
            "hit_2_plus": hit_2_plus,
            "hit_3_plus": hit_3_plus,
            "hit_4_plus": hit_4_plus,
            "profitability_index": profitability,
        }

    def autopilot_recommendation(self) -> Dict[str, Any]:
        """
        Autopilot AI v1:
        na podstawie ocenionych kuponów wybiera moduł i ustawienia,
        które najczęściej wypadały najlepiej.
        """
        df = self.load_evaluated()

        if df.empty:
            return {
                "ready": False,
                "message": "Autopilot potrzebuje ocenionych kuponów. Minimum praktyczne: 20, sensownie: 50+.",
            }

        df["hits"] = pd.to_numeric(df["hits"], errors="coerce").fillna(0).astype(int)
        df["sum"] = pd.to_numeric(df.get("sum", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)
        df["even"] = pd.to_numeric(df.get("even", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)
        df["low"] = pd.to_numeric(df.get("low", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)

        module_summary = (
            df.groupby("module")
            .agg(
                attempts=("hits", "count"),
                avg_hits=("hits", "mean"),
                max_hits=("hits", "max"),
                hit_2_plus=("hits", lambda x: int((x >= 2).sum())),
                hit_3_plus=("hits", lambda x: int((x >= 3).sum())),
            )
            .reset_index()
        )

        if module_summary.empty:
            return {
                "ready": False,
                "message": "Brak danych modułów.",
            }

        module_summary["score"] = (
            module_summary["avg_hits"] * 40
            + module_summary["hit_2_plus"] / module_summary["attempts"].clip(lower=1) * 25
            + module_summary["hit_3_plus"] / module_summary["attempts"].clip(lower=1) * 45
            + module_summary["max_hits"] * 4
        )

        module_summary = module_summary.sort_values("score", ascending=False).reset_index(drop=True)
        best = module_summary.iloc[0]

        # Dobre kupony definiujemy względnie, żeby działało także przy małej próbie.
        threshold = max(2, int(df["hits"].quantile(0.70)))
        good = df[df["hits"] >= threshold].copy()

        if good.empty:
            good = df.copy()

        sum_min = int(max(21, good["sum"].quantile(0.20)))
        sum_max = int(min(294, good["sum"].quantile(0.80)))
        even_mode = int(good["even"].mode().iloc[0]) if not good["even"].mode().empty else 3
        low_mode = int(good["low"].mode().iloc[0]) if not good["low"].mode().empty else 3

        samples = int(len(df))
        if samples >= 80:
            confidence = "wysoka"
        elif samples >= 35:
            confidence = "średnia"
        elif samples >= 10:
            confidence = "wstępna"
        else:
            confidence = "bardzo niska"

        weak_modules = module_summary.tail(min(3, len(module_summary))).copy()
        weak_list = weak_modules["module"].astype(str).tolist()

        return {
            "ready": True,
            "samples": samples,
            "confidence": confidence,
            "best_module": str(best["module"]),
            "best_score": round(float(best["score"]), 2),
            "best_avg_hits": round(float(best["avg_hits"]), 3),
            "best_max_hits": int(best["max_hits"]),
            "recommended_sum_min": sum_min,
            "recommended_sum_max": sum_max,
            "recommended_even": even_mode,
            "recommended_low": low_mode,
            "weak_modules": weak_list,
            "module_table": module_summary,
        }

# ============================================================
# UI
# ============================================================

def ticket_html(ticket: Sequence[int], title: str) -> str:
    selected = set(map(int, ticket))
    cells = []

    for n in range(NUMBER_MIN, NUMBER_MAX + 1):
        cls = "selected" if n in selected else ""
        cells.append(f'<div class="lotto-cell {cls}">{n:02d}</div>')

    return f"""
    <div class="lotto-wrapper">
        <div class="lotto-title">{title}</div>
        <div class="lotto-grid">
            {''.join(cells)}
        </div>
    </div>
    <style>
    .lotto-wrapper {{
        background: linear-gradient(145deg, #07111f, #111827);
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 18px;
        margin: 12px 0 24px 0;
        max-width: 620px;
        box-shadow: 0 18px 42px rgba(0,0,0,0.35);
    }}
    .lotto-title {{
        color: #f9fafb;
        font-weight: 900;
        font-size: 17px;
        margin-bottom: 14px;
    }}
    .lotto-grid {{
        display: grid;
        grid-template-columns: repeat(7, 44px);
        gap: 7px;
    }}
    .lotto-cell {{
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #1f2937;
        border: 1px solid #4b5563;
        color: #d1d5db;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 13px;
    }}
    .lotto-cell.selected {{
        background: radial-gradient(circle at 30% 30%, #fff7ed, #fbbf24 38%, #dc2626 100%);
        color: #111827;
        border: 2px solid #fde68a;
        box-shadow: 0 0 18px rgba(251, 191, 36, 0.95), 0 0 34px rgba(220, 38, 38, 0.55);
        transform: scale(1.08);
    }}
    </style>
    """


class LottoApp:
    def __init__(self):
        st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")
        self.backend = FirebaseBackend()
        self.df = self.load_database()
        self.analytics = LottoAnalytics(self.df)
        self.generator = LottoGenerator(self.analytics)
        self.memory = LearningMemory(self.backend)
        self.columns = [f"N{i}" for i in range(1, DRAW_SIZE + 1)]

    def load_database(self) -> pd.DataFrame:
        with st.sidebar:
            st.header("📄 Plik i parser")
            pdf_name = st.text_input(
                "Nazwa PDF",
                DEFAULT_PDF_NAME,
                help="Plik musi być w repozytorium/folderze aplikacji. Aplikacja próbuje znaleźć .pdf i .PDF bez względu na wielkość liter.",
            )

            if st.button("🔄 Wyczyść cache i odśwież", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

            st.header("☁️ Pamięć")
            st.write(self.backend.status_label())
            if not self.backend.enabled:
                st.caption(self.backend.error_message)
                st.caption("Aplikacja obsługuje format Secrets [firebase].")

            keep = st.checkbox("☕ Tryb czuwania Streamlit", value=False)
            if keep:
                minutes = st.slider("Odświeżaj co minut", 5, 30, 12)
                keep_awake_block(minutes)

        path = resolve_case_insensitive_file(pdf_name)
        sig = file_signature(path)

        try:
            return cached_load_pdf(str(path), sig)
        except Exception as exc:
            st.error(f"Błąd odczytu PDF: {exc}")
            st.stop()

    def render_header(self):
        st.title("🧠 Lotto 6/49 Anty-Błąd PRO AI FINAL")
        st.info(
            "To nie jest magiczny przewidywacz. To system uczący się na Twoich realnych wynikach: "
            "zapisuje kupony, ocenia trafienia, buduje DNA Gracza i rekomenduje strategie, które faktycznie działają najlepiej u Ciebie."
        )

        newest = self.df.iloc[0]
        oldest = self.df.iloc[-1]
        ok, warn = validate_database(self.df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Losowania", len(self.df))
        c2.metric("Najnowsze", int(newest["Losowanie"]))
        c3.metric("Najstarsze", int(oldest["Losowanie"]))
        c4.metric("Pamięć", self.backend.status_label())

        for msg in ok:
            st.success(msg)

        for msg in warn:
            st.warning(msg)

        with st.expander("Podgląd parsera", expanded=False):
            st.write("Najnowsze losowanie:")
            st.code(f'{int(newest["Losowanie"])}: ' + " ".join(f"{int(newest[col]):02d}" for col in self.columns))
            st.write("Najstarsze losowanie:")
            st.code(f'{int(oldest["Losowanie"])}: ' + " ".join(f"{int(oldest[col]):02d}" for col in self.columns))

    def settings_ui(self, dna: Optional[Dict[str, Any]] = None) -> GeneratorSettings:
        st.subheader("⚙️ Ustawienia jakości")
        st.caption("Dobry start: suma 120–180, parzyste 2–4, niskie 2–4, max 2 z sektora, max 2 z ostatniego losowania.")

        default_sum_min = 120
        default_sum_max = 180
        default_even_min = 2
        default_even_max = 4
        default_low_min = 2
        default_low_max = 4

        if dna and dna.get("ready"):
            default_sum_min = int(dna["recommended_sum_min"])
            default_sum_max = int(dna["recommended_sum_max"])
            even = int(dna["recommended_even"])
            low = int(dna["recommended_low"])
            default_even_min = max(0, even - 1)
            default_even_max = min(6, even + 1)
            default_low_min = max(0, low - 1)
            default_low_max = min(6, low + 1)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            count = st.slider("Liczba kuponów", 1, 20, 3)
            rolling_window = st.select_slider("Okno analizy", options=[80, 120, 160, 220, 300, 500], value=160)

        with c2:
            sum_min = st.number_input("Minimalna suma", 21, 294, default_sum_min)
            sum_max = st.number_input("Maksymalna suma", 21, 294, default_sum_max)

        with c3:
            even_min = st.slider("Min. parzystych", 0, 6, default_even_min)
            even_max = st.slider("Max. parzystych", 0, 6, default_even_max)

        with c4:
            low_min = st.slider("Min. niskich 1–24", 0, 6, default_low_min)
            low_max = st.slider("Max. niskich 1–24", 0, 6, default_low_max)

        c5, c6, c7 = st.columns(3)

        with c5:
            max_chain = st.slider("Max ciąg kolejnych liczb", 1, 4, 2)

        with c6:
            max_sector = st.slider("Max z jednego sektora", 1, 6, 2)

        with c7:
            max_latest = st.slider("Max z ostatniego losowania", 0, 6, 2)

        profile = st.selectbox("Profil pracy", ["Ekspres", "Szybki PRO", "Dokładny"], index=1)
        attempts = {"Ekspres": 800, "Szybki PRO": 1800, "Dokładny": 4000}[profile]

        risk_profile = st.selectbox(
            "Poziom ryzyka Strażnika AI",
            ["🟢 Bezpieczny", "🟡 Zrównoważony", "🔴 Agresywny"],
            index=1,
            help=(
                "Bezpieczny: najmocniejsza kontrola jakości. "
                "Zrównoważony: najlepszy kompromis. "
                "Agresywny: większa zmienność, ale nadal z kontrolą błędów."
            ),
        )

        st.session_state["risk_profile_final"] = risk_profile

        if sum_min > sum_max:
            sum_min, sum_max = sum_max, sum_min

        if even_min > even_max:
            even_min, even_max = even_max, even_min

        if low_min > low_max:
            low_min, low_max = low_max, low_min

        return GeneratorSettings(
            module_name="Anty-Błąd PRO AI",
            count=int(count),
            rolling_window=int(rolling_window),
            sum_min=int(sum_min),
            sum_max=int(sum_max),
            even_min=int(even_min),
            even_max=int(even_max),
            low_min=int(low_min),
            low_max=int(low_max),
            max_chain=int(max_chain),
            max_one_sector=int(max_sector),
            max_from_latest=int(max_latest),
            candidate_attempts=int(attempts),
            profile_name=profile,
        )

    def show_result(self, result: pd.DataFrame, settings: GeneratorSettings, source: str):
        if result.empty:
            st.error("Brak kuponu. Poluzuj filtry.")
            return

        st.subheader("✅ Kupony")
        st.dataframe(result, use_container_width=True, hide_index=True)

        if "Jakość_FINAL_0_100" in result.columns:
            best_final = float(result["Jakość_FINAL_0_100"].max())
            st.success(
                f"Najlepszy wskaźnik jakości FINAL: {best_final}/100 — "
                f"{FinalQualityIndex.label(best_final)}. "
                "To nie jest gwarancja trafienia, tylko ocena jakości kuponu po Strażniku AI."
            )

        text = "\n".join(f"{row['Moduł']}: {row['Zestaw']} | jakość={row['Jakość']} | suma={row['Suma']}" for _, row in result.iterrows())
        st.text_area("Kopiuj kupony", text, height=130)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ CSV", result.to_csv(index=False).encode("utf-8-sig"), "kupony_lotto_ai.csv", "text/csv", use_container_width=True)
        with c2:
            if st.button("💾 Zapisz do pamięci AI", use_container_width=True):
                saved = self.memory.save_generated(result, settings, source)
                st.success(f"Zapisano {saved} kuponów.")

        st.subheader("🎫 Blankiet")
        for _, row in result.iterrows():
            nums = parse_number_list(str(row["Zestaw"]))
            st.markdown(ticket_html(nums, f"{row['Moduł']}: {row['Zestaw']}"), unsafe_allow_html=True)

    def tab_generator(self):
        st.header("🛡️ Generator Anty-Błąd PRO AI")
        dna = self.memory.dna_player()

        if dna.get("ready"):
            st.success(
                f"DNA aktywne: najlepszy moduł: {dna['best_module']} | średnia: {dna['best_module_avg']} | pewność: {dna['confidence']}"
            )
        else:
            st.info(dna.get("message", "Brak DNA."))

        settings = self.settings_ui(dna if dna.get("ready") else None)
        model = self.analytics.build_model(settings.rolling_window)

        c1, c2, c3 = st.columns(3)
        with c1:
            run = st.button("🛡️ Anty-Błąd PRO", use_container_width=True, type="primary")
        with c2:
            elite = st.button("💎 Elitarny", use_container_width=True)
        with c3:
            ab = st.button("🧪 Test A/B", use_container_width=True)

        if run:
            result = self.generator.generate(settings.count, settings, model, "Anty-Błąd PRO", st.session_state.get("risk_profile_final", "🟡 Zrównoważony"))
            self.show_result(result, settings, "Anty-Błąd PRO")

        if elite:
            elite_settings = GeneratorSettings(**{**asdict(settings), "count": 1, "candidate_attempts": max(3500, settings.candidate_attempts * 2)})
            result = self.generator.generate(1, elite_settings, model, "Elitarny", st.session_state.get("risk_profile_final", "🟡 Zrównoważony"))
            self.show_result(result, elite_settings, "Elitarny")

        if ab:
            result = self.generator.generate_ab(settings, model)
            self.show_result(result, settings, "Test A/B")

    def tab_evaluate(self):
        st.header("📊 Ocena kuponów i Liga Modułów")
        st.write("Po losowaniu wpisz wynik i oceń wszystkie zapisane kupony. To jest paliwo dla DNA Gracza.")

        generated = self.memory.load_generated()
        evaluated = self.memory.load_evaluated()

        c1, c2, c3 = st.columns(3)
        c1.metric("Zapisane kupony", len(generated))
        c2.metric("Ocenione kupony", len(evaluated))
        if not generated.empty and "evaluated" in generated.columns:
            c3.metric("Do oceny", len(generated))
        else:
            c3.metric("Do oceny", len(generated))

        latest = self.df.iloc[0]
        default_draw = int(latest["Losowanie"])
        default_nums = " ".join(f"{int(latest[col]):02d}" for col in self.columns)

        e1, e2 = st.columns(2)
        with e1:
            draw_id = st.number_input("Numer losowania", min_value=1, value=default_draw)
        with e2:
            draw_text = st.text_input("Wynik losowania", value=default_nums)

        draw_nums = parse_number_list(draw_text)

        if len(draw_nums) != DRAW_SIZE:
            st.warning("Wpisz dokładnie 6 liczb.")
        else:
            if st.button("✅ Oceń zapisane kupony", use_container_width=True, type="primary"):
                count = self.memory.evaluate_pending(int(draw_id), draw_nums)
                st.success(f"Oceniono {count} kuponów.")

        st.subheader("🏆 Ranking modułów")
        summary = self.memory.module_summary()
        if summary.empty:
            st.info("Brak ocenionych kuponów.")
        else:
            st.dataframe(summary, use_container_width=True, hide_index=True)

        with st.expander("Zapisane kupony", expanded=False):
            st.dataframe(generated, use_container_width=True, hide_index=True)

        with st.expander("Ocenione kupony", expanded=False):
            st.dataframe(evaluated, use_container_width=True, hide_index=True)

    def tab_dna(self):
        st.header("🧠 DNA Gracza")
        dna = self.memory.dna_player()

        if not dna.get("ready"):
            st.warning(dna.get("message", "Brak danych."))
            st.markdown(
                """
                Żeby DNA zaczęło działać:
                1. Generuj kupony.
                2. Klikaj **Zapisz do pamięci AI**.
                3. Po losowaniu wpisuj wynik w zakładce **Ocena i Liga**.
                4. Po 20–50 ocenach aplikacja zacznie mieć sensowne rekomendacje.
                """
            )
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Próby", dna["samples"])
        c2.metric("Najlepszy moduł", dna["best_module"])
        c3.metric("Średnia modułu", dna["best_module_avg"])
        c4.metric("Pewność", dna["confidence"])

        st.subheader("🎯 Rekomendacja ustawień")
        st.write(f"Suma: **{dna['recommended_sum_min']}–{dna['recommended_sum_max']}**")
        st.write(f"Parzyste najczęściej w dobrych wynikach: **{dna['recommended_even']}**")
        st.write(f"Niskie 1–24 najczęściej w dobrych wynikach: **{dna['recommended_low']}**")

        if dna["samples"] < 20:
            st.info("To jeszcze wczesne DNA. Zbieraj dalej wyniki.")
        elif dna["samples"] < 50:
            st.success("DNA ma już średnią pewność. Można korzystać z rekomendacji.")
        else:
            st.success("DNA ma wysoką pewność. To już wartościowa baza decyzji.")

    def tab_experiments(self):
        st.header("🧪 Laboratorium eksperymentalne")
        st.warning("Te tryby nie są główną strategią. Używaj ich do testów, a skuteczność oceniaj w Lidze Modułów.")

        settings = self.settings_ui(None)
        model = self.analytics.build_model(settings.rolling_window)
        mode = st.selectbox("Tryb", ["Gorące", "Zimne", "Hybryda", "Losowe kontrolowane"])

        if st.button("🧪 Generuj eksperymentalnie", use_container_width=True):
            result = self.generator.generate(settings.count, settings, model, f"Eksperymentalny: {mode}", st.session_state.get("risk_profile_final", "🔴 Agresywny"))
            self.show_result(result, settings, f"Eksperymentalny: {mode}")


    def tab_manual_check(self):
        st.header("🎯 Sprawdź mój kupon")
        st.write(
            "Ten moduł jest najważniejszy dla uczenia AI. Wpisujesz kupon, który faktycznie zagrałeś "
            "albo chcesz sprawdzić, wybierasz moduł/strategię i wpisujesz wynik losowania. "
            "Dopiero takie dane budują prawdziwe DNA Gracza."
        )

        latest = self.df.iloc[0]
        default_draw = int(latest["Losowanie"])
        default_result = " ".join(f"{int(latest[col]):02d}" for col in self.columns)

        c1, c2 = st.columns(2)

        with c1:
            module = st.selectbox(
                "Z jakiej strategii pochodził kupon?",
                [
                    "Anty-Błąd PRO",
                    "Elitarny",
                    "A/B: Bezpieczny balans",
                    "A/B: Kontrtrend",
                    "A/B: Elitarny",
                    "Eksperymentalny: Gorące",
                    "Eksperymentalny: Zimne",
                    "Eksperymentalny: Hybryda",
                    "Ręczny własny kupon",
                    "Inne",
                ],
                help="To pozwala AI budować Ligę Modułów i oceniać, które strategie realnie trafiają najlepiej.",
            )
            ticket_text = st.text_input("Mój kupon", value="", help="Wpisz 6 liczb, np. 05 12 17 36 39 49.")
            note = st.text_input("Notatka do ustawień / strategii", value="", help="Opcjonalnie: np. suma 130-170, max 1 z ostatniego.")

        with c2:
            draw_id = st.number_input("Numer losowania", min_value=1, value=default_draw)
            draw_text = st.text_input("Wynik losowania", value=default_result)

        ticket = parse_number_list(ticket_text)
        draw_numbers = parse_number_list(draw_text)

        if ticket_text and len(ticket) != DRAW_SIZE:
            st.warning("Kupon musi mieć dokładnie 6 unikalnych liczb z zakresu 1–49.")

        if draw_text and len(draw_numbers) != DRAW_SIZE:
            st.warning("Wynik losowania musi mieć dokładnie 6 unikalnych liczb.")

        if len(ticket) == DRAW_SIZE and len(draw_numbers) == DRAW_SIZE:
            hits = len(set(ticket).intersection(set(draw_numbers)))
            st.metric("Trafienia", f"{hits}/6")
            st.markdown(ticket_html(ticket, "Twój kupon: " + " ".join(f"{n:02d}" for n in ticket)), unsafe_allow_html=True)

            if st.button("💾 Zapisz sprawdzenie do DNA Gracza", use_container_width=True, type="primary"):
                ok = self.memory.save_manual_check(
                    module=module,
                    ticket=ticket,
                    draw_id=int(draw_id),
                    draw_numbers=draw_numbers,
                    settings_note=note,
                )
                if ok:
                    st.success("Zapisano wynik do Firebase/DNA Gracza.")
                else:
                    st.error("Nie udało się zapisać wyniku.")

    def tab_autopilot(self):
        st.header("🤖 Autopilot AI")
        st.write(
            "Autopilot nie przewiduje przyszłości. Analizuje Twoje ocenione kupony i podpowiada, "
            "które strategie oraz ustawienia warto promować, a które ograniczać."
        )

        recommendation = self.memory.autopilot_recommendation()
        center = self.memory.effectiveness_center()

        if not recommendation.get("ready"):
            st.warning(recommendation.get("message", "Brak danych."))
            st.info("Zacznij od zapisywania kuponów i sprawdzania ich w zakładce 🎯 Sprawdź mój kupon.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Próby", recommendation["samples"])
        c2.metric("Pewność", recommendation["confidence"])
        c3.metric("Najlepszy moduł", recommendation["best_module"])
        c4.metric("Średnia trafień", recommendation["best_avg_hits"])

        st.subheader("🎯 Rekomendowane ustawienia")
        st.success(
            f"Najlepszy kierunek: **{recommendation['best_module']}** | "
            f"Suma: **{recommendation['recommended_sum_min']}–{recommendation['recommended_sum_max']}** | "
            f"Parzyste około: **{recommendation['recommended_even']}** | "
            f"Niskie 1–24 około: **{recommendation['recommended_low']}**"
        )

        weak = recommendation.get("weak_modules", [])
        if weak:
            st.warning("Moduły do ograniczenia według Twojej historii: " + ", ".join(weak))

        if center.get("ready"):
            st.subheader("📈 Centrum Skuteczności")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ocenione kupony", center["total"])
            m2.metric("Średnia trafień", center["avg_hits"])
            m3.metric("Najlepszy wynik", f"{center['max_hits']}/6")
            m4.metric("Współczynnik opłacalności", f"{center['profitability_index']}/100")

            hit_counts = center["hit_counts"]
            hit_df = pd.DataFrame(
                {
                    "Trafienia": list(hit_counts.keys()),
                    "Liczba kuponów": list(hit_counts.values()),
                }
            )
            st.bar_chart(hit_df.set_index("Trafienia"))

        with st.expander("Tabela modułów Autopilota", expanded=False):
            table = recommendation.get("module_table")
            if isinstance(table, pd.DataFrame):
                st.dataframe(table, use_container_width=True, hide_index=True)

        st.info(
            "Najlepsza praktyka: korzystaj z rekomendacji dopiero po minimum 20 ocenionych kuponach. "
            "Po 50+ próbach Autopilot zaczyna mieć dużo większy sens."
        )



    def tab_error_killer(self):
        st.header("🛡️ Strażnik Błędów AI")
        st.write(
            "Ten moduł sprawdza dowolny kupon i pokazuje, czy nie ma typowych błędów: "
            "zbyt skrajnej sumy, złego balansu, za dużego podobieństwa do 999 losowań, "
            "za wielu gorących/zimnych liczb albo wzorów wybieranych często przez ludzi."
        )

        settings = self.settings_ui(None)
        model = self.analytics.build_model(settings.rolling_window)
        guard = ErrorKillerAI()

        ticket_text = st.text_input("Wpisz kupon do kontroli", value="", help="Przykład: 05 12 17 36 39 49")
        ticket = parse_number_list(ticket_text)

        if ticket_text and len(ticket) != DRAW_SIZE:
            st.warning("Wpisz dokładnie 6 liczb.")
            return

        if len(ticket) == DRAW_SIZE:
            meta = self.analytics.quality_score(ticket, model, settings)
            info = guard.inspect(ticket, self.analytics, model, settings, st.session_state.get("risk_profile_final", "🟡 Zrównoważony"))
            final_q = FinalQualityIndex.calculate(meta, info, settings)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ocena FINAL", f"{final_q}/100")
            c2.metric("Etykieta", FinalQualityIndex.label(final_q))
            c3.metric("Podobieństwo 999", f"{info['max_similarity_999']}/6")
            c4.metric("Strażnik", "OK" if info["passed"] else "RYZYKO")

            if info["passed"]:
                st.success("Kupon przeszedł kontrolę Strażnika Błędów AI.")
            else:
                st.warning("Kupon wymaga ostrożności: " + "; ".join(info["reasons"]))

            st.markdown(ticket_html(ticket, "Kontrolowany kupon: " + " ".join(f"{n:02d}" for n in ticket)), unsafe_allow_html=True)
            st.json({**meta, **info})

    def tab_final_lab(self):
        st.header("🏆 Laboratorium FINAL TOP")
        st.write(
            "Generator tworzy dużą pulę kandydatów, przepuszcza je przez Strażnika Błędów AI, "
            "Anty-Powtórkę 999 i wskaźnik jakości 0–100, a potem pokazuje wyłącznie TOP selekcję."
        )

        settings = self.settings_ui(self.memory.dna_player() if self.memory.dna_player().get("ready") else None)
        model = self.analytics.build_model(settings.rolling_window)

        top_count = st.slider("Ile kuponów TOP pokazać?", 1, 10, 3)
        style = st.selectbox(
            "Styl bazowy",
            ["Anty-Błąd PRO", "Elitarny", "A/B: Bezpieczny balans", "A/B: Kontrtrend", "Eksperymentalny: Hybryda"],
            index=0,
        )

        if st.button("🏆 Generuj TOP FINAL", use_container_width=True, type="primary"):
            lab_settings = GeneratorSettings(
                **{
                    **asdict(settings),
                    "count": int(top_count),
                    "candidate_attempts": max(5000, settings.candidate_attempts * 3),
                }
            )
            result = self.generator.generate(
                int(top_count),
                lab_settings,
                model,
                style,
                st.session_state.get("risk_profile_final", "🟡 Zrównoważony"),
                final_mode=True,
            )
            self.show_result(result, lab_settings, "Laboratorium FINAL TOP")


    def tab_stats(self):
        st.header("📈 Statystyka bazy")
        window = st.select_slider("Okno", options=[80, 120, 160, 220, 300, 500, None], value=160)
        freq = self.analytics.frequency_table(window)

        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(freq, use_container_width=True, hide_index=True)
        with c2:
            st.bar_chart(freq.sort_values("Liczba").set_index("Liczba")["Wystąpienia"])

        display = self.df.copy()
        display["Liczby"] = display[self.columns].apply(lambda r: " ".join(f"{int(x):02d}" for x in r), axis=1)
        st.subheader("Archiwum")
        st.dataframe(display[["Losowanie", "Liczby", "Suma", "Parzyste", "Nieparzyste", "Niskie_1_24", "Wysokie_25_49"]], use_container_width=True, hide_index=True)

    def tab_guide(self):
        st.header("📘 Instrukcja")
        st.markdown(
            """
            ## Najlepszy sposób użycia

            1. Generuj kupony w trybie **Anty-Błąd PRO** albo **Test A/B**.
            2. Klikaj **Zapisz do pamięci AI**.
            3. Po losowaniu przejdź do **Ocena i Liga** albo **Sprawdź mój kupon**.
            4. Wpisz realny wynik i oceń kupony.
            5. Po 20–50 ocenionych kuponach zobacz **DNA Gracza** i **Autopilot AI**.
            6. Korzystaj z rekomendowanych ustawień DNA, ale dalej zapisuj wyniki.

            ## Co oznacza AI w tej aplikacji?

            Nie oznacza przewidywania przyszłości.

            Oznacza:
            - pamięć Twoich wyników,
            - ranking modułów,
            - naukę najlepszych ustawień,
            - eliminację słabych strategii.

            ## Rekomendowany start FINAL

            - zakładka: **🏆 FINAL TOP** albo **🛡️ Generator AI**,
            - profil ryzyka: **🟡 Zrównoważony**,
            - suma: **120–180**,
            - parzyste: **2–4**,
            - niskie: **2–4**,
            - max ciąg: **2**,
            - max z sektora: **2**,
            - max z ostatniego: **1–2**,
            - profil pracy: **Szybki PRO**.

            ## Co robi FINAL?

            - tworzy dużą pulę kandydatów,
            - odrzuca słabe układy przez Strażnika Błędów AI,
            - sprawdza podobieństwo do 999 losowań,
            - karze zbyt ludzkie wzory,
            - wybiera TOP kupony według jakości 0–100,
            - zapisuje wyniki w Firebase,
            - buduje DNA Gracza i Autopilota AI.
            """
        )

    def run(self):
        self.render_header()

        tabs = st.tabs(
            [
                "🛡️ Generator AI",
                "🏆 FINAL TOP",
                "🛡️ Strażnik AI",
                "🎯 Sprawdź mój kupon",
                "📊 Ocena i Liga",
                "🧠 DNA Gracza",
                "🤖 Autopilot AI",
                "🧪 Eksperymenty",
                "📈 Statystyka",
                "📘 Instrukcja",
            ]
        )

        with tabs[0]:
            self.tab_generator()
        with tabs[1]:
            self.tab_final_lab()
        with tabs[2]:
            self.tab_error_killer()
        with tabs[3]:
            self.tab_manual_check()
        with tabs[4]:
            self.tab_evaluate()
        with tabs[5]:
            self.tab_dna()
        with tabs[6]:
            self.tab_autopilot()
        with tabs[7]:
            self.tab_experiments()
        with tabs[8]:
            self.tab_stats()
        with tabs[9]:
            self.tab_guide()


if __name__ == "__main__":
    LottoApp().run()
