# generator_lotto_649_antyblad_pro.py
# Lotto 6/49 Laboratorium Anty-Błąd PRO
#
# Nowa filozofia aplikacji:
# - główny cel nie brzmi już: „przewidzieć następne losowanie”,
# - główny cel brzmi: „odrzucić słabe, skrajne i typowo ludzkie kupony”.
#
# Aplikacja nadal zawiera dawne moduły eksperymentalne, ale zostały przeniesione
# do osobnej kategorii i opisane jako mniej zalecane. Główna część aplikacji
# skupia się na jakości kuponu, balansie, dywersyfikacji i testach historycznych.
#
# Plik źródłowy:
# - Wyniki060626.PDF w tym samym folderze co aplikacja.
# - PDF ma stały format tabel, czcionek i układu.
# - Baza ma 999 losowań.
# - Najnowsze losowanie przykładowe: 7365 = 05 13 17 20 31 32.
# - Najstarsze losowanie przykładowe: 6367 = 10 12 20 32 35 37.
#
# Uruchomienie:
# pip install -r requirements_lotto_649_antyblad_pro.txt
# streamlit run generator_lotto_649_antyblad_pro.py

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader


APP_TITLE = "Lotto 6/49 Laboratorium Anty-Błąd PRO"
PDF_PATH = Path("./Wyniki060626.PDF")
HISTORY_FILE = Path("./historia_skutecznosci_lotto_649.csv")

NUMBER_MIN = 1
NUMBER_MAX = 49
DRAW_SIZE = 6
TICKET_SIZE = 6
POOL_SIZE = NUMBER_MAX - NUMBER_MIN + 1

DRAW_ID_MIN = 1000
DRAW_ID_MAX = 9999
EXPECTED_DRAWS = 999
EXPECTED_NEWEST_ID = 7365
EXPECTED_NEWEST_NUMBERS = (5, 13, 17, 20, 31, 32)
EXPECTED_OLDEST_ID = 6367
EXPECTED_OLDEST_NUMBERS = (10, 12, 20, 32, 35, 37)

SECTOR_RANGES = [
    (1, 8),
    (9, 16),
    (17, 24),
    (25, 32),
    (33, 40),
    (41, 49),
]
SECTOR_LABEL = "1-8 / 9-16 / 17-24 / 25-32 / 33-40 / 41-49"


@dataclass(frozen=True)
class ParsedDraw:
    draw_id: int
    numbers: Tuple[int, ...]


@dataclass
class NumberModel:
    probability_table: pd.DataFrame
    global_weight: np.ndarray
    recent_weight: np.ndarray
    delay_weight: np.ndarray
    hybrid_weight: np.ndarray
    resonance_weight: np.ndarray
    pair_matrix_raw: np.ndarray
    pair_matrix_norm: np.ndarray
    latest_draw: Tuple[int, ...]
    sum_mean: float
    sum_std: float
    draws_matrix: np.ndarray


# ============================================================
# PARSER PDF
# ============================================================

class LottoPdfParser:
    """
    Parser PDF dopasowany do układu Lotto 6/49 z map liczbowych.

    Oczekiwany układ strony:
    - najpierw linie z wynikami: 6 liczb dwucyfrowych, np. 05 13 17 20 31 32,
    - niżej/osobno linie z numerami losowań: 4 cyfry, np. 7365,
    - liczba wyników na stronie odpowiada liczbie numerów losowań,
    - kolejność jest od najnowszego do najstarszego.

    Parser jest odporny na sytuację, gdy pypdf skleja liczby, ponieważ czyści
    linię do cyfr i rozcina wynik na pary dwucyfrowe.
    """

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def exists(self) -> bool:
        return self.pdf_path.exists() and self.pdf_path.is_file()

    @staticmethod
    def file_signature(pdf_path: Path) -> str:
        if not pdf_path.exists():
            return "missing"

        stat = pdf_path.stat()
        h = hashlib.sha256()

        with pdf_path.open("rb") as file:
            h.update(file.read(1024 * 1024))

        return f"{stat.st_size}_{stat.st_mtime_ns}_{h.hexdigest()[:16]}"

    @staticmethod
    def _parse_result_line(line: str) -> Optional[Tuple[int, ...]]:
        digits = re.sub(r"\D", "", line)

        if len(digits) != DRAW_SIZE * 2:
            return None

        numbers = tuple(int(digits[i:i + 2]) for i in range(0, len(digits), 2))

        if (
            len(numbers) == DRAW_SIZE
            and len(set(numbers)) == DRAW_SIZE
            and all(NUMBER_MIN <= number <= NUMBER_MAX for number in numbers)
            and tuple(sorted(numbers)) == numbers
        ):
            return numbers

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
                parsed for parsed in (self._parse_result_line(line) for line in lines[:split_index])
                if parsed is not None
            ]
            draw_ids = [
                parsed for parsed in (self._parse_draw_id_line(line) for line in lines[split_index:])
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
        if not self.exists():
            raise FileNotFoundError(
                f'Nie znaleziono pliku "{self.pdf_path.name}". Umieść go w tym samym folderze co aplikacja.'
            )

        reader = PdfReader(str(self.pdf_path))
        all_draws: Dict[int, ParsedDraw] = {}

        for page in reader.pages:
            text = page.extract_text() or ""
            for draw in self.parse_page(text):
                all_draws[draw.draw_id] = draw

        if not all_draws:
            raise ValueError("Nie udało się odczytać poprawnych losowań z PDF.")

        rows: List[Dict[str, int]] = []
        for draw in all_draws.values():
            row: Dict[str, int] = {"Losowanie": int(draw.draw_id)}
            for index, number in enumerate(draw.numbers, start=1):
                row[f"N{index}"] = int(number)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Losowanie"])
        df = df.sort_values("Losowanie", ascending=False).reset_index(drop=True)

        number_columns = [f"N{i}" for i in range(1, DRAW_SIZE + 1)]
        df["Liczby"] = df[number_columns].apply(lambda row: tuple(int(x) for x in row), axis=1)
        df["Suma"] = df[number_columns].sum(axis=1)
        df["Parzyste"] = df[number_columns].apply(lambda row: sum(int(x) % 2 == 0 for x in row), axis=1)
        df["Nieparzyste"] = DRAW_SIZE - df["Parzyste"]
        df["Niskie_1_24"] = df[number_columns].apply(lambda row: sum(int(x) <= 24 for x in row), axis=1)
        df["Wysokie_25_49"] = DRAW_SIZE - df["Niskie_1_24"]

        for sector_index, (start, end) in enumerate(SECTOR_RANGES, start=1):
            df[f"Sektor_{sector_index}_{start}_{end}"] = df[number_columns].apply(
                lambda row, s=start, e=end: sum(s <= int(x) <= e for x in row),
                axis=1,
            )

        return df


@st.cache_data(show_spinner="Wczytywanie i parsowanie pliku PDF...")
def cached_load_lotto(pdf_path: str, signature: str) -> pd.DataFrame:
    return LottoPdfParser(Path(pdf_path)).parse()


def validate_database(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    ok: List[str] = []
    warnings: List[str] = []

    if len(df) == EXPECTED_DRAWS:
        ok.append("Odczytano dokładnie 999 losowań.")
    else:
        warnings.append(f"Odczytano {len(df)} losowań. Oczekiwano 999.")

    if df.empty:
        warnings.append("Baza jest pusta.")
        return ok, warnings

    ids = df["Losowanie"].astype(int).tolist()

    if all(ids[i] > ids[i + 1] for i in range(len(ids) - 1)):
        ok.append("Kolejność losowań jest poprawna: od najnowszego do najstarszego.")
    else:
        warnings.append("Numery losowań nie są idealnie malejące. Sprawdź PDF.")

    diffs = [ids[i] - ids[i + 1] for i in range(len(ids) - 1)]
    if diffs and all(diff == 1 for diff in diffs):
        ok.append("Numeracja losowań jest ciągła, bez luk.")
    elif diffs:
        warnings.append(f"Wykryto {sum(diff != 1 for diff in diffs)} przerw/skoków w numeracji.")

    newest = df.iloc[0]
    oldest = df.iloc[-1]
    newest_numbers = tuple(int(newest[f"N{i}"]) for i in range(1, DRAW_SIZE + 1))
    oldest_numbers = tuple(int(oldest[f"N{i}"]) for i in range(1, DRAW_SIZE + 1))

    if int(newest["Losowanie"]) == EXPECTED_NEWEST_ID and newest_numbers == EXPECTED_NEWEST_NUMBERS:
        ok.append("Kontrola najnowszego losowania zgodna ze wzorcem: 7365.")
    else:
        warnings.append(
            "Najnowsze losowanie różni się od wzorca 7365. Jeżeli PDF został zaktualizowany, to normalne."
        )

    if int(oldest["Losowanie"]) == EXPECTED_OLDEST_ID and oldest_numbers == EXPECTED_OLDEST_NUMBERS:
        ok.append("Kontrola najstarszego losowania zgodna ze wzorcem: 6367.")
    else:
        warnings.append(
            "Najstarsze losowanie różni się od wzorca 6367. Jeżeli PDF został zaktualizowany, to normalne."
        )

    bad_rows: List[int] = []
    number_columns = [f"N{i}" for i in range(1, DRAW_SIZE + 1)]
    for _, row in df.iterrows():
        nums = [int(row[col]) for col in number_columns]
        if (
            len(nums) != DRAW_SIZE
            or len(set(nums)) != DRAW_SIZE
            or nums != sorted(nums)
            or not all(NUMBER_MIN <= number <= NUMBER_MAX for number in nums)
        ):
            bad_rows.append(int(row["Losowanie"]))

    if not bad_rows:
        ok.append("Każde losowanie ma 6 unikalnych liczb z zakresu 1–49.")
    else:
        warnings.append(f"Wykryto błędne wiersze, np. {bad_rows[:5]}.")

    return ok, warnings


# ============================================================
# FUNKCJE POMOCNICZE I ANALITYKA
# ============================================================

def normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mn = float(values.min())
    mx = float(values.max())
    if mx == mn:
        return np.ones_like(values, dtype=np.float64) / len(values)
    out = (values - mn) / (mx - mn)
    out = out + 0.000001
    return out / out.sum()


def parse_number_list(text: str) -> List[int]:
    if not text.strip():
        return []
    nums = [int(x) for x in re.findall(r"\d+", text)]
    return sorted(set(number for number in nums if NUMBER_MIN <= number <= NUMBER_MAX))


def format_ticket(numbers: Sequence[int]) -> str:
    return " ".join(f"{int(number):02d}" for number in sorted(numbers))


def sectors(numbers: Sequence[int]) -> List[int]:
    nums = list(map(int, numbers))
    return [sum(start <= number <= end for number in nums) for start, end in SECTOR_RANGES]


def consecutive_chain(numbers: Sequence[int]) -> int:
    nums = sorted(map(int, numbers))
    if not nums:
        return 0
    longest = 1
    current = 1
    for idx in range(1, len(nums)):
        if nums[idx] == nums[idx - 1] + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def same_last_digit_max(numbers: Sequence[int]) -> int:
    endings = Counter(int(number) % 10 for number in numbers)
    return max(endings.values()) if endings else 0


def arithmetic_progression_score(numbers: Sequence[int]) -> int:
    nums = sorted(map(int, numbers))
    score = 0
    for triple in itertools.combinations(nums, 3):
        if triple[1] - triple[0] == triple[2] - triple[1]:
            score += 1
    return score


def ticket_overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return len(set(map(int, a)).intersection(set(map(int, b))))


class LottoAnalyticsEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("Losowanie", ascending=False).reset_index(drop=True)
        self.columns = [f"N{i}" for i in range(1, DRAW_SIZE + 1)]
        self.draws_matrix = self.df[self.columns].to_numpy(dtype=np.int16)

    def frequency_counter(self, window: Optional[int] = None) -> Counter[int]:
        source = self.df.head(window) if window else self.df
        values = source[self.columns].to_numpy().flatten()
        return Counter(map(int, values))

    def frequency_table(self, window: Optional[int] = None) -> pd.DataFrame:
        source = self.df.head(window) if window else self.df
        counts = self.frequency_counter(window=window)
        rows = []
        for number in range(NUMBER_MIN, NUMBER_MAX + 1):
            count = counts[number]
            rows.append(
                {
                    "Liczba": number,
                    "Wystąpienia": int(count),
                    "Procent_losowań": round(count / max(1, len(source)) * 100.0, 4),
                }
            )
        table = pd.DataFrame(rows)
        hot_border = table["Wystąpienia"].quantile(0.75)
        cold_border = table["Wystąpienia"].quantile(0.25)
        table["Stan"] = "Neutralna"
        table.loc[table["Wystąpienia"] >= hot_border, "Stan"] = "Gorąca"
        table.loc[table["Wystąpienia"] <= cold_border, "Stan"] = "Zimna"
        return table.sort_values(["Wystąpienia", "Liczba"], ascending=[False, True]).reset_index(drop=True)

    def delays(self) -> Dict[int, int]:
        rows = [set(map(int, row)) for row in self.draws_matrix]
        output: Dict[int, int] = {}
        for number in range(NUMBER_MIN, NUMBER_MAX + 1):
            delay = len(rows)
            for idx, row in enumerate(rows):
                if number in row:
                    delay = idx
                    break
            output[number] = int(delay)
        return output

    def pair_matrix(self, window: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        source = self.df.head(window) if window else self.df
        matrix = np.zeros((POOL_SIZE, POOL_SIZE), dtype=np.float64)
        for row in source[self.columns].to_numpy():
            nums = sorted(map(int, row))
            for a, b in itertools.combinations(nums, 2):
                ia = a - NUMBER_MIN
                ib = b - NUMBER_MIN
                matrix[ia, ib] += 1
                matrix[ib, ia] += 1
        mx = matrix.max()
        norm = matrix / mx if mx > 0 else matrix.copy()
        return matrix, norm

    def top_pairs(self, window: int, limit: int = 50) -> pd.DataFrame:
        raw, _ = self.pair_matrix(window=window)
        rows = []
        for a in range(NUMBER_MIN, NUMBER_MAX + 1):
            for b in range(a + 1, NUMBER_MAX + 1):
                count = int(raw[a - NUMBER_MIN, b - NUMBER_MIN])
                if count > 0:
                    rows.append({"Para": f"{a:02d}-{b:02d}", "Wystąpienia": count})
        if not rows:
            return pd.DataFrame(columns=["Para", "Wystąpienia"])
        return pd.DataFrame(rows).sort_values("Wystąpienia", ascending=False).head(limit).reset_index(drop=True)

    def top_triples(self, window: int, limit: int = 50) -> pd.DataFrame:
        source = self.df.head(window)
        counter: Counter[Tuple[int, int, int]] = Counter()
        for row in source[self.columns].to_numpy():
            for triple in itertools.combinations(sorted(map(int, row)), 3):
                counter[triple] += 1
        rows = [
            {"Trójka": f"{a:02d}-{b:02d}-{c:02d}", "Wystąpienia": count}
            for (a, b, c), count in counter.items()
        ]
        if not rows:
            return pd.DataFrame(columns=["Trójka", "Wystąpienia"])
        return pd.DataFrame(rows).sort_values("Wystąpienia", ascending=False).head(limit).reset_index(drop=True)

    def resonance_weight(self, window: int = 220) -> np.ndarray:
        """
        Eksperymentalny model podobieństwa.
        Dla każdego historycznego losowania liczy podobieństwo do najnowszego wyniku,
        a potem wzmacnia liczby z losowania następnego po podobnym stanie.
        """
        latest = set(map(int, self.draws_matrix[0]))
        sample = self.draws_matrix[: min(window, len(self.draws_matrix))]
        scores = np.ones(POOL_SIZE, dtype=np.float64)

        if len(sample) < 3:
            return scores / scores.sum()

        # Dane są od najnowszego do najstarszego. Historyczny "następnik" dla starszego wiersza i
        # to wiersz o indeksie i - 1.
        for i in range(1, len(sample)):
            current = set(map(int, sample[i]))
            newer_after_current = set(map(int, sample[i - 1]))
            overlap = len(latest.intersection(current))
            sector_similarity = 0
            latest_sectors = sectors(latest)
            current_sectors = sectors(current)
            for a, b in zip(latest_sectors, current_sectors):
                sector_similarity += max(0, 2 - abs(a - b))
            sim = overlap * 1.7 + sector_similarity * 0.35
            recency = 1.0 / math.sqrt(i + 1)
            for number in newer_after_current:
                scores[number - NUMBER_MIN] += sim * recency

        return scores / scores.sum()

    def prepare_model(self, recent_window: int = 160) -> NumberModel:
        global_counter = self.frequency_counter(window=None)
        recent_counter = self.frequency_counter(window=recent_window)
        delays = self.delays()

        global_counts = np.array([global_counter[n] for n in range(NUMBER_MIN, NUMBER_MAX + 1)], dtype=np.float64)
        recent_counts = np.array([recent_counter[n] for n in range(NUMBER_MIN, NUMBER_MAX + 1)], dtype=np.float64)
        delay_values = np.array([delays[n] for n in range(NUMBER_MIN, NUMBER_MAX + 1)], dtype=np.float64)

        global_weight = normalize_array(global_counts + 1)
        recent_weight = normalize_array(recent_counts + 1)
        delay_weight = normalize_array(delay_values + 1)
        resonance = self.resonance_weight(window=recent_window)
        hybrid = 0.34 * global_weight + 0.30 * recent_weight + 0.18 * delay_weight + 0.18 * resonance
        hybrid = hybrid / hybrid.sum()

        pair_raw, pair_norm = self.pair_matrix(window=recent_window)
        sums = self.df["Suma"].to_numpy(dtype=np.float64)
        sum_mean = float(sums.mean())
        sum_std = float(sums.std(ddof=0)) if float(sums.std(ddof=0)) > 0 else 18.0

        probability_table = pd.DataFrame(
            {
                "Liczba": list(range(NUMBER_MIN, NUMBER_MAX + 1)),
                "Global": np.round(global_weight, 8),
                "Ostatnie_okno": np.round(recent_weight, 8),
                "Opóźnienie": [delays[n] for n in range(NUMBER_MIN, NUMBER_MAX + 1)],
                "Waga_opóźnienia": np.round(delay_weight, 8),
                "Rezonans": np.round(resonance, 8),
                "Hybryda": np.round(hybrid, 8),
                "Wystąpienia_globalnie": global_counts.astype(int),
                "Wystąpienia_ostatnie": recent_counts.astype(int),
            }
        ).sort_values("Hybryda", ascending=False).reset_index(drop=True)

        latest_draw = tuple(int(x) for x in self.draws_matrix[0])

        return NumberModel(
            probability_table=probability_table,
            global_weight=global_weight,
            recent_weight=recent_weight,
            delay_weight=delay_weight,
            hybrid_weight=hybrid,
            resonance_weight=resonance,
            pair_matrix_raw=pair_raw,
            pair_matrix_norm=pair_norm,
            latest_draw=latest_draw,
            sum_mean=sum_mean,
            sum_std=sum_std,
            draws_matrix=self.draws_matrix,
        )


# ============================================================
# OCENA KUPONU I STRAŻNIK JAKOŚCI
# ============================================================

class QualityGuard:
    def __init__(self, model: NumberModel):
        self.model = model

    def historical_hit_profile(self, ticket: Sequence[int]) -> Dict[str, float | int]:
        ticket_set = set(map(int, ticket))
        hits = []
        for row in self.model.draws_matrix:
            hits.append(len(ticket_set.intersection(set(map(int, row)))))
        arr = np.array(hits, dtype=np.int16)
        if len(arr) == 0:
            return {
                "Średnia_trafień": 0.0,
                "Max_trafień": 0,
                "Trafienia_2+": 0,
                "Trafienia_3+": 0,
                "Trafienia_4+": 0,
                "Wynik_historyczny": 0.0,
            }
        hit_2 = int(np.sum(arr >= 2))
        hit_3 = int(np.sum(arr >= 3))
        hit_4 = int(np.sum(arr >= 4))
        avg_hit = float(np.mean(arr))
        max_hit = int(np.max(arr))
        score = avg_hit * 16.0 + hit_2 * 0.045 + hit_3 * 0.35 + hit_4 * 2.25 + max_hit * 4.0
        return {
            "Średnia_trafień": round(avg_hit, 4),
            "Max_trafień": max_hit,
            "Trafienia_2+": hit_2,
            "Trafienia_3+": hit_3,
            "Trafienia_4+": hit_4,
            "Wynik_historyczny": round(float(score), 4),
        }

    def accept(
        self,
        ticket: Sequence[int],
        min_sum: Optional[int],
        max_sum: Optional[int],
        max_chain: int,
        max_one_sector: int,
        max_from_latest: int,
        max_same_last_digit: int,
        max_arithmetic_triples: int,
    ) -> bool:
        nums = sorted(map(int, ticket))
        if len(nums) != TICKET_SIZE:
            return False
        if len(set(nums)) != TICKET_SIZE:
            return False
        if not all(NUMBER_MIN <= n <= NUMBER_MAX for n in nums):
            return False
        total = sum(nums)
        if min_sum is not None and total < min_sum:
            return False
        if max_sum is not None and total > max_sum:
            return False
        if consecutive_chain(nums) > max_chain:
            return False
        if max(sectors(nums)) > max_one_sector:
            return False
        if ticket_overlap(nums, self.model.latest_draw) > max_from_latest:
            return False
        if same_last_digit_max(nums) > max_same_last_digit:
            return False
        if arithmetic_progression_score(nums) > max_arithmetic_triples:
            return False
        return True

    def score(self, ticket: Sequence[int]) -> Dict[str, float | int | str]:
        nums = sorted(map(int, ticket))
        idx = np.array([n - NUMBER_MIN for n in nums], dtype=int)
        total = int(sum(nums))
        even = int(sum(n % 2 == 0 for n in nums))
        odd = TICKET_SIZE - even
        low = int(sum(n <= 24 for n in nums))
        high = TICKET_SIZE - low
        sec = sectors(nums)
        chain = consecutive_chain(nums)
        last_digit_max = same_last_digit_max(nums)
        arithmetic_triples = arithmetic_progression_score(nums)
        latest_overlap = ticket_overlap(nums, self.model.latest_draw)

        sum_z = abs((total - self.model.sum_mean) / max(1.0, self.model.sum_std))
        sum_score = math.exp(-0.5 * sum_z * sum_z)
        parity_score = 1.0 - min(abs(even - 3) / 3.0, 1.0)
        low_high_score = 1.0 - min(abs(low - 3) / 3.0, 1.0)
        sector_score = 1.0 - min(max(0, max(sec) - 2) / 4.0, 1.0)
        chain_score = 1.0 - min(max(0, chain - 2) / 4.0, 1.0)
        ending_score = 1.0 - min(max(0, last_digit_max - 2) / 4.0, 1.0)
        arithmetic_score = 1.0 - min(arithmetic_triples / 5.0, 1.0)
        latest_score = 1.0 - min(max(0, latest_overlap - 1) / 5.0, 1.0)

        model_weight = float(np.mean(self.model.hybrid_weight[idx]))
        pair_bonus = float(self.model.pair_matrix_raw[np.ix_(idx, idx)].sum() / 2.0) if len(idx) >= 2 else 0.0
        pair_score = min(pair_bonus / 35.0, 1.0)
        history = self.historical_hit_profile(nums)
        hist_score = min(float(history["Wynik_historyczny"]) / 180.0, 1.0)

        # Główna filozofia: filtry jakości mają większą wagę niż próba przepowiedni.
        quality = (
            sum_score * 18.0
            + parity_score * 14.0
            + low_high_score * 14.0
            + sector_score * 16.0
            + chain_score * 10.0
            + ending_score * 8.0
            + arithmetic_score * 7.0
            + latest_score * 7.0
            + pair_score * 3.0
            + hist_score * 3.0
            + model_weight * 180.0
        )

        risk_flags: List[str] = []
        if total < 95 or total > 205:
            risk_flags.append("skrajna suma")
        if even in (0, 1, 5, 6):
            risk_flags.append("skrajny parytet")
        if low in (0, 1, 5, 6):
            risk_flags.append("skrajny niski/wysoki")
        if max(sec) >= 4:
            risk_flags.append("skupienie sektora")
        if chain >= 3:
            risk_flags.append("ciąg kolejnych")
        if last_digit_max >= 4:
            risk_flags.append("końcówki")
        if latest_overlap >= 3:
            risk_flags.append("dużo z ostatniego")

        return {
            "Jakość": round(float(quality), 2),
            "Suma": total,
            "Parzyste": even,
            "Nieparzyste": odd,
            "Niskie_1_24": low,
            "Wysokie_25_49": high,
            "Sektory": "-".join(str(x) for x in sec),
            "Łańcuch": chain,
            "Max_końcówka": last_digit_max,
            "Trójki_arytm.": arithmetic_triples,
            "Z_ostatniego": latest_overlap,
            "Para_bonus": round(pair_bonus, 2),
            "Średnia_waga_modelu": round(model_weight, 8),
            "Ryzyka": ", ".join(risk_flags) if risk_flags else "brak",
            **history,
        }


# ============================================================
# GENERATORY
# ============================================================

class LottoGenerator:
    def __init__(self, engine: LottoAnalyticsEngine, model: NumberModel):
        self.engine = engine
        self.model = model
        self.guard = QualityGuard(model)

    @staticmethod
    def weighted_sample(candidates: Sequence[int], weights: np.ndarray, k: int, rng: np.random.Generator) -> Tuple[int, ...]:
        candidates = sorted(set(map(int, candidates)))
        if k <= 0:
            return tuple()
        local_weights = np.array([max(0.000001, weights[n - NUMBER_MIN]) for n in candidates], dtype=np.float64)
        local_weights = local_weights / local_weights.sum()
        chosen = rng.choice(np.array(candidates, dtype=int), size=k, replace=False, p=local_weights)
        return tuple(sorted(map(int, chosen.tolist())))

    def generate_antyblad_candidates(
        self,
        candidate_count: int,
        min_sum: Optional[int],
        max_sum: Optional[int],
        max_chain: int,
        max_one_sector: int,
        max_from_latest: int,
        max_same_last_digit: int,
        max_arithmetic_triples: int,
        banned_numbers: Sequence[int],
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        banned = set(map(int, banned_numbers))
        available = [n for n in range(NUMBER_MIN, NUMBER_MAX + 1) if n not in banned]
        rows: List[Dict[str, object]] = []
        seen: set[Tuple[int, ...]] = set()
        attempts = max(candidate_count * 18, 800)

        # Celowo mieszamy losowanie równe i łagodnie ważone, ale bez agresywnej wiary w gorące/zimne.
        uniform = np.ones(POOL_SIZE, dtype=np.float64) / POOL_SIZE
        gentle = 0.80 * uniform + 0.20 * self.model.hybrid_weight
        gentle = gentle / gentle.sum()

        for _ in range(attempts):
            if len(rows) >= candidate_count:
                break
            if rng.random() < 0.55:
                ticket = tuple(sorted(rng.choice(np.array(available), size=TICKET_SIZE, replace=False).tolist()))
            else:
                ticket = self.weighted_sample(available, gentle, TICKET_SIZE, rng)
            if ticket in seen:
                continue
            seen.add(ticket)
            if not self.guard.accept(
                ticket,
                min_sum=min_sum,
                max_sum=max_sum,
                max_chain=max_chain,
                max_one_sector=max_one_sector,
                max_from_latest=max_from_latest,
                max_same_last_digit=max_same_last_digit,
                max_arithmetic_triples=max_arithmetic_triples,
            ):
                continue
            meta = self.guard.score(ticket)
            rows.append({"Zestaw": format_ticket(ticket), **meta})

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("Jakość", ascending=False).reset_index(drop=True)

    def generate_portfolio(
        self,
        portfolio_size: int,
        candidate_count: int,
        max_overlap_between_coupons: int,
        **kwargs,
    ) -> pd.DataFrame:
        candidates = self.generate_antyblad_candidates(candidate_count=max(candidate_count, portfolio_size * 80), **kwargs)
        if candidates.empty:
            return candidates
        selected_rows = []
        selected_tickets: List[Tuple[int, ...]] = []
        for _, row in candidates.iterrows():
            ticket = tuple(int(x) for x in re.findall(r"\d+", str(row["Zestaw"])))
            if all(ticket_overlap(ticket, existing) <= max_overlap_between_coupons for existing in selected_tickets):
                selected_tickets.append(ticket)
                selected_rows.append(row.to_dict())
            if len(selected_rows) >= portfolio_size:
                break
        if not selected_rows:
            return pd.DataFrame()
        out = pd.DataFrame(selected_rows)
        out.insert(0, "Kupon", range(1, len(out) + 1))
        return out.reset_index(drop=True)

    def auto_mode(self, recent_window: int = 80) -> Dict[str, object]:
        recent = self.engine.df.head(recent_window)
        sums = recent["Suma"].to_numpy(dtype=np.float64)
        latest = self.model.latest_draw
        latest_sum = sum(latest)
        global_mean = self.model.sum_mean
        recent_mean = float(sums.mean())
        recent_std = float(sums.std(ddof=0)) if float(sums.std(ddof=0)) > 0 else self.model.sum_std

        # Prosty doradca ustawień, bez udawania przewidywania.
        recommended_min = int(max(90, round(global_mean - self.model.sum_std * 0.85)))
        recommended_max = int(min(210, round(global_mean + self.model.sum_std * 0.85)))
        note = []
        if latest_sum > global_mean + self.model.sum_std:
            note.append("ostatnia suma była wysoka — nie kopiuj agresywnie wysokiego zakresu")
        elif latest_sum < global_mean - self.model.sum_std:
            note.append("ostatnia suma była niska — unikaj skrajnie niskiego kuponu")
        else:
            note.append("ostatnia suma jest w normalnym zakresie")
        if abs(recent_mean - global_mean) > self.model.sum_std * 0.35:
            note.append("ostatnie okno ma przesunięcie sumy, więc lepszy jest tryb defensywny")
        else:
            note.append("ostatnie okno nie odbiega mocno od całości")
        return {
            "Rekomendowany_tryb": "Portfel Anty-Błąd" if recent_std > self.model.sum_std * 0.92 else "Kupon Anty-Błąd",
            "Zakres_sumy": f"{recommended_min}–{recommended_max}",
            "Limit_z_ostatniego": 1,
            "Max_sektor": 2,
            "Max_łańcuch": 2,
            "Max_końcówka": 2,
            "Komentarz": "; ".join(note),
        }

    def generate_experimental(
        self,
        strategy: str,
        count: int,
        min_sum: Optional[int],
        max_sum: Optional[int],
        max_chain: int,
        max_one_sector: int,
        max_from_latest: int,
        banned_numbers: Sequence[int],
        hot_count: int = 3,
        cold_count: int = 3,
    ) -> pd.DataFrame:
        rng = np.random.default_rng()
        banned = set(map(int, banned_numbers))
        available = [n for n in range(NUMBER_MIN, NUMBER_MAX + 1) if n not in banned]
        table = self.model.probability_table.copy()
        rows: List[Dict[str, object]] = []
        seen: set[Tuple[int, ...]] = set()

        if strategy == "Złota 6 z najczęstszych":
            ranked = table.sort_values(["Wystąpienia_globalnie", "Liczba"], ascending=[False, True])["Liczba"].astype(int).tolist()
            ticket = tuple(sorted([n for n in ranked if n in available][:6]))
            meta = self.guard.score(ticket)
            return pd.DataFrame([{"Zestaw": format_ticket(ticket), **meta}])

        if strategy == "Najzimniejsza 6":
            ranked = table.sort_values(["Wystąpienia_globalnie", "Liczba"], ascending=[True, True])["Liczba"].astype(int).tolist()
            ticket = tuple(sorted([n for n in ranked if n in available][:6]))
            meta = self.guard.score(ticket)
            return pd.DataFrame([{"Zestaw": format_ticket(ticket), **meta}])

        if strategy == "Mix gorące/zimne":
            hot_count = max(0, min(6, int(hot_count)))
            cold_count = 6 - hot_count
            hot_rank = table.sort_values(["Wystąpienia_globalnie", "Liczba"], ascending=[False, True])["Liczba"].astype(int).tolist()
            cold_rank = table.sort_values(["Wystąpienia_globalnie", "Liczba"], ascending=[True, True])["Liczba"].astype(int).tolist()
            hot_pool = [n for n in hot_rank if n in available][:25]
            cold_pool = [n for n in cold_rank if n in available][:25]
            attempts = max(500, count * 80)
            for _ in range(attempts):
                if len(rows) >= count:
                    break
                part_hot = rng.choice(np.array(hot_pool), size=hot_count, replace=False).tolist() if hot_count else []
                cold_available = [n for n in cold_pool if n not in part_hot]
                if len(cold_available) < cold_count:
                    continue
                part_cold = rng.choice(np.array(cold_available), size=cold_count, replace=False).tolist() if cold_count else []
                ticket = tuple(sorted(part_hot + part_cold))
                if ticket in seen:
                    continue
                seen.add(ticket)
                if not self.guard.accept(ticket, min_sum, max_sum, max_chain, max_one_sector, max_from_latest, 3, 5):
                    continue
                rows.append({"Zestaw": format_ticket(ticket), **self.guard.score(ticket)})
            return pd.DataFrame(rows).sort_values("Jakość", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()

        if strategy == "Rezonans":
            weights = self.model.resonance_weight
        elif strategy == "Najlepszy traf / hybryda":
            weights = self.model.hybrid_weight
        elif strategy == "Opóźnione":
            weights = self.model.delay_weight
        else:
            weights = self.model.hybrid_weight

        attempts = max(800, count * 120)
        for _ in range(attempts):
            if len(rows) >= count:
                break
            ticket = self.weighted_sample(available, weights, TICKET_SIZE, rng)
            if ticket in seen:
                continue
            seen.add(ticket)
            if not self.guard.accept(ticket, min_sum, max_sum, max_chain, max_one_sector, max_from_latest, 3, 5):
                continue
            rows.append({"Zestaw": format_ticket(ticket), **self.guard.score(ticket)})
        return pd.DataFrame(rows).sort_values("Jakość", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


# ============================================================
# HISTORIA SKUTECZNOŚCI WŁASNYCH KUPONÓW
# ============================================================

class CouponHistory:
    COLUMNS = [
        "Data_dodania",
        "Moduł",
        "Kupon",
        "Wynik_losowania",
        "Trafione",
        "Notatka",
    ]

    @staticmethod
    def load() -> pd.DataFrame:
        if not HISTORY_FILE.exists():
            return pd.DataFrame(columns=CouponHistory.COLUMNS)
        try:
            df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
            for col in CouponHistory.COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[CouponHistory.COLUMNS]
        except Exception:
            return pd.DataFrame(columns=CouponHistory.COLUMNS)

    @staticmethod
    def save(df: pd.DataFrame) -> None:
        df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

    @staticmethod
    def add_record(module: str, coupon: Sequence[int], result: Sequence[int], note: str) -> None:
        df = CouponHistory.load()
        coupon_set = set(map(int, coupon))
        result_set = set(map(int, result))
        hits = len(coupon_set.intersection(result_set))
        row = {
            "Data_dodania": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Moduł": module,
            "Kupon": format_ticket(coupon),
            "Wynik_losowania": format_ticket(result),
            "Trafione": hits,
            "Notatka": note,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        CouponHistory.save(df)

    @staticmethod
    def summary(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        df2 = df.copy()
        df2["Trafione"] = pd.to_numeric(df2["Trafione"], errors="coerce").fillna(0).astype(int)
        grouped = df2.groupby("Moduł").agg(
            Próby=("Trafione", "count"),
            Średnia=("Trafione", "mean"),
            Max=("Trafione", "max"),
            Trafienia_2plus=("Trafione", lambda s: int((s >= 2).sum())),
            Trafienia_3plus=("Trafione", lambda s: int((s >= 3).sum())),
        ).reset_index()
        grouped["Średnia"] = grouped["Średnia"].round(3)
        return grouped.sort_values(["Średnia", "Max", "Próby"], ascending=[False, False, False])


# ============================================================
# RENDERERY
# ============================================================

class LottoGridRenderer:
    @staticmethod
    def render(selected: Sequence[int], title: str) -> str:
        selected_set = set(map(int, selected))
        cells = []
        for number in range(NUMBER_MIN, NUMBER_MAX + 1):
            css = "selected" if number in selected_set else ""
            cells.append(f'<div class="lotto-cell {css}">{number:02d}</div>')
        return f"""
        <div class="lotto-wrapper">
            <div class="lotto-main-title">{title}</div>
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
            margin: 16px 0 28px 0;
            max-width: 620px;
            box-shadow: 0 18px 42px rgba(0,0,0,0.30);
        }}
        .lotto-main-title {{
            color: #f9fafb;
            font-weight: 900;
            font-size: 18px;
            margin-bottom: 16px;
        }}
        .lotto-grid {{
            display: grid;
            grid-template-columns: repeat(7, 44px);
            gap: 8px;
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
            background: radial-gradient(circle at 30% 30%, #fff7ed, #fde047 35%, #f97316 75%, #dc2626 100%);
            color: #111827;
            border: 2px solid #fde68a;
            box-shadow: 0 0 18px rgba(250, 204, 21, 0.9), 0 0 30px rgba(220, 38, 38, 0.45);
            transform: scale(1.08);
        }}
        </style>
        """


# ============================================================
# STREAMLIT APP
# ============================================================

class LottoAntyBladApp:
    def __init__(self):
        st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide")
        self.df = self.load_database()
        self.engine = LottoAnalyticsEngine(self.df)
        self.columns = [f"N{i}" for i in range(1, DRAW_SIZE + 1)]

    def load_database(self) -> pd.DataFrame:
        with st.sidebar:
            st.header("📄 Plik i parser")
            pdf_path_text = st.text_input(
                "Nazwa pliku PDF",
                value=str(PDF_PATH),
                help="Domyślnie Wyniki060626.PDF. Plik musi być w tym samym folderze co aplikacja.",
            )
            if st.button("🔄 Wyczyść cache i odśwież", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            st.caption("Po podmianie PDF kliknij odświeżenie, aby aplikacja czytała nową bazę.")

        pdf_path = Path(pdf_path_text)
        signature = LottoPdfParser.file_signature(pdf_path)
        try:
            return cached_load_lotto(str(pdf_path), signature)
        except Exception as error:
            st.error(f"Nie udało się odczytać PDF: {error}")
            st.stop()

    def render_header(self) -> None:
        st.title("🛡️ Lotto 6/49 Laboratorium Anty-Błąd PRO")
        st.info(
            "Ta wersja zmienia filozofię: aplikacja nie udaje, że przewiduje przyszłe losowanie. "
            "Główna część służy do eliminowania słabych, skrajnych i typowo ludzkich kuponów. "
            "Dawne moduły typu gorące/zimne/najlepszy traf są dalej dostępne, ale przeniesione do osobnej sekcji eksperymentalnej."
        )
        newest = self.df.iloc[0]
        oldest = self.df.iloc[-1]
        ok, warnings = validate_database(self.df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Odczytane losowania", len(self.df))
        c2.metric("Najnowsze", int(newest["Losowanie"]))
        c3.metric("Najstarsze", int(oldest["Losowanie"]))
        c4.metric("Zakres", "6 z 49")

        for msg in ok:
            st.success(msg)
        for msg in warnings:
            st.warning(msg)

        with st.expander("📌 Dlaczego aplikacja została przebudowana?", expanded=False):
            st.write(
                "W uczciwej grze losowej historia nie daje pewnej przewagi predykcyjnej. "
                "Dlatego główny moduł nie próbuje zgadywać, tylko tworzy kupony o dobrym balansie i usuwa typowe błędy: "
                "skrajne sumy, zbyt wiele liczb z jednego sektora, ciągi kolejnych liczb, zbyt dużo liczb z ostatniego losowania, "
                "dziwne układy końcówek oraz nadmiernie podobne kupony w portfelu."
            )
            st.write("Najlepszy przykład ustawień na start:")
            st.code(
                "Zakres sumy: średnia historyczna ± 0.85 odchylenia\n"
                "Maksymalnie z ostatniego losowania: 1\n"
                "Maksymalnie z jednego sektora: 2\n"
                "Maksymalny ciąg kolejnych liczb: 2\n"
                "Maksymalnie ta sama końcówka: 2\n"
                "Portfel kuponów: maksymalny overlap 2"
            )

        with st.expander("Podgląd kontroli parsera", expanded=False):
            newest_numbers = [int(newest[col]) for col in self.columns]
            oldest_numbers = [int(oldest[col]) for col in self.columns]
            st.write("Najnowszy wynik:")
            st.code(f'{int(newest["Losowanie"])}: ' + format_ticket(newest_numbers))
            st.write("Najstarszy wynik:")
            st.code(f'{int(oldest["Losowanie"])}: ' + format_ticket(oldest_numbers))

    def common_quality_controls(self, model: NumberModel, prefix: str = "main") -> Dict[str, object]:
        st.subheader("⚙️ Strażnik jakości — ustawienia")
        st.caption("Te filtry są najważniejszą częścią aplikacji. To one eliminują słabe układy.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_sum = st.checkbox(
                "Filtr sumy",
                value=True,
                key=f"{prefix}_use_sum",
                help="Zalecane. Dla Lotto 6/49 skrajnie niskie i skrajnie wysokie sumy są zwykle mniej praktyczne.",
            )
            if use_sum:
                default_min = int(max(80, round(model.sum_mean - model.sum_std * 0.85)))
                default_max = int(min(220, round(model.sum_mean + model.sum_std * 0.85)))
                min_sum = st.number_input("Minimalna suma", 21, 279, default_min, key=f"{prefix}_min_sum")
                max_sum = st.number_input("Maksymalna suma", 21, 279, default_max, key=f"{prefix}_max_sum")
            else:
                min_sum = None
                max_sum = None
        with c2:
            max_from_latest = st.slider(
                "Maks. z ostatniego losowania",
                0,
                6,
                1,
                key=f"{prefix}_latest",
                help="Zalecane: 1. Ustaw 2, jeśli chcesz dopuścić więcej powtórek, ale nie przesadzaj.",
            )
            max_chain = st.slider(
                "Maks. ciąg kolejnych liczb",
                1,
                6,
                2,
                key=f"{prefix}_chain",
                help="Zalecane: 2. Odrzuca np. 10-11-12 jako zbyt ładny ciąg.",
            )
        with c3:
            max_one_sector = st.slider(
                "Maks. z jednego sektora",
                1,
                6,
                2,
                key=f"{prefix}_sector",
                help=f"Zalecane: 2. Sektory: {SECTOR_LABEL}.",
            )
            max_same_last_digit = st.slider(
                "Maks. ta sama końcówka",
                1,
                6,
                2,
                key=f"{prefix}_ending",
                help="Zalecane: 2. Odrzuca np. za dużo liczb kończących się na 7.",
            )
        with c4:
            max_arithmetic_triples = st.slider(
                "Maks. trójek arytmetycznych",
                0,
                8,
                4,
                key=f"{prefix}_arith",
                help="Niżej = ostrzej. Odrzuca zbyt geometryczne układy typu 10-20-30.",
            )
            banned_text = st.text_input(
                "Liczby wykluczone",
                value="",
                key=f"{prefix}_banned",
                help="Opcjonalnie. Przykład: 01 02 49.",
            )
        return {
            "min_sum": min_sum,
            "max_sum": max_sum,
            "max_chain": max_chain,
            "max_one_sector": max_one_sector,
            "max_from_latest": max_from_latest,
            "max_same_last_digit": max_same_last_digit,
            "max_arithmetic_triples": max_arithmetic_triples,
            "banned_numbers": parse_number_list(banned_text),
        }

    def render_recommended_tab(self) -> None:
        st.header("🛡️ Główne narzędzie: Anty-Błąd i Portfel kuponów")
        st.info(
            "To jest teraz zalecany sposób pracy. Aplikacja generuje wiele kandydatów, "
            "odrzuca słabe i pokazuje tylko najlepsze według Strażnika jakości."
        )
        recent_window = st.select_slider(
            "Okno analizy statystycznej",
            options=[60, 100, 160, 220, 300, 500, 999],
            value=220,
            help="Nie jest to próba przepowiedni. Okno pomaga jedynie w łagodnej ocenie jakości i par.",
        )
        model = self.engine.prepare_model(recent_window=recent_window)
        generator = LottoGenerator(self.engine, model)
        controls = self.common_quality_controls(model, prefix="rec")

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            mode = st.selectbox(
                "Tryb zalecany",
                ["Kupon Anty-Błąd", "Portfel Anty-Błąd"],
                index=0,
                help="Kupon = jeden najmocniejszy zestaw. Portfel = kilka różnych kuponów z ograniczonym podobieństwem.",
            )
        with c2:
            candidate_count = st.select_slider(
                "Ilu kandydatów sprawdzić",
                options=[500, 1000, 2000, 5000, 10000],
                value=2000,
                help="Im więcej, tym dokładniej, ale wolniej. Dla laptopa HP Victus 2000–5000 jest dobrym wyborem.",
            )
        with c3:
            portfolio_size = st.slider("Liczba kuponów w portfelu", 1, 20, 5)
        with c4:
            max_overlap = st.slider(
                "Maks. wspólnych liczb między kuponami",
                0,
                5,
                2,
                help="Zalecane: 2. Dzięki temu kupony nie są kopiami siebie.",
            )

        if st.button("🛡️ GENERUJ ZALECANY KUPOŃ / PORTFEL", type="primary", use_container_width=True):
            start = time.perf_counter()
            if mode == "Kupon Anty-Błąd":
                result = generator.generate_antyblad_candidates(candidate_count=candidate_count, seed=None, **controls).head(1)
                if not result.empty:
                    result.insert(0, "Kupon", [1])
            else:
                result = generator.generate_portfolio(
                    portfolio_size=portfolio_size,
                    candidate_count=candidate_count,
                    max_overlap_between_coupons=max_overlap,
                    seed=None,
                    **controls,
                )
            elapsed = time.perf_counter() - start
            if result.empty:
                st.error("Nie znaleziono kuponu spełniającego filtry. Poluzuj sumę, sektor albo limit z ostatniego losowania.")
            else:
                st.success(f"Gotowe. Czas liczenia: {elapsed:.2f} s.")
                st.dataframe(result, use_container_width=True, hide_index=True)
                text = "\n".join(
                    f"Kupon {int(row['Kupon'])}: {row['Zestaw']} | jakość={row['Jakość']} | suma={row['Suma']} | ryzyka={row['Ryzyka']}"
                    for _, row in result.iterrows()
                )
                st.text_area("Kopiuj kupony", value=text, height=140)
                st.download_button("⬇️ Pobierz TXT", text.encode("utf-8"), "lotto_antyblad_kupony.txt", "text/plain")
                st.download_button(
                    "⬇️ Pobierz CSV",
                    result.to_csv(index=False).encode("utf-8-sig"),
                    "lotto_antyblad_kupony.csv",
                    "text/csv",
                )
                st.subheader("🎫 Blankiet")
                for _, row in result.iterrows():
                    nums = [int(x) for x in re.findall(r"\d+", str(row["Zestaw"]))]
                    st.markdown(LottoGridRenderer.render(nums, f"Kupon {int(row['Kupon'])}: {format_ticket(nums)}"), unsafe_allow_html=True)

    def render_auto_tab(self) -> None:
        st.header("⏱️ Auto-tryb: szybka decyzja po aktualizacji PDF")
        st.info(
            "Ten moduł nie przewiduje losowania. Po odświeżeniu PDF podpowiada bezpieczne ustawienia, "
            "żeby szybko wygenerować kupon bez ręcznego strojenia filtrów."
        )
        recent_window = st.select_slider("Okno auto-analizy", options=[60, 100, 160, 220, 300], value=100)
        model = self.engine.prepare_model(recent_window=recent_window)
        generator = LottoGenerator(self.engine, model)
        auto = generator.auto_mode(recent_window=recent_window)
        st.dataframe(pd.DataFrame([auto]), use_container_width=True, hide_index=True)

        if st.button("⏱️ WYGNERUJ AUTO-KUPON ANTY-BŁĄD", type="primary", use_container_width=True):
            min_sum, max_sum = [int(x) for x in str(auto["Zakres_sumy"]).split("–")]
            result = generator.generate_antyblad_candidates(
                candidate_count=2500,
                min_sum=min_sum,
                max_sum=max_sum,
                max_chain=int(auto["Max_łańcuch"]),
                max_one_sector=int(auto["Max_sektor"]),
                max_from_latest=int(auto["Limit_z_ostatniego"]),
                max_same_last_digit=2,
                max_arithmetic_triples=4,
                banned_numbers=[],
                seed=None,
            ).head(1)
            if result.empty:
                st.error("Auto-tryb nie znalazł kuponu. Spróbuj ręcznie w module Anty-Błąd.")
            else:
                result.insert(0, "Kupon", [1])
                st.dataframe(result, use_container_width=True, hide_index=True)
                nums = [int(x) for x in re.findall(r"\d+", str(result.iloc[0]["Zestaw"]))]
                st.markdown(LottoGridRenderer.render(nums, f"Auto-kupon: {format_ticket(nums)}"), unsafe_allow_html=True)

    def render_history_tab(self) -> None:
        st.header("📘 Historia skuteczności moich kuponów")
        st.info(
            "To jeden z najważniejszych modułów. Zapisuj realne wyniki swoich kuponów. "
            "Po czasie zobaczysz, które tryby i ustawienia faktycznie dawały najlepsze trafienia u Ciebie."
        )
        c1, c2 = st.columns(2)
        with c1:
            module = st.selectbox(
                "Moduł, którym był wygenerowany kupon",
                ["Anty-Błąd", "Portfel Anty-Błąd", "Auto-tryb", "Eksperymentalny", "Chybił-trafił", "Inny"],
            )
            coupon_text = st.text_input("Mój kupon", help="Wpisz 6 liczb, np. 05 12 17 36 39 49")
        with c2:
            result_text = st.text_input("Wynik losowania", help="Wpisz 6 liczb z faktycznego losowania")
            note = st.text_input("Notatka", value="")

        if st.button("➕ Zapisz wynik kuponu", use_container_width=True):
            coupon = parse_number_list(coupon_text)
            result = parse_number_list(result_text)
            if len(coupon) != 6 or len(result) != 6:
                st.error("Kupon i wynik muszą mieć dokładnie 6 liczb z zakresu 1–49.")
            else:
                CouponHistory.add_record(module, coupon, result, note)
                st.success(f"Zapisano. Trafione: {ticket_overlap(coupon, result)}/6")

        df_hist = CouponHistory.load()
        st.subheader("Podsumowanie modułów")
        summary = CouponHistory.summary(df_hist)
        if summary.empty:
            st.warning("Brak zapisanych wyników. Dodaj pierwsze realne sprawdzenie kuponu.")
        else:
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("Pełna historia")
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        if not df_hist.empty:
            st.download_button(
                "⬇️ Pobierz historię CSV",
                df_hist.to_csv(index=False).encode("utf-8-sig"),
                "historia_skutecznosci_lotto.csv",
                "text/csv",
            )
        if st.button("🗑️ Wyczyść historię", use_container_width=True):
            CouponHistory.save(pd.DataFrame(columns=CouponHistory.COLUMNS))
            st.rerun()

    def render_analysis_tab(self) -> None:
        st.header("📊 Analiza bazy")
        recent_window = st.select_slider("Okno analizy", options=[60, 100, 160, 220, 300, 500, 999], value=220)
        model = self.engine.prepare_model(recent_window=recent_window)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Średnia suma", round(model.sum_mean, 2))
        c2.metric("Odchylenie sumy", round(model.sum_std, 2))
        c3.metric("Najnowszy wynik", format_ticket(model.latest_draw))
        c4.metric("Suma ostatniego", sum(model.latest_draw))

        tab1, tab2, tab3, tab4 = st.tabs(["Liczby", "Pary", "Trójki", "Archiwum"])
        with tab1:
            st.subheader("Tabela wag i częstotliwości")
            st.dataframe(model.probability_table, use_container_width=True, hide_index=True)
            chart_data = self.engine.frequency_table(window=recent_window).sort_values("Liczba").set_index("Liczba")["Wystąpienia"]
            st.bar_chart(chart_data)
        with tab2:
            st.subheader("Najczęstsze pary")
            st.dataframe(self.engine.top_pairs(window=recent_window, limit=80), use_container_width=True, hide_index=True)
        with tab3:
            st.subheader("Najczęstsze trójki")
            st.dataframe(self.engine.top_triples(window=recent_window, limit=80), use_container_width=True, hide_index=True)
        with tab4:
            display = self.df.copy()
            display["Liczby"] = display[self.columns].apply(lambda row: format_ticket([int(x) for x in row]), axis=1)
            st.dataframe(display[["Losowanie", "Liczby", "Suma", "Parzyste", "Nieparzyste", "Niskie_1_24", "Wysokie_25_49"]], use_container_width=True, hide_index=True)

    def render_experimental_tab(self) -> None:
        st.header("⚠️ Moduły eksperymentalne — dostępne, ale mniej zalecane")
        st.warning(
            "Te moduły zostają w aplikacji, ale nie są główną rekomendacją. "
            "Gorące, zimne, rezonans i najlepszy traf dobrze wyglądają analitycznie, jednak nie ma dowodu, "
            "że realnie przewidują uczciwe losowanie. Używaj ich jako ciekawostki lub drugi kierunek, nie jako podstawę."
        )
        recent_window = st.select_slider("Okno modelu eksperymentalnego", options=[60, 100, 160, 220, 300, 500, 999], value=220)
        model = self.engine.prepare_model(recent_window=recent_window)
        generator = LottoGenerator(self.engine, model)
        controls = self.common_quality_controls(model, prefix="exp")

        c1, c2, c3 = st.columns(3)
        with c1:
            strategy = st.selectbox(
                "Strategia eksperymentalna",
                [
                    "Najlepszy traf / hybryda",
                    "Rezonans",
                    "Opóźnione",
                    "Złota 6 z najczęstszych",
                    "Najzimniejsza 6",
                    "Mix gorące/zimne",
                ],
            )
        with c2:
            count = st.slider("Ile kuponów", 1, 20, 5)
        with c3:
            hot_count = st.slider(
                "Mix: ile gorących",
                0,
                6,
                3,
                help="Dotyczy tylko strategii Mix gorące/zimne. Reszta to zimne.",
            )

        if st.button("⚠️ Generuj eksperymentalnie", use_container_width=True):
            result = generator.generate_experimental(strategy=strategy, count=count, hot_count=hot_count, cold_count=6-hot_count, **controls)
            if result.empty:
                st.error("Nie znaleziono kuponu spełniającego filtry.")
            else:
                result.insert(0, "Kupon", range(1, len(result) + 1))
                st.dataframe(result, use_container_width=True, hide_index=True)
                for _, row in result.head(5).iterrows():
                    nums = [int(x) for x in re.findall(r"\d+", str(row["Zestaw"]))]
                    st.markdown(LottoGridRenderer.render(nums, f"Eksperymentalny {int(row['Kupon'])}: {format_ticket(nums)}"), unsafe_allow_html=True)

    def run(self) -> None:
        self.render_header()
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "🛡️ Anty-Błąd",
                "⏱️ Auto-tryb",
                "📘 Historia skuteczności",
                "📊 Analiza bazy",
                "⚠️ Eksperymentalne",
            ]
        )
        with tab1:
            self.render_recommended_tab()
        with tab2:
            self.render_auto_tab()
        with tab3:
            self.render_history_tab()
        with tab4:
            self.render_analysis_tab()
        with tab5:
            self.render_experimental_tab()


if __name__ == "__main__":
    LottoAntyBladApp().run()
