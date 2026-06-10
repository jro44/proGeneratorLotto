from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader


PDF_PATH = Path("./Wyniki060626.PDF")
DRAW_MIN = 6000
DRAW_MAX = 8000
NUMBER_MIN = 1
NUMBER_MAX = 49
BALLS_PER_DRAW = 6
SUM_MIN = 95
SUM_MAX = 205
ALLOWED_ODD_COUNTS = {2, 3, 4}
MAX_CONSECUTIVE_CHAIN = 2


@dataclass(frozen=True)
class LottoDraw:
    """Represents one clean Lotto 6/49 draw."""

    draw_id: int
    numbers: Tuple[int, int, int, int, int, int]


class LocalPdfParser:
    """Parser PDF MultiPasko: wyniki są najpierw, numery losowań dopiero pod nimi."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def exists(self) -> bool:
        return self.pdf_path.exists() and self.pdf_path.is_file()

    @staticmethod
    def split_number_token(token: str) -> List[int]:
        """
        MultiPasko czasem skleja liczby:
        0203 -> 02, 03
        4647 -> 46, 47
        020304 -> 02, 03, 04
        """
        if len(token) == 4 and 6000 <= int(token) <= 8000:
            return [int(token)]

        if len(token) > 2 and len(token) % 2 == 0:
            return [int(token[i:i + 2]) for i in range(0, len(token), 2)]

        return [int(token)]

    def read_text(self) -> str:
        reader = PdfReader(str(self.pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def parse_page(self, text: str) -> List[LottoDraw]:
        result_rows = []
        draw_ids = []

        for line in text.splitlines():
            line = line.strip()

            if not line or "Lotto" in line:
                continue

            tokens = re.findall(r"\d+", line)

            if not tokens:
                continue

            if len(tokens) == 1 and len(tokens[0]) == 4:
                value = int(tokens[0])
                if DRAW_MIN <= value <= DRAW_MAX:
                    draw_ids.append(value)
                    continue

            numbers = []

            for token in tokens:
                if len(token) == 4 and DRAW_MIN <= int(token) <= DRAW_MAX:
                    continue

                numbers.extend(self.split_number_token(token))

            numbers = [n for n in numbers if NUMBER_MIN <= n <= NUMBER_MAX]

            if len(numbers) == BALLS_PER_DRAW and len(set(numbers)) == BALLS_PER_DRAW:
                result_rows.append(tuple(sorted(numbers)))

        draws = []

        for draw_id, numbers in zip(draw_ids, result_rows):
            draws.append(LottoDraw(draw_id=draw_id, numbers=numbers))

        return draws

    def parse(self) -> pd.DataFrame:
        if not self.exists():
            raise FileNotFoundError(
                'Nie znaleziono pliku "Wyniki060626.PDF". '
                "Umieść go w tym samym folderze co aplikacja."
            )

        reader = PdfReader(str(self.pdf_path))
        all_draws: Dict[int, LottoDraw] = {}

        for page in reader.pages:
            text = page.extract_text() or ""
            page_draws = self.parse_page(text)

            for draw in page_draws:
                all_draws[draw.draw_id] = draw

        if not all_draws:
            raise ValueError("Nie udało się odczytać poprawnych losowań z PDF.")

        rows = []

        for draw in all_draws.values():
            rows.append(
                {
                    "Losowanie": draw.draw_id,
                    "L1": draw.numbers[0],
                    "L2": draw.numbers[1],
                    "L3": draw.numbers[2],
                    "L4": draw.numbers[3],
                    "L5": draw.numbers[4],
                    "L6": draw.numbers[5],
                }
            )

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Losowanie"])
        df = df.sort_values("Losowanie", ascending=False).reset_index(drop=True)

        return df


class FrequencyAnalyzer:
    """Frequency analyzer for global and rolling hot/cold states."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = [f"L{i}" for i in range(1, 7)]

    def frequency_table(self, window: int | None = None) -> pd.DataFrame:
        source = self.df.head(window) if window else self.df
        values = source[self.columns].to_numpy().flatten()
        counts = pd.Series(values).value_counts().reindex(range(1, 50), fill_value=0)
        table = pd.DataFrame({"Liczba": counts.index, "Wystąpienia": counts.values, "Procent_losowań": np.round(counts.values / len(source) * 100, 2)})
        hot_border = table["Wystąpienia"].quantile(0.75)
        cold_border = table["Wystąpienia"].quantile(0.25)
        table["Stan"] = "Neutralna"
        table.loc[table["Wystąpienia"] >= hot_border, "Stan"] = "Gorąca"
        table.loc[table["Wystąpienia"] <= cold_border, "Stan"] = "Zimna"
        return table.sort_values(["Wystąpienia", "Liczba"], ascending=[False, True]).reset_index(drop=True)

    def probability(self, window: int | None = None) -> np.ndarray:
        table = self.frequency_table(window=window).sort_values("Liczba")
        weights = table["Wystąpienia"].to_numpy(dtype=float) + 1.0
        return weights / weights.sum()


class CoOccurrenceCouplingEngine:
    """Co-occurrence coupling matrix for pairs and triples."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = [f"L{i}" for i in range(1, 7)]

    def pair_matrix(self, window: int | None = None) -> np.ndarray:
        source = self.df.head(window) if window else self.df
        matrix = np.zeros((49, 49), dtype=float)
        for row in source[self.columns].to_numpy():
            nums = sorted(map(int, row))
            for a, b in itertools.combinations(nums, 2):
                matrix[a - 1, b - 1] += 1.0
                matrix[b - 1, a - 1] += 1.0
        max_value = matrix.max()
        return matrix / max_value if max_value > 0 else matrix

    def coupling_factor(self, selected: Sequence[int], window: int = 50, strength: float = 1.0) -> np.ndarray:
        matrix = self.pair_matrix(window=window)
        factor = np.ones(49, dtype=float)
        for number in selected:
            if NUMBER_MIN <= number <= NUMBER_MAX:
                factor *= 1.0 + strength * matrix[number - 1]
        for number in selected:
            if NUMBER_MIN <= number <= NUMBER_MAX:
                factor[number - 1] = 0.0
        if factor.sum() <= 0:
            factor = np.ones(49, dtype=float)
        return factor / factor.sum()

    def top_pairs(self, window: int = 50, limit: int = 30) -> pd.DataFrame:
        source = self.df.head(window)
        counter: Dict[Tuple[int, int], int] = {}
        for row in source[self.columns].to_numpy():
            for pair in itertools.combinations(sorted(map(int, row)), 2):
                counter[pair] = counter.get(pair, 0) + 1
        rows = [{"Para": f"{p[0]:02d}-{p[1]:02d}", "Wystąpienia": c} for p, c in counter.items()]
        return pd.DataFrame(rows).sort_values("Wystąpienia", ascending=False).head(limit).reset_index(drop=True)

    def top_triples(self, window: int = 50, limit: int = 30) -> pd.DataFrame:
        source = self.df.head(window)
        counter: Dict[Tuple[int, int, int], int] = {}
        for row in source[self.columns].to_numpy():
            for triple in itertools.combinations(sorted(map(int, row)), 3):
                counter[triple] = counter.get(triple, 0) + 1
        rows = [{"Trójka": f"{t[0]:02d}-{t[1]:02d}-{t[2]:02d}", "Wystąpienia": c} for t, c in counter.items()]
        return pd.DataFrame(rows).sort_values("Wystąpienia", ascending=False).head(limit).reset_index(drop=True)


class KineticMemoryEngine:
    """Short-distance kinetic memory and Markov successor engine."""

    def __init__(self, df: pd.DataFrame):
        self.df_newest = df.sort_values("Losowanie", ascending=False).reset_index(drop=True)
        self.columns = [f"L{i}" for i in range(1, 7)]

    def rolling_delta_frame(self, window: int = 30) -> pd.DataFrame:
        source = self.df_newest.head(window).sort_values("Losowanie", ascending=True).reset_index(drop=True)
        previous_ids = source["Losowanie"].iloc[:-1].to_numpy()
        next_ids = source["Losowanie"].iloc[1:].to_numpy()
        deltas = source[self.columns].diff().dropna().astype(int)
        deltas.columns = [f"Delta_{c}" for c in self.columns]
        return pd.concat([pd.DataFrame({"Z_losowania": previous_ids, "Do_losowania": next_ids}), deltas.reset_index(drop=True)], axis=1)

    def recency_weighted_delta_probability(self, windows: Sequence[int] = (15, 30, 50)) -> np.ndarray:
        scores = np.ones(49, dtype=float)
        newest_numbers = self.df_newest.loc[0, self.columns].to_numpy(dtype=int)
        for window in windows:
            source = self.df_newest.head(window).sort_values("Losowanie", ascending=True).reset_index(drop=True)
            if len(source) < 2:
                continue
            values = source[self.columns].to_numpy(dtype=int)
            deltas = np.diff(values, axis=0)
            for idx, delta_vector in enumerate(deltas):
                recency_rank = len(deltas) - idx
                weight = max(1.0, float(recency_rank))
                for position in range(BALLS_PER_DRAW):
                    candidate = int(newest_numbers[position] + delta_vector[position])
                    if NUMBER_MIN <= candidate <= NUMBER_MAX:
                        scores[candidate - 1] += weight / max(1, window / 10)
                    for near in (candidate - 1, candidate + 1):
                        if NUMBER_MIN <= near <= NUMBER_MAX:
                            scores[near - 1] += 0.20 * weight / max(1, window / 10)
        return scores / scores.sum()

    def successor_probability(self, window: int = 50) -> np.ndarray:
        source = self.df_newest.head(window).sort_values("Losowanie", ascending=True).reset_index(drop=True)
        scores = np.ones(49, dtype=float)
        if len(source) < 2:
            return scores / scores.sum()
        latest_numbers = set(map(int, self.df_newest.loc[0, self.columns].tolist()))
        rows = source[self.columns].to_numpy(dtype=int)
        for idx in range(len(rows) - 1):
            current = set(map(int, rows[idx]))
            successor = set(map(int, rows[idx + 1]))
            recency_weight = idx + 1
            overlap_strength = len(current.intersection(latest_numbers)) + 1
            for number in successor:
                scores[number - 1] += recency_weight * overlap_strength * 0.15
        return scores / scores.sum()

    def final_kinetic_probability(self, recency_weight: float, global_probability: np.ndarray) -> np.ndarray:
        kinetic = 0.60 * self.recency_weighted_delta_probability() + 0.40 * self.successor_probability(window=50)
        kinetic = kinetic / kinetic.sum()
        final = recency_weight * kinetic + (1.0 - recency_weight) * global_probability
        return final / final.sum()


class RealityFilter:
    """Triple physical probability filter."""

    @staticmethod
    def valid_sum(numbers: Tuple[int, ...]) -> bool:
        return SUM_MIN <= sum(numbers) <= SUM_MAX

    @staticmethod
    def valid_odd_even(numbers: Tuple[int, ...]) -> bool:
        return sum(1 for n in numbers if n % 2 == 1) in ALLOWED_ODD_COUNTS

    @staticmethod
    def valid_consecutive_blocks(numbers: Tuple[int, ...]) -> bool:
        sorted_numbers = sorted(numbers)
        longest = 1
        current = 1
        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return longest <= MAX_CONSECUTIVE_CHAIN

    @staticmethod
    def valid_exact_sum(numbers: Tuple[int, ...], target_sum: int | None = None) -> bool:
        if target_sum is None:
            return True
        return sum(numbers) == int(target_sum)

    @classmethod
    def accept(cls, numbers: Tuple[int, ...], target_sum: int | None = None) -> bool:
        return (
            cls.valid_sum(numbers)
            and cls.valid_exact_sum(numbers, target_sum)
            and cls.valid_odd_even(numbers)
            and cls.valid_consecutive_blocks(numbers)
        )


class CyberMechanicalPredictionEngine:
    """Mechanical-cycle emulation prediction engine."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.frequency = FrequencyAnalyzer(df)
        self.kinetic = KineticMemoryEngine(df)
        self.cooccurrence = CoOccurrenceCouplingEngine(df)

    @staticmethod
    def _weighted_pick(available: Sequence[int], weights: np.ndarray) -> int:
        available = sorted(set(int(n) for n in available))
        local_weights = np.array([max(0.000001, weights[n - 1]) for n in available], dtype=float)
        local_weights = local_weights / local_weights.sum()
        return int(np.random.choice(available, p=local_weights))

    def model_probability_table(self, rolling_window: int, recency_weight: float) -> pd.DataFrame:
        global_probability = self.frequency.probability(window=None)
        rolling_probability = self.frequency.probability(window=rolling_window)
        mixed_global = 0.55 * global_probability + 0.45 * rolling_probability
        mixed_global = mixed_global / mixed_global.sum()
        final = self.kinetic.final_kinetic_probability(recency_weight=recency_weight, global_probability=mixed_global)
        return pd.DataFrame({"Liczba": range(1, 50), "Waga_globalna_i_cykl": np.round(mixed_global, 6), "Waga_kinetyczna": np.round(final, 6), "Waga_końcowa": np.round(final, 6)}).sort_values("Waga_końcowa", ascending=False).reset_index(drop=True)

    def _sample_one(self, rolling_window: int, recency_weight: float, coupling_strength: float) -> Tuple[int, int, int, int, int, int]:
        base_weights = self.model_probability_table(rolling_window, recency_weight).sort_values("Liczba")["Waga_końcowa"].to_numpy(dtype=float)
        base_weights = base_weights / base_weights.sum()
        selected: List[int] = []
        available = list(range(1, 50))
        while len(selected) < BALLS_PER_DRAW:
            coupling = self.cooccurrence.coupling_factor(selected, window=rolling_window, strength=coupling_strength)
            dynamic_weights = base_weights * (1.0 + coupling_strength * coupling * 49.0)
            dynamic_weights = dynamic_weights / dynamic_weights.sum()
            chosen = self._weighted_pick(available, dynamic_weights)
            selected.append(chosen)
            available.remove(chosen)
        return tuple(sorted(selected))  # type: ignore[return-value]

    @staticmethod
    def _sum_bounds(available: Sequence[int], picks_left: int) -> Tuple[int, int]:
        sorted_available = sorted(set(int(n) for n in available))
        if picks_left <= 0 or len(sorted_available) < picks_left:
            return 0, 0
        return sum(sorted_available[:picks_left]), sum(sorted_available[-picks_left:])

    def _sample_one_for_target_sum(
        self,
        rolling_window: int,
        recency_weight: float,
        coupling_strength: float,
        target_sum: int,
        restart_limit: int = 400,
    ) -> Tuple[int, int, int, int, int, int] | None:
        """
        Losuje kupon pod konkretną sumę, ale nadal korzysta z wag modelu,
        sprzężenia par i lokalnego sprawdzania, czy suma jest jeszcze osiągalna.
        """
        base_weights = self.model_probability_table(rolling_window, recency_weight).sort_values("Liczba")["Waga_końcowa"].to_numpy(dtype=float)
        base_weights = base_weights / base_weights.sum()

        for _ in range(restart_limit):
            selected: List[int] = []
            available = list(range(NUMBER_MIN, NUMBER_MAX + 1))
            partial_sum = 0

            while len(selected) < BALLS_PER_DRAW:
                picks_left_after_choice = BALLS_PER_DRAW - len(selected) - 1
                coupling = self.cooccurrence.coupling_factor(selected, window=rolling_window, strength=coupling_strength)
                dynamic_weights = base_weights * (1.0 + coupling_strength * coupling * 49.0)

                possible_choices: List[int] = []
                for number in available:
                    remaining = [n for n in available if n != number]
                    min_rest, max_rest = self._sum_bounds(remaining, picks_left_after_choice)
                    new_sum = partial_sum + number
                    if new_sum + min_rest <= target_sum <= new_sum + max_rest:
                        possible_choices.append(number)

                if not possible_choices:
                    break

                local_weights = np.array([max(0.000001, dynamic_weights[n - 1]) for n in possible_choices], dtype=float)
                local_weights = local_weights / local_weights.sum()
                chosen = int(np.random.choice(possible_choices, p=local_weights))
                selected.append(chosen)
                available.remove(chosen)
                partial_sum += chosen

            if len(selected) == BALLS_PER_DRAW:
                candidate = tuple(sorted(selected))
                if sum(candidate) == target_sum:
                    return candidate  # type: ignore[return-value]

        return None

    def generate(
        self,
        count: int,
        rolling_window: int,
        recency_weight: float,
        coupling_strength: float,
        target_sum: int | None = None,
    ) -> pd.DataFrame:
        rows: List[Dict[str, int | str]] = []
        used: set[Tuple[int, ...]] = set()
        attempts = 0
        max_attempts = max(25000, count * 9000) if target_sum is not None else max(12000, count * 6000)

        while len(rows) < count and attempts < max_attempts:
            attempts += 1
            if target_sum is None:
                candidate = self._sample_one(rolling_window, recency_weight, coupling_strength)
            else:
                sampled = self._sample_one_for_target_sum(rolling_window, recency_weight, coupling_strength, target_sum)
                if sampled is None:
                    continue
                candidate = sampled

            if candidate in used or not RealityFilter.accept(candidate, target_sum=target_sum):
                continue

            used.add(candidate)
            odd = sum(1 for n in candidate if n % 2 == 1)
            even = BALLS_PER_DRAW - odd
            rows.append({
                "Kupon": len(rows) + 1,
                "L1": candidate[0],
                "L2": candidate[1],
                "L3": candidate[2],
                "L4": candidate[3],
                "L5": candidate[4],
                "L6": candidate[5],
                "Suma": sum(candidate),
                "Suma_docelowa": int(target_sum) if target_sum is not None else "Auto",
                "Nieparzyste": odd,
                "Parzyste": even,
                "Balans": f"{odd}:{even}",
            })

        return pd.DataFrame(rows)


class LottoBlanketRenderer:
    """HTML renderer for premium digital Lotto blankets."""

    @staticmethod
    def render(numbers: Sequence[int], title: str) -> str:
        selected = set(int(n) for n in numbers)
        cells = []
        for number in range(1, 50):
            css = "selected" if number in selected else ""
            cells.append(f'<div class="lotto-cell {css}">{number:02d}</div>')
        return f"""
        <div class="blanket-wrapper"><div class="blanket-title">{title}</div><div class="lotto-grid">{''.join(cells)}</div></div>
        <style>
        .blanket-wrapper {{ background: linear-gradient(145deg, #0f172a, #111827); border: 1px solid #374151; border-radius: 18px; padding: 18px; margin: 16px 0 26px 0; max-width: 475px; box-shadow: 0 18px 42px rgba(0,0,0,0.35); }}
        .blanket-title {{ color: #f9fafb; font-weight: 900; font-size: 17px; margin-bottom: 14px; letter-spacing: 0.2px; }}
        .lotto-grid {{ display: grid; grid-template-columns: repeat(7, 48px); gap: 8px; }}
        .lotto-cell {{ width: 48px; height: 48px; border-radius: 50%; background: #1f2937; border: 1px solid #4b5563; color: #d1d5db; display: flex; align-items: center; justify-content: center; font-weight: 900; font-size: 14px; }}
        .lotto-cell.selected {{ background: radial-gradient(circle at 30% 30%, #fff7ed, #fbbf24 38%, #dc2626 100%); color: #111827; border: 2px solid #fde68a; box-shadow: 0 0 18px rgba(251, 191, 36, 0.95), 0 0 34px rgba(220, 38, 38, 0.55); transform: scale(1.08); }}
        </style>
        """


class LottoStreamlitApp:
    """Professional Streamlit application controller."""

    def __init__(self):
        st.set_page_config(page_title="Cyber-Maszyna Lotto 6/49", page_icon="🎯", layout="wide")
        self.df = self.load_database()
        self.engine = CyberMechanicalPredictionEngine(self.df)

    @staticmethod
    @st.cache_data(show_spinner="Wczytywanie i parsowanie pliku Wyniki060626.PDF...")
    def cached_parse(path: str) -> pd.DataFrame:
        return LocalPdfParser(Path(path)).parse()

    def load_database(self) -> pd.DataFrame:
        if not PDF_PATH.exists():
            st.error('Nie znaleziono pliku "Wyniki060626.PDF". Umieść go w tym samym folderze co plik aplikacji.')
            st.stop()
        try:
            return self.cached_parse(str(PDF_PATH))
        except Exception as error:
            st.error(f"Nie udało się odczytać pliku PDF: {error}")
            st.stop()

    def render_header(self) -> None:
        st.title("🎯 Cyber-Maszyna Lotto 6/49")
        st.info("Model analizuje krótkodystansową pamięć kinetyczną: ostatnie cykle, przesunięcia kolumn L1–L6, następniki Markowa, współwystępowanie par i trójek oraz filtry fizycznego prawdopodobieństwa. To symulacja statystyczna, nie gwarancja trafienia.")

    def render_generator_tab(self) -> None:
        st.header("🚀 Cyber-Maszyna (Generator)")
        st.info("Generator wzmacnia najnowsze przesunięcia i natychmiastowe następniki historyczne. Macierz sprzężenia wzmacnia liczby, które często tworzyły pary z już wybranymi liczbami. Każdy kupon przechodzi trzy bramki: suma 95–205, balans 3:3/4:2/2:4 i brak łańcucha dłuższego niż dwie kolejne liczby.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            count = st.slider("Liczba kuponów", 1, 10, 5)
        with c2:
            rolling_window = st.select_slider("Okno cyklu", options=[15, 30, 50], value=30)
        with c3:
            recency_weight = st.slider("Nacisk na świeży cykl", 0.10, 0.95, 0.75, 0.05)
        with c4:
            coupling_strength = st.slider("Sprzężenie par", 0.10, 2.00, 1.00, 0.10)

        st.divider()
        s1, s2 = st.columns([1, 2])
        with s1:
            use_target_sum = st.checkbox("🎯 Generuj pod konkretną sumę", value=False)
        with s2:
            target_sum = st.number_input(
                "Podaj sumę kuponu",
                min_value=SUM_MIN,
                max_value=SUM_MAX,
                value=150,
                step=1,
                disabled=not use_target_sum,
                help="Aplikacja wygeneruje 6 liczb, których suma będzie dokładnie taka jak podana wartość. Zachowane zostają filtry: balans parzyste/nieparzyste i brak długiego łańcucha kolejnych liczb.",
            )

        if use_target_sum:
            st.caption(f"Aktywny tryb sumy: aplikacja będzie szukać kuponów z dokładną sumą {int(target_sum)}.")

        if st.button("⚙️ URUCHOM SYMULACJĘ MECHANICZNĄ", use_container_width=True, type="primary"):
            selected_sum = int(target_sum) if use_target_sum else None
            result = self.engine.generate(count, rolling_window, recency_weight, coupling_strength, target_sum=selected_sum)
            if result.empty:
                st.error("Symulacja nie znalazła kuponów spełniających wszystkie filtry. Spróbuj zmienić sumę, okno cyklu albo siłę sprzężenia.")
                return
            st.success("Symulacja mechaniczna zakończona.")
            st.subheader("✅ Wynik symulacji")
            st.dataframe(result, use_container_width=True, hide_index=True)
            lines = []
            for _, row in result.iterrows():
                nums = [int(row[f"L{i}"]) for i in range(1, 7)]
                lines.append(f"Kupon {int(row['Kupon'])}: " + " ".join(f"{n:02d}" for n in nums) + f" | suma={int(row['Suma'])} | balans={row['Balans']}")
            text = "\n".join(lines)
            st.text_area("Kopiuj Zestawy do Schowka", value=text, height=150)
            st.download_button("⬇️ Pobierz kupony TXT", data=text.encode("utf-8"), file_name="cyber_kupony_lotto.txt", mime="text/plain", use_container_width=True)
            st.download_button("⬇️ Pobierz kupony CSV", data=result.to_csv(index=False).encode("utf-8"), file_name="cyber_kupony_lotto.csv", mime="text/csv", use_container_width=True)
            st.subheader("🎫 Interaktywne blankiety 1–49")
            for _, row in result.iterrows():
                nums = [int(row[f"L{i}"]) for i in range(1, 7)]
                title = f"Kupon {int(row['Kupon'])}: {' '.join(f'{n:02d}' for n in nums)}"
                st.markdown(LottoBlanketRenderer.render(nums, title), unsafe_allow_html=True)
        st.subheader("Wagi modelu dla aktualnego ustawienia domyślnego")
        st.dataframe(self.engine.model_probability_table(30, 0.75), use_container_width=True, hide_index=True)

    def render_archive_tab(self) -> None:
        st.header("📄 Archiwum Faktów")
        st.info("To jest surowe archiwum faktów odczytane automatycznie z lokalnego PDF. Najnowsze losowanie znajduje się na górze.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Liczba losowań", len(self.df))
        c2.metric("Najnowsze", int(self.df["Losowanie"].max()))
        c3.metric("Najstarsze", int(self.df["Losowanie"].min()))
        st.dataframe(self.df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Pobierz archiwum CSV", data=self.df.to_csv(index=False).encode("utf-8"), file_name="archiwum_lotto.csv", mime="text/csv", use_container_width=True)

    def render_cycles_tab(self) -> None:
        st.header("🔥❄️ Cykle i Anomalie")
        st.info("Analiza działa na żywym oknie cyklu. Dzięki temu widać, które liczby są gorące lub zimne w ostatnich 15, 30 albo 50 losowaniach.")
        window = st.select_slider("Okno analizy cyklu", options=[15, 30, 50], value=30)
        freq = FrequencyAnalyzer(self.df)
        table = freq.frequency_table(window=window)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Stan liczb w ostatnich {window} losowaniach")
            st.dataframe(table, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Rozkład częstotliwości w cyklu")
            st.bar_chart(table.sort_values("Liczba").set_index("Liczba")["Wystąpienia"])
        co = CoOccurrenceCouplingEngine(self.df)
        pcol, tcol = st.columns(2)
        with pcol:
            st.subheader("Bliźniaki: najczęstsze pary")
            st.dataframe(co.top_pairs(window=window, limit=30), use_container_width=True, hide_index=True)
        with tcol:
            st.subheader("Relacje trójkowe")
            st.dataframe(co.top_triples(window=window, limit=30), use_container_width=True, hide_index=True)

    def render_vectors_tab(self) -> None:
        st.header("📈 Wektory Naciągu i Trajektorii")
        st.info("Wektory pokazują, jak przesuwały się kolumny L1–L6 między kolejnymi losowaniami w ostatnim cyklu. To wizualizacja krótkodystansowej pamięci mechanicznej.")
        kinetic = KineticMemoryEngine(self.df)
        deltas = kinetic.rolling_delta_frame(window=30)
        st.subheader("Macierz przesunięć — ostatnie 30 losowań")
        st.dataframe(deltas, use_container_width=True, hide_index=True)
        st.subheader("Linie trajektorii delta")
        st.line_chart(deltas.set_index("Do_losowania")[["Delta_L1", "Delta_L2", "Delta_L3", "Delta_L4", "Delta_L5", "Delta_L6"]])
        averages = pd.DataFrame({"Pozycja": [f"L{i}" for i in range(1, 7)], "Średnia_delta": [round(float(deltas[f"Delta_L{i}"].mean()), 3) for i in range(1, 7)], "Mediana_delta": [round(float(deltas[f"Delta_L{i}"].median()), 3) for i in range(1, 7)], "Odchylenie": [round(float(deltas[f"Delta_L{i}"].std()), 3) for i in range(1, 7)]})
        st.subheader("Średnie wektory naciągu")
        st.dataframe(averages, use_container_width=True, hide_index=True)

    def run(self) -> None:
        self.render_header()
        tab1, tab2, tab3, tab4 = st.tabs(["🚀 Cyber-Maszyna (Generator)", "📄 Archiwum Faktów", "🔥❄️ Cykle i Anomalie", "📈 Wektory Naciągu i Trajektorii"])
        with tab1:
            self.render_generator_tab()
        with tab2:
            self.render_archive_tab()
        with tab3:
            self.render_cycles_tab()
        with tab4:
            self.render_vectors_tab()


if __name__ == "__main__":
    LottoStreamlitApp().run()
