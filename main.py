"""Полное демо MindEngine.

Запуск:
    uv run main.py
"""
from __future__ import annotations

import time
import torch

from mind_engine import (
    MindEngine, Concept, Thought, KeywordEncoder, pick_device,
)


# ============================================================
# Утилиты вывода
# ============================================================

def hr(title: str = "") -> None:
    line = "─" * 70
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def fmt_focus(focus: dict, max_items: int = 6) -> str:
    items = focus["dominant"][:max_items]
    if not items:
        return "(пусто)"
    return ", ".join(f"{it['concept']}={it['activation']:+.3f}" for it in items)


# ============================================================
# Набор концептов и мыслей
# ============================================================

def build_concepts() -> list[Concept]:
    return [
        Concept("work",    "работа",      target=0.3, keywords=["работ", "делаю", "занят"]),
        Concept("rest",    "отдых",       target=0.5, keywords=["отдох", "отдых", "передох"]),
        Concept("stress",  "стресс",      target=0.1, keywords=["стресс", "тревог", "напряж"]),
        Concept("calm",    "спокойствие", target=0.5, keywords=["спокой", "умиротвор", "тихо"]),
        Concept("coffee",  "кофе",                    keywords=["кофе", "кофеин"]),
        Concept("morning", "утро",                    keywords=["утр", "проснул", "встал"]),
        Concept("evening", "вечер",                   keywords=["вечер", "закат"]),
        Concept("walk",    "прогулка",    target=0.4, keywords=["гул", "прогул", "ходи"]),
        Concept("focus",   "фокус",       target=0.5, keywords=["фокус", "сосредот", "вниман"]),
        Concept("fatigue", "усталость",   target=0.2, keywords=["устал", "утомл", "вымот"]),
        Concept("joy",     "радость",     target=0.5, keywords=["рад", "счаст", "весел"]),
        Concept("sadness", "грусть",      target=0.1, keywords=["груст", "печал", "тоск"]),
        Concept("hunger",  "голод",       target=0.2, keywords=["голод", "хочу есть", "проголод"]),
        Concept("food",    "еда",                     keywords=["ед", "поесть", "обед", "ужин"]),
    ]


def build_library() -> list[Thought]:
    return [
        Thought.make("Пойти погулять",      {"walk": 1.0, "calm": 0.5}, id="walk_out"),
        Thought.make("Ещё поработать",      {"work": 1.0, "focus": 0.6, "stress": 0.4}, id="work_more"),
        Thought.make("Отдохнуть",           {"rest": 1.0, "calm": 0.7, "fatigue": -0.3}, id="rest_now"),
        Thought.make("Выпить кофе",         {"coffee": 1.0, "morning": 0.4}, id="coffee_now"),
        Thought.make("Поесть",              {"food": 1.0, "hunger": -0.8, "joy": 0.3}, id="eat"),
        Thought.make("Проверить почту",     {"work": 0.7, "focus": 0.3, "stress": 0.2}, id="check_mail"),
        Thought.make("Поговорить с другом", {"joy": 1.0, "calm": 0.3, "stress": -0.4}, id="friend"),
    ]


# ============================================================
# Сценарии
# ============================================================

def scenario_1_basics(mind: MindEngine, enc: KeywordEncoder) -> None:
    hr("Сценарий 1: базовые мысли через think_text")
    for text in [
        "проснулся, хочу кофе",
        "много работы, чувствую стресс",
        "надо отдохнуть",
        "хочу погулять, но устал",
        "не хочу кофе",
        "не хочу пить кофе",
        "не хочу, но люблю кофе",
    ]:
        sig = enc(text)
        mind.think(Thought.make(content=text, signature=sig))
        print(f"  «{text}»")
        print(f"    сигнатура: {sig}")
        print(f"    фокус:     {fmt_focus(mind.current_focus(), 4)}")


def scenario_2_ambient(mind: MindEngine) -> None:
    hr("Сценарий 2: ambient (фоновое настроение)")
    mind.reset()
    mood = Thought.make("усталость", {"fatigue": 0.7, "rest": 0.4}, id="mood_tired")
    mind.set_ambient(mood)
    print(f"  ambient = {mood.content} {mood.signature}")

    with mind.branched():
        mind.think(Thought.make("ещё кофе", {"coffee": 1.0}))
        print(f"  с ambient:  {fmt_focus(mind.current_focus(), 4)}")

    mind.set_ambient(None)
    with mind.branched():
        mind.think(Thought.make("ещё кофе", {"coffee": 1.0}))
        print(f"  без ambient:{fmt_focus(mind.current_focus(), 4)}")


def scenario_3_probe(mind: MindEngine) -> None:
    hr("Сценарий 3: probe — «а что если подумать X?»")
    mind.reset()
    mind.think(Thought.make("хочу погулять, но устал",
                             {"walk": 1.0, "fatigue": 1.0}))

    before = mind.state.clone()
    print(f"  текущий фокус: {fmt_focus(mind.current_focus(), 4)}")

    trial = Thought.make("пойти погулять", {"walk": 1.0, "calm": 0.6})
    future = mind.probe(trial)
    print(f"  probe '{trial.content}': {fmt_focus(future, 4)}")

    assert torch.equal(before, mind.state), "probe изменил состояние!"
    print("  состояние не изменилось — ok")

    with mind.branched():
        mind.think(Thought.make("ещё поработать", {"work": 1.0, "stress": 0.5}))
        mind.think(Thought.make("выпить кофе",   {"coffee": 1.0}))
        print(f"  в ветке (work+coffee): {fmt_focus(mind.current_focus(), 4)}")
    print(f"  после выхода из ветки:  {fmt_focus(mind.current_focus(), 4)}")


def scenario_4_recommend(mind: MindEngine, library: list[Thought]) -> None:
    hr("Сценарий 4: рекомендации в трёх режимах")
    mind.reset()
    mind.think(Thought.make("устал, но надо работать",
                             {"fatigue": 0.8, "work": 0.6, "stress": 0.4}))
    for mode in ("resonance", "homeostasis", "mixed"):
        recs = mind.recommend(library, mode=mode, top_k=3)
        print(f"  mode={mode:12s}")
        for r in recs:
            print(f"    {r['thought'].content:22s}  score={r['score']:+.3f}")


def scenario_5_flow(mind: MindEngine, library: list[Thought]) -> None:
    hr("Сценарий 5: continue_thought — поток сознания")

    print("  без anti-rumination (видно залипание на 'кофе'):")
    mind.reset()
    mind.think(Thought.make("хочу кофе", {"coffee": 1.0}))
    print(f"    старт: {fmt_focus(mind.current_focus(), 4)}")
    for step in range(3):
        top = mind.continue_thought(library, mode="mixed")
        if top is None:
            break
        print(f"    шаг {step + 1}: «{top['thought'].content}»  score={top['score']:+.3f}")

    print("  с avoid_recent=2 (поток уходит с 'кофе'):")
    mind.reset()
    mind.think(Thought.make("хочу кофе", {"coffee": 1.0}))
    print(f"    старт: {fmt_focus(mind.current_focus(), 4)}")
    for step in range(4):
        top = mind.continue_thought(library, mode="mixed", avoid_recent=2)
        if top is None:
            break
        print(f"    шаг {step + 1}: «{top['thought'].content}»  score={top['score']:+.3f}")
        print(f"            фокус: {fmt_focus(mind.current_focus(), 4)}")


def scenario_6_reinforce(mind: MindEngine) -> None:
    hr("Сценарий 6: reinforce — Hebbian-обучение")
    mind.reset()

    def edge_strength(a_id: str, b_id: str) -> float:
        """How strongly thinking `a` activates `b` = assoc[b, a]."""
        ia, ib = mind._idx[a_id], mind._idx[b_id]
        if mind.mode == "low_rank":
            return float((mind.assoc_U[ib] * mind.assoc_V[ia]).sum())
        if mind.mode == "sparse":
            return float(mind.assoc[ib].to_dense()[ia])
        return float(mind.assoc[ib, ia])

    a = Thought.make("кофе", {"coffee": 1.0})
    b = Thought.make("утро", {"morning": 1.0})

    print(f"  coffee → morning (strength) до:    {edge_strength('coffee', 'morning'):+.4f}")
    for _ in range(20):
        mind.reinforce(a, b, lr=0.02)
    print(f"  coffee → morning (strength) после: {edge_strength('coffee', 'morning'):+.4f}")

    with mind.branched():
        mind.think(a)
        print(f"  после think('кофе') фокус: {fmt_focus(mind.current_focus(), 4)}")
        print("    (если reinforcement сработал, 'утро' теперь выше, чем до)")


def scenario_7_persistence(mind: MindEngine) -> None:
    hr("Сценарий 7: save / load (бинарь)")
    mind.reset()
    mind.think(Thought.make("хочу кофе", {"coffee": 1.0}))
    mind.think(Thought.make("устал", {"fatigue": 0.8, "rest": 0.4}))

    path = "demo_snapshot.pt"
    mind.save(path)

    loaded = MindEngine.load(path, device="auto")
    diff = (loaded.state.cpu() - mind.state.cpu()).abs().max().item()
    print(f"  сохранено в {path}")
    print(f"  загружено обратно, max|Δstate| = {diff:.2e}")
    print(f"  фокус до save:     {fmt_focus(mind.current_focus(), 4)}")
    print(f"  фокус после load:  {fmt_focus(loaded.current_focus(), 4)}")
    assert diff < 1e-5


def benchmark() -> None:
    hr("Сценарий 8: бенчмарк (think vs think_many)")

    def run(N: int, mode: str | None, T: int, batch: bool) -> float:
        concepts = [Concept(f"c{i}") for i in range(N)]
        eng = MindEngine(concepts, associations="random", seed=42, mode=mode)
        thoughts = [
            Thought.make(signature={
                f"c{(i * 7) % N}": 1.0,
                f"c{(i * 13 + 5) % N}": 0.5,
                f"c{(i * 31 + 11) % N}": 0.3,
            })
            for i in range(T)
        ]
        is_cuda = eng.device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if batch:
            eng.think_many(thoughts, chunk_size=512)
        else:
            for th in thoughts:
                eng.think(th)
        if is_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    configs = [
        (100,  "dense",    500),
        (500,  "low_rank", 500),
        (1000, "low_rank", 500),
        (5000, "low_rank", 500),
        (5000, "sparse",   500),
    ]
    print(f"  {'N':>6} {'mode':>9} {'T':>4} {'think ms':>11} {'think_many ms':>16} {'speedup':>8}")
    for N, mode, T in configs:
        run(N, mode, min(T, 20), batch=False)     # warm-up CUDA kernels
        t_seq = run(N, mode, T, batch=False)
        t_bat = run(N, mode, T, batch=True)
        print(f"  {N:>6} {mode:>9} {T:>4} {t_seq:>11.1f} {t_bat:>16.1f} {t_seq / t_bat:>7.1f}×")


# ============================================================
# Главный сценарий
# ============================================================

def main() -> None:
    print("=" * 70)
    print("  MindEngine — полное демо")
    print("=" * 70)
    print(f"  device: {pick_device('auto')}")
    print(f"  torch:  {torch.__version__}  cuda: {torch.cuda.is_available()}")

    concepts = build_concepts()
    library = build_library()
    encoder = KeywordEncoder(concepts)

    mind = MindEngine(concepts, associations="random", seed=42, forgetting=0.9, mode="dense")
    print(f"  концептов: {len(concepts)}  мыслей в библиотеке: {len(library)}")
    print(f"  режим: {mind.mode}  rank/avg_degree: {mind.rank or mind.avg_degree}")

    scenario_1_basics(mind, encoder)
    scenario_2_ambient(mind)
    scenario_3_probe(mind)
    scenario_4_recommend(mind, library)
    scenario_5_flow(mind, library)
    scenario_6_reinforce(mind)
    scenario_7_persistence(mind)
    benchmark()

    hr()
    print("  демо завершено.")
    print("=" * 70)


if __name__ == "__main__":
    main()