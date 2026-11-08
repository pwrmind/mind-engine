"""mscomplex_loader.py — загрузка MSComplexTasks и генерация сессий.

Поддерживает оба формата файла:
    task_subtask_dataset.json — JSON-массив (начинается с '[')
    task_subtask_data.json    — JSONL (построчный JSON)

Идентификаторы (id, parent, child) приводятся к строкам для
согласованности при построении графа зависимостей.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional


def load_mscomplex(path: str = "task_subtask_dataset.json") -> list[dict]:
    """Загружает датасет, автоопределяя формат: JSON-массив или JSONL."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} не найден. Скачайте task_subtask_dataset.json из "
            f"https://github.com/microsoft/MSComplexTasks"
        )

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # JSON-массив
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: ожидался список, получен {type(data).__name__}")
        return data

    # JSONL (построчный JSON)
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _topological_sort(
    subtask_ids: list[str],
    edges: list[tuple[str, str]],
) -> Optional[list[str]]:
    """Kahn's algorithm. Возвращает порядок или None, если есть цикл."""
    indeg: dict[str, int] = {sid: 0 for sid in subtask_ids}
    children: dict[str, list[str]] = defaultdict(list)

    for parent, child in edges:
        if parent not in indeg or child not in indeg:
            continue
        indeg[child] += 1
        children[parent].append(child)

    queue = deque([sid for sid in subtask_ids if indeg[sid] == 0])
    order: list[str] = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for ch in children[cur]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                queue.append(ch)

    if len(order) != len(subtask_ids):
        return None  # цикл
    return order


def build_sessions_from_mscomplex(
    entries: list[dict],
    *,
    min_subtasks: int = 2,
    max_subtasks: int = 8,
    require_order: bool = True,
    verbose: bool = True,
) -> list[list[str]]:
    """Для каждой задачи с DAG строит сессию через топологическую сортировку.

    Все идентификаторы приводятся к строкам, чтобы parent/child
    совпадали с id подзадач независимо от исходного типа.
    """
    sessions: list[list[str]] = []
    skipped_no_order = 0
    skipped_short = 0
    skipped_cycle = 0
    skipped_no_deps = 0

    for entry in entries:
        if require_order and str(entry.get("subtask_order_matters", "")).strip() != "Yes":
            skipped_no_order += 1
            continue

        subtasks = entry.get("subtasks") or []
        if len(subtasks) < min_subtasks:
            skipped_short += 1
            continue

        # id -> name, все id как строки
        id_to_name: dict[str, str] = {}
        for s in subtasks:
            if "id" not in s or "name" not in s:
                continue
            id_to_name[str(s["id"])] = s["name"]

        deps = entry.get("subtask_dependencies") or []
        if not deps:
            skipped_no_deps += 1
            continue

        edges = [
            (str(d["parent"]), str(d["child"])) for d in deps
            if isinstance(d, dict) and "parent" in d and "child" in d
        ]
        order = _topological_sort(list(id_to_name.keys()), edges)
        if order is None:
            skipped_cycle += 1
            continue

        session = [id_to_name[sid] for sid in order if sid in id_to_name]
        if len(session) < min_subtasks:
            continue
        if len(session) > max_subtasks:
            session = session[:max_subtasks]

        sessions.append(session)

    if verbose:
        print(f"  MSComplexTasks: {len(entries)} задач всего")
        print(f"    пропущено (порядок не важен): {skipped_no_order}")
        print(f"    пропущено (< {min_subtasks} подзадач): {skipped_short}")
        print(f"    пропущено (нет зависимостей): {skipped_no_deps}")
        print(f"    пропущено (цикл в DAG): {skipped_cycle}")
        print(f"    сессий построено: {len(sessions)}")
        if sessions:
            avg_len = sum(len(s) for s in sessions) / len(sessions)
            print(f"    средняя длина сессии: {avg_len:.1f}")
    return sessions


def print_example_sessions(sessions: list[list[str]], n: int = 3) -> None:
    print("\n  Примеры сессий (порядок подзадач из DAG):")
    for s in sessions[:n]:
        print(f"    {' → '.join(s)}")


if __name__ == "__main__":
    entries = load_mscomplex()
    sessions = build_sessions_from_mscomplex(entries)
    print_example_sessions(sessions)