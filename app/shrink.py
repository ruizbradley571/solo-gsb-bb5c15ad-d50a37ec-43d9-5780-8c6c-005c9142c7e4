"""兼容性反例缩减引擎（纯只读）。

接收**一篇**已确认属于某个单向破坏类别的 JSON 文档，用分析记录保存的
**旧版 / 新版 schema** 反复做双版校验，并在保持该破坏方向的前提下，
贪心式地缩减文档的 JSON 序列化长度，得到更小的最小化反例。

方向语义与回放诊断（``app.replay``）一致：

* ``backward``：旧版合法、新版拒绝（向后破坏实锤）；
* ``forward``：新版合法、旧版拒绝（向前破坏实锤）。

文档不属于指定方向（两版都合法、两版都拒绝或方向相反）时抛出
:class:`NotDirectionalCounterexample`，由接口层转为明确的 ``422`` 错误。

缩减算子按**固定顺序**枚举，每一轮：

1. ``delete_object_property``：按 DFS 先序遍历删除对象属性
   （对象键按插入顺序，根对象优先）；
2. ``delete_array_element``：按 DFS 先序遍历删除数组元素
   （下标从小到大）；
3. ``shorten_string``：按 DFS 先序遍历缩短字符串值（删除末字符，
   每次一个字符；不改动对象键名）。

每个候选只在满足以下两条时才被接受：

* 候选文档的 JSON 序列化长度（``json.dumps`` 紧凑形式）**严格变小**；
* 对旧 / 新两版 schema 重新校验后**仍保持指定方向**。

接受一个候选后从头开始下一轮枚举；一整轮没有任何候选可接受时停止
（已最小化），或在双版校验尝试次数达到 :data:`MAX_VALIDATIONS` 上限时
提前停止——即使因上限结束，也只返回**最后一次通过验证**的文档。

引擎为纯函数：不读取数据库、不修改入参文档（所有编辑都在深拷贝上
进行）、不做任何持久化。
"""
from __future__ import annotations

import copy
import json
from typing import Any, Iterator

from app.compat import escape_pointer, make_validator

# 一次缩减允许的最大双版校验尝试次数（初始分类 + 每个候选重验各计一次）。
MAX_VALIDATIONS = 1000

# 三类缩减算子，按固定尝试顺序排列。
OP_DELETE_OBJECT_PROPERTY = "delete_object_property"
OP_DELETE_ARRAY_ELEMENT = "delete_array_element"
OP_SHORTEN_STRING = "shorten_string"
OPERATION_ORDER = (
    OP_DELETE_OBJECT_PROPERTY,
    OP_DELETE_ARRAY_ELEMENT,
    OP_SHORTEN_STRING,
)


class NotDirectionalCounterexample(ValueError):
    """文档不属于指定的单向破坏类别。

    ``classification`` 为 ``valid``（两版均合法）/ ``invalid_both``
    （两版均拒绝）/ ``backward`` / ``forward``（与请求方向相反时）。
    """

    def __init__(self, direction: str, classification: str, message: str):
        self.direction = direction
        self.classification = classification
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# 校验与分类
# ---------------------------------------------------------------------------


def _is_valid(validator: Any, document: Any) -> bool:
    return next(validator.iter_errors(document), None) is None


def classify_document(document: Any, old_validator: Any,
                      new_validator: Any) -> str:
    """用旧 / 新两版校验文档，返回四向分类之一。

    ``valid`` / ``backward``（旧合法新拒绝）/ ``forward``（新合法旧
    拒绝）/ ``invalid_both``（两版均拒绝）。
    """
    old_ok = _is_valid(old_validator, document)
    new_ok = _is_valid(new_validator, document)
    if old_ok and new_ok:
        return "valid"
    if old_ok:
        return "backward"
    if new_ok:
        return "forward"
    return "invalid_both"


def _matches_direction(classification: str, direction: str) -> bool:
    return classification == direction


# ---------------------------------------------------------------------------
# JSON 序列化长度 / 指针
# ---------------------------------------------------------------------------


def _serialized_length(document: Any) -> int:
    """紧凑 JSON 序列化长度（``ensure_ascii=False``，与键序无关地逐值比较）。"""
    return len(json.dumps(document, ensure_ascii=False, separators=(",", ":")))


def _append_pointer(pointer: str, token: Any) -> str:
    segment = escape_pointer(str(token))
    return f"#/{segment}" if pointer == "#" else f"{pointer}/{segment}"


# ---------------------------------------------------------------------------
# 候选算子枚举与应用
# ---------------------------------------------------------------------------


def _walk_nodes(value: Any, pointer: str = "#",
                segments: tuple[Any, ...] = ()) -> Iterator[
                    tuple[Any, str, tuple[Any, ...]]]:
    """DFS 先序遍历文档节点，产出 (节点, 实例指针, 根到该节点的路径段)。

    路径段中对象键为字符串、数组下标为整数；根节点的路径段为空元组。
    """
    yield value, pointer, segments
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_nodes(item, _append_pointer(pointer, key),
                                   segments + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_nodes(item, _append_pointer(pointer, index),
                                   segments + (index,))


def _apply_edit(document: Any, segments: tuple[Any, ...],
                operation: str, token: Any) -> Any:
    """在文档深拷贝上沿 ``segments`` 定位容器并施加单次编辑。

    ``token`` 为被删对象键、被删数组下标，或缩短字符串时其所在容器的
    访问键 / 下标（根字符串没有容器，由调用方直接切片处理）。
    """
    candidate = copy.deepcopy(document)
    target: Any = candidate
    for segment in segments:
        target = target[segment]

    if operation == OP_DELETE_OBJECT_PROPERTY:
        del target[token]
    elif operation == OP_DELETE_ARRAY_ELEMENT:
        target.pop(token)
    else:  # OP_SHORTEN_STRING：token 为字符串所在容器的访问键 / 下标
        target[token] = target[token][:-1]
    return candidate


def _iter_candidates(document: Any, operation: str) -> Iterator[dict]:
    """按 DFS 先序枚举某一类算子的全部候选（延迟生成，含编辑后深拷贝）。"""
    for node, pointer, segments in _walk_nodes(document):
        if operation == OP_DELETE_OBJECT_PROPERTY and isinstance(node, dict):
            for key in node:
                child_pointer = _append_pointer(pointer, key)
                yield {
                    "operation": operation,
                    "pointer": child_pointer,
                    "candidate": _apply_edit(document, segments, operation, key),
                }
        elif operation == OP_DELETE_ARRAY_ELEMENT and isinstance(node, list):
            for index in range(len(node)):
                yield {
                    "operation": operation,
                    "pointer": _append_pointer(pointer, index),
                    "candidate": _apply_edit(document, segments, operation, index),
                }
        elif operation == OP_SHORTEN_STRING and isinstance(node, str) and node:
            if not segments:
                # 根节点本身就是字符串：没有容器，直接返回截掉末字符的副本
                candidate = document[:-1]
            else:
                token = segments[-1]
                candidate = _apply_edit(document, segments[:-1], operation, token)
            yield {
                "operation": operation,
                "pointer": pointer,
                "candidate": candidate,
            }


# ---------------------------------------------------------------------------
# 缩减主流程
# ---------------------------------------------------------------------------


def shrink_counterexample(old_schema: Any, new_schema: Any, document: Any,
                          direction: str,
                          max_validations: int = MAX_VALIDATIONS) -> dict:
    """把单篇单向破坏反例缩减为保持方向的更小文档（纯函数）。

    返回原文、缩减后文档、逐步操作路径（含每步前后序列化长度）、
    实际双版校验次数以及是否因达到上限提前结束。文档不属于
    ``direction`` 时抛出 :class:`NotDirectionalCounterexample`。
    """
    old_validator = make_validator(old_schema)
    new_validator = make_validator(new_schema)

    validations = 1  # 初始分类计一次双版校验
    classification = classify_document(document, old_validator, new_validator)
    if not _matches_direction(classification, direction):
        if classification == "valid":
            message = "文档对旧版与新版 schema 均校验通过，不属于单向破坏反例"
        elif classification == "invalid_both":
            message = "文档被旧版与新版 schema 同时拒绝，无法归因于单向破坏"
        else:
            message = (
                f"文档属于 {classification} 方向破坏，"
                f"并非请求的 {direction} 方向反例"
            )
        raise NotDirectionalCounterexample(direction, classification, message)

    current = copy.deepcopy(document)
    current_length = _serialized_length(current)
    steps: list[dict] = []
    limit_reached = False

    while True:
        accepted = False
        for operation in OPERATION_ORDER:
            if accepted:
                break
            for trial in _iter_candidates(current, operation):
                candidate = trial["candidate"]
                candidate_length = _serialized_length(candidate)
                # 只接受序列化长度严格变小的候选；未变短则不消耗重验
                if candidate_length >= current_length:
                    continue
                if validations >= max_validations:
                    limit_reached = True
                    break
                validations += 1
                candidate_class = classify_document(
                    candidate, old_validator, new_validator)
                if not _matches_direction(candidate_class, direction):
                    continue
                steps.append({
                    "operation": operation,
                    "pointer": trial["pointer"],
                    "length_before": current_length,
                    "length_after": candidate_length,
                })
                current = candidate
                current_length = candidate_length
                accepted = True
                break
            if limit_reached:
                break
        if not accepted:
            break

    return {
        "direction": direction,
        "original_document": document,
        "reduced_document": current,
        "original_length": _serialized_length(document),
        "reduced_length": current_length,
        "steps": steps,
        "validations": validations,
        "max_validations": max_validations,
        "limit_reached": limit_reached,
    }
