"""兼容性反例缩减引擎单元测试。

覆盖：三类算子按固定顺序的接受/拒绝、方向保持（退化到另一条破坏规则仍
算保持方向）、非指定方向四类报错、根字符串/嵌套字符串缩短、序列化长度
严格递减、校验次数上限提前结束（只返回已验证结果）、入参不被修改、
确定性（无随机源、重复运行一致）。
"""
import json

import pytest

from app import compat, shrink


# ---------------------------------------------------------------------------
# 方向分类报错
# ---------------------------------------------------------------------------


def test_raises_when_document_valid_under_both():
    old = {"type": "object"}
    new = {"type": "object", "required": ["status"]}
    with pytest.raises(shrink.NotDirectionalCounterexample) as exc:
        shrink.shrink_counterexample(old, new, {"status": 1}, "backward")
    assert exc.value.classification == "valid"
    assert "均校验通过" in exc.value.message


def test_raises_when_document_rejected_by_both():
    old = {"type": "object"}
    new = {"type": "object", "required": ["status"]}
    with pytest.raises(shrink.NotDirectionalCounterexample) as exc:
        shrink.shrink_counterexample(old, new, 42, "backward")
    assert exc.value.classification == "invalid_both"
    assert "同时拒绝" in exc.value.message


def test_raises_when_direction_is_opposite():
    # 旧必填 email、新版无约束：{} 是 forward 实锤，请求 backward 应拒绝
    old = {"type": "object", "required": ["email"]}
    new = {"type": "object"}
    with pytest.raises(shrink.NotDirectionalCounterexample) as exc:
        shrink.shrink_counterexample(old, new, {}, "backward")
    assert exc.value.classification == "forward"
    assert exc.value.direction == "backward"
    assert "并非请求的 backward 方向" in exc.value.message


def test_forward_direction_accepts_forward_counterexample():
    old = {"type": "object", "required": ["email"], "additionalProperties": False}
    new = {"type": "object"}
    result = shrink.shrink_counterexample(
        old, new, {"name": "bob", "junk": [1, 2, 3]}, "forward")
    assert result["reduced_document"] == {}
    assert result["limit_reached"] is False


# ---------------------------------------------------------------------------
# 删除对象属性
# ---------------------------------------------------------------------------


def test_delete_object_properties_until_required_floor():
    # 新版必填 status，但文档缺 status（旧版无约束 -> 旧合法）；
    # 缩减只能删除旁支属性，最终稳定在 {}
    old = {"type": "object"}
    new = {"type": "object", "required": ["status"]}
    document = {"extra": {"n": [1, 2]}, "noise": "hello"}
    result = shrink.shrink_counterexample(old, new, document, "backward")

    assert result["reduced_document"] == {}
    assert all(step["operation"] == "delete_object_property" for step in result["steps"])
    assert document == {"extra": {"n": [1, 2]}, "noise": "hello"}


def test_required_property_in_old_schema_is_retained():
    # 旧必填 id、新必填 id+status：文档缺 status（backward）；
    # id 必须保留（删掉会被旧版拒绝），旁支与缺省的 status 无关
    old = {"type": "object", "required": ["id"]}
    new = {"type": "object", "required": ["id", "status"]}
    document = {"id": 7, "noise": "x", "junk": [1, 2]}
    result = shrink.shrink_counterexample(old, new, document, "backward")
    assert result["reduced_document"] == {"id": 7}
    assert [step["pointer"] for step in result["steps"]] == ["#/noise", "#/junk"]


def test_object_keys_deleted_in_dfs_insertion_order():
    # 新必填 a/b，文档只含 c/d/nested，缺 a/b（backward）
    old = {"type": "object"}
    new = {"type": "object", "required": ["a", "b"]}
    document = {"c": 1, "d": 2, "nested": {"x": 1}}
    result = shrink.shrink_counterexample(old, new, document, "backward")
    assert result["reduced_document"] == {}
    # 第一轮从最外层插入顺序开始：c -> d -> nested(整个属性一步删除)
    assert [step["pointer"] for step in result["steps"]] == [
        "#/c", "#/d", "#/nested"]


# ---------------------------------------------------------------------------
# 删除数组元素
# ---------------------------------------------------------------------------


def test_delete_array_elements_keeps_direction_and_old_validity():
    # 旧 minItems 3、新 minItems 5：4 个元素满足旧下限但不满足新下限（backward）
    old = {"type": "array", "minItems": 3}
    new = {"type": "array", "minItems": 5}
    document = [1, 2, 3, 4]
    result = shrink.shrink_counterexample(old, new, document, "backward")
    # 下标从小到大尝试：删下标 0 后仍 backward（3 个元素满足旧下限、违反新下限）
    assert result["reduced_document"] == [2, 3, 4]
    assert [step["pointer"] for step in result["steps"]] == ["#/0"]
    assert document == [1, 2, 3, 4]


def test_array_element_removal_blocked_when_old_schema_rejects():
    old = {"type": "array", "minItems": 4}
    new = {"type": "array", "minItems": 5}
    document = [1, 2, 3, 4]
    result = shrink.shrink_counterexample(old, new, document, "backward")
    # 一个元素都不能删（删掉后旧版也拒绝），文档已最小
    assert result["reduced_document"] == [1, 2, 3, 4]
    assert result["steps"] == []


# ---------------------------------------------------------------------------
# 缩短字符串
# ---------------------------------------------------------------------------


def test_shorten_root_string_one_char_each_step():
    # 长度 4：旧 minLength 3 接受、新 minLength 5 拒绝；下限 3
    old = {"type": "string", "minLength": 3}
    new = {"type": "string", "minLength": 5}
    result = shrink.shrink_counterexample(old, new, "abcd", "backward")
    assert result["reduced_document"] == "abc"
    assert all(step["operation"] == "shorten_string" and step["pointer"] == "#"
               for step in result["steps"])
    assert [step["length_after"] for step in result["steps"]] == [5]


def test_shorten_nested_string_value():
    old = {"type": "object", "properties": {"s": {"type": "string", "minLength": 2}}}
    new = {"type": "object", "properties": {"s": {"type": "string", "minLength": 5}}}
    document = {"s": "abcd"}
    result = shrink.shrink_counterexample(old, new, document, "backward")
    assert result["reduced_document"] == {"s": "ab"}
    assert [step["pointer"] for step in result["steps"]] == ["#/s"] * 2


def test_string_keys_are_never_shortened():
    # 只有字符串"值"可缩短；键名不允许动（JSON 不支持对象键路径编辑）
    old = {"type": "object"}
    new = {"type": "object", "required": ["other"]}
    document = {"abcdef": 1}
    result = shrink.shrink_counterexample(old, new, document, "backward")
    # 删除属性后文档为 {}，仍因缺 other 失败，方向保持；键名从未被缩短
    assert result["reduced_document"] == {}
    assert [step["operation"] for step in result["steps"]] == [
        "delete_object_property"]
    assert result["steps"][0]["pointer"] == "#/abcdef"


# ---------------------------------------------------------------------------
# 算子固定顺序
# ---------------------------------------------------------------------------


def test_operations_tried_in_fixed_order_property_then_element_then_string():
    # 旧/新都 required a、s（两个属性都不可删）；新版额外要求 a.minItems 4、
    # s.minLength 4。文档 a=[1,2]（缺额 backward）、s="xxxxx"（长度
    # backward）；旁支属性 b 最先删；a 只有 2 个元素（旧下限），没有可删
    # 元素；之后进入字符串阶段，逐字符缩短 s 到旧下限 "x"
    old = {"type": "object", "required": ["a", "s"], "properties": {
        "a": {"type": "array", "minItems": 2},
        "s": {"type": "string", "minLength": 1}}}
    new = {"type": "object", "required": ["a", "s"], "properties": {
        "a": {"type": "array", "minItems": 4},
        "s": {"type": "string", "minLength": 4}}}
    document = {"a": [1, 2], "b": "drop", "s": "xxxxx"}
    result = shrink.shrink_counterexample(old, new, document, "backward")

    operations = [(step["operation"], step["pointer"]) for step in result["steps"]]
    assert operations[0] == ("delete_object_property", "#/b")
    # 之后没有可接受的数组删除；全部剩余操作都是缩短 #/s
    assert operations[1:] == [("shorten_string", "#/s")] * 4
    assert result["reduced_document"] == {"a": [1, 2], "s": "x"}


def test_property_deletion_preferred_over_array_and_string_in_same_round():
    # 旧/新都 required a、s；新版额外要求 a.minItems 2、s.minLength 3。
    # 文档 a 有 4 个元素（合法）、s 长 2（新拒绝 -> backward）、旁支 extra 可删；
    # 第一轮三类算子都有候选时，删除对象属性 #/extra 最先被接受
    old = {"type": "object", "required": ["a", "s"],
           "properties": {"a": {"type": "array"}, "s": {"type": "string"}}}
    new = {"type": "object", "required": ["a", "s"],
           "properties": {"a": {"type": "array", "minItems": 2},
                          "s": {"type": "string", "minLength": 3}}}
    document = {"a": [1, 2, 3, 4], "s": "ab", "extra": 9}
    result = shrink.shrink_counterexample(old, new, document, "backward")
    # 第一轮最先接受的一定是删除对象属性 #/extra
    assert result["steps"][0]["operation"] == "delete_object_property"
    assert result["steps"][0]["pointer"] == "#/extra"
    # 缩减全程保持 backward：旧版接受、新版拒绝
    assert next(compat.make_validator(old).iter_errors(result["reduced_document"]), None) is None
    assert next(compat.make_validator(new).iter_errors(result["reduced_document"]), None) is not None


# ---------------------------------------------------------------------------
# 长度、步数与终止
# ---------------------------------------------------------------------------


def test_every_step_strictly_decreases_serialized_length():
    # 旧/新都要求 a、s；新 a.minItems 2、s.minLength 4。
    # 文档 a 单元素（backward）、s 长 5（满足新约束）；b 为可删旁支
    old = {"type": "object", "required": ["a", "s"], "properties": {
        "a": {"type": "array"}, "s": {"type": "string"}}}
    new = {"type": "object", "required": ["a", "s"], "properties": {
        "a": {"type": "array", "minItems": 2},
        "s": {"type": "string", "minLength": 4}}}
    document = {"a": [1], "s": "xxxxx", "b": [9]}
    result = shrink.shrink_counterexample(old, new, document, "backward")
    for step in result["steps"]:
        assert step["length_after"] < step["length_before"]
    assert result["original_length"] == len(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")))
    assert result["reduced_length"] == len(
        json.dumps(result["reduced_document"], ensure_ascii=False,
                   separators=(",", ":")))


def test_irreducible_document_returns_empty_steps():
    old = {"type": "object"}
    new = {"type": "object", "required": ["x"]}
    result = shrink.shrink_counterexample(old, new, {}, "backward")
    assert result["reduced_document"] == {}
    assert result["steps"] == []
    assert result["limit_reached"] is False
    assert result["validations"] == 1


def test_scalar_non_string_roots_are_irreducible():
    # 42.5：旧 type number 接受，新 integer 拒绝（backward）；三类算子无候选
    old = {"type": "number"}
    new = {"type": "integer"}
    result = shrink.shrink_counterexample(old, new, 42.5, "backward")
    assert result["reduced_document"] == 42.5
    assert result["steps"] == []


def test_limit_reached_returns_last_validated_document():
    # 旧/新都要求字符串：旧 minLength 90、新 minLength 100；95 个字符的文档
    # 旧合法新拒绝（backward），每步缩 1 字符，自然下限为 90（需 5 步）。
    # 把校验上限压到 4（初始 1 次 + 最多 3 步）强制提前结束，
    # 返回 92 字符的已验证文档
    old = {"type": "string", "minLength": 90}
    new = {"type": "string", "minLength": 100}
    result = shrink.shrink_counterexample(
        old, new, "x" * 95, "backward", max_validations=4)
    assert result["limit_reached"] is True
    assert result["validations"] == 4
    assert result["max_validations"] == 4
    assert result["reduced_document"] == "x" * 92
    # 返回文档确实仍满足指定方向
    assert next(compat.make_validator(old).iter_errors(result["reduced_document"]), None) is None
    assert next(compat.make_validator(new).iter_errors(result["reduced_document"]), None) is not None


def test_validation_count_includes_initial_classification():
    # 新必填 x、文档缺 x（backward）；两个旁支属性各删一次
    old = {"type": "object"}
    new = {"type": "object", "required": ["x"]}
    result = shrink.shrink_counterexample(old, new, {"y": 2, "z": 3}, "backward")
    # 初始 1 次 + 删除 y、z 各 1 次
    assert result["validations"] == 3


def test_rejected_candidates_still_consume_validation_budget():
    # 旧 minItems 4、新 minItems 5：4 元素文档删任何元素都会被旧版拒绝，
    # 每个删除候选都要重验一次（计入预算）；上限极小 -> 提前结束
    old = {"type": "array", "minItems": 4}
    new = {"type": "array", "minItems": 5}
    document = [1, 2, 3, 4]
    result = shrink.shrink_counterexample(
        old, new, document, "backward", max_validations=3)
    assert result["limit_reached"] is True
    assert result["validations"] == 3
    # 所有候选都被拒绝 -> 缩减文档等于原文
    assert result["reduced_document"] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 只读 / 确定性
# ---------------------------------------------------------------------------


def test_input_document_is_never_mutated():
    # 新必填 b，文档含 b? 不含：文档 {a:[...], c,d} 缺 b（backward）
    old = {"type": "object", "required": ["a"]}
    new = {"type": "object", "required": ["a", "b"]}
    document = {"a": [1, 2, 3], "c": "xx", "d": {"e": 1}}
    snapshot = json.dumps(document, sort_keys=True)
    shrink.shrink_counterexample(old, new, document, "backward")
    assert json.dumps(document, sort_keys=True) == snapshot


def test_deterministic_across_runs():
    # 新必填 missing + a.minItems 3 + s.minLength 4；文档缺 missing（backward），
    # 同时含可删属性、可删数组元素、可缩短字符串
    old = {"type": "object"}
    new = {"type": "object", "required": ["missing"], "properties": {
        "a": {"type": "array", "minItems": 3},
        "s": {"type": "string", "minLength": 4}}}
    document = {"z": 1, "a": [1, 2, 3, 4, 5], "s": "abcdef", "m": [1, {"k": 2}]}
    first = shrink.shrink_counterexample(old, new, document, "backward")
    second = shrink.shrink_counterexample(old, new, document, "backward")
    assert first["reduced_document"] == second["reduced_document"]
    assert first["steps"] == second["steps"]
    assert first["validations"] == second["validations"]
