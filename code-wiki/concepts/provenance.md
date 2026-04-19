# Provenance (溯源)

## 概述
Provenance 模块负责追踪因子和挖掘会话的来源信息，确保可复现性和审计追踪。

## 核心功能

### 因子溯源
每个因子都携带 provenance 信息：

```python
@dataclass
class Provenance:
    generator: str           # 生成器名称
    generated_at: float      # Unix timestamp
    parameters: dict         # 生成参数
    parent_factors: list     # 父因子（如果有）
    transformations: list   # 应用的变换
```

### 会话溯源
```python
@dataclass
class RunManifest:
    config_hash: str         # 配置哈希
    data_hash: str          # 数据哈希
    started_at: float       # 开始时间
    completed_at: float     # 完成时间
    factors_admitted: list  # 准入因子列表
    factors_rejected: list   # 拒绝因子列表
```

## 哈希计算

```python
def stable_digest(payload: Any) -> str:
    """计算 JSON 可序列化对象的稳定 SHA256 哈希"""
    normalized = _json_safe(payload)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
```

## JSON 安全转换

`_json_safe()` 将 numpy/Python 对象转换为 JSON 安全格式：

```python
def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    ...
```

## 因子引用列表

```python
def _compact_reference_list(entries, limit=8) -> List[str]:
    """将混合的因子引用列表规范化为可读字符串"""
```

## 记忆信号压缩

```python
def _compact_memory_signal(memory_signal: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """保留最实用的记忆上下文片段"""
```

## 用途

1. **可复现性**: 记录每次运行的完整配置和数据哈希
2. **审计**: 追踪因子的生成来源和历史
3. **对比**: 相同配置/数据运行应产生相同结果

## 来源
- `factorminer/core/provenance.py`
