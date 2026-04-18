# Factor Library (因子库)

## 概述
FactorLibrary 是因子的集合管理器，负责添加、查询、相关性维护。内置 110 个论文因子。

## 数据结构

### Factor
```python
@dataclass
class Factor:
    id: str              # 唯一标识
    name: str             # 名称
    expression: str        # DSL 公式
    tree: ExpressionTree   # 表达式树
    admitted_at: float     # Unix timestamp
    provenance: dict       # 来源信息
    ic_series: np.ndarray # IC 时间序列
    stats: dict           # 统计指标
    signals: np.ndarray   # 信号矩阵 (可选)
```

### FactorLibrary
```python
class FactorLibrary:
    def __init__(
        self,
        ic_threshold: float = 0.02,
        icir_threshold: float = 0.5,
        correlation_threshold: float = 0.5
    ):
        self._factors: Dict[str, Factor] = {}
        self._id_to_index: Dict[str, int] = {}
        self.correlation_matrix: np.ndarray = None
    
    def add(self, factor: Factor) -> bool:
        """添加因子，返回是否成功"""
    
    def list_factors(self) -> List[Factor]:
        """列出所有因子"""
    
    def get_factor(self, id: str) -> Factor:
        """获取指定因子"""
```

## 相关性管理

入库时检查：
1. 新因子与库中因子的 IC 相关性
2. 如果相关但 IC 更高，可考虑替换

相关性矩阵维护：
```
rho(i, j) = corr(IC_i, IC_j)
```

## 110 个论文因子

内置在 `library_io.py` 的 `build_factor_miner_catalog()`：
- 来自论文附录
- 覆盖多种策略类型
- 可作为基线对比

## 序列化

保存格式：
- `<path>.json`: 元数据 + 因子定义
- `<path>_signals.npz`: 二进制信号缓存

## 来源
- `factorminer/core/factor_library.py`
- `factorminer/core/library_io.py`
