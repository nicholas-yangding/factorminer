# Auto Inventor (自动算子发明)

## 概述
Auto Inventor 是 Phase 2 Helix 的可选组件，使用 LLM 引导的提议和验证流程自动发现新算子。

## 工作流程

```
1. LLM 提议 --> 2. 验证 --> 3. IC 贡献检查 --> 4. 注册
```

## ProposedOperator 数据类

```python
@dataclass
class ProposedOperator:
    name: str                    # 算子名称（如 "ExpDecayDiff"）
    arity: int                   # 参数个数（1 = 一元，2 = 二元）
    description: str             # 人类可读描述
    numpy_code: str              # Python 源码（compute 函数）
    param_names: Tuple[str, ...] # 额外数值参数名
    param_defaults: Dict[str, float]  # 参数默认值
    param_ranges: Dict[str, Tuple[float, float]]  # 参数有效范围
    rationale: str               # 为什么这个算子有用
    based_on: List[str]          # 启发它的已有算子
```

## ValidationResult 数据类

```python
@dataclass
class ValidationResult:
    valid: bool              # 是否通过所有验证
    error: str               # 错误信息
    output_shape_ok: bool    # 输出形状是否正确 (M, T)
    nan_ratio: float         # NaN 值比例
    ic_contribution: float   # IC 贡献度
```

## 验证流程

1. **语法验证**: 执行 numpy_code 检查语法错误
2. **形状验证**: 确保输出为 (M, T) 形状
3. **数值验证**: 检查 NaN 比例、可finite值
4. **IC 验证**: 在样本数据上计算 IC 贡献

## IC 贡献检查

```python
def check_ic_contribution(
    proposed_op: ProposedOperator,
    validation_result: ValidationResult,
    baseline_ic: float,
    ic_gain_threshold: float = 0.01
) -> bool:
    """检查新算子是否对 IC 有正向贡献"""
```

## 自动注册

验证通过后，新算子自动注册到 OPERATOR_REGISTRY：

```python
def register_proposed_operator(op: ProposedOperator) -> None:
    """将提议的算子注册到全局注册表"""
```

## 使用示例

```bash
# 启用 auto_inventor
factorminer --cpu helix --mock --auto-inventor -n 2 -b 8 -t 10
```

## 设计意图

- **LLM 引导**: 利用 LLM 的创造力发现新算子
- **沙箱验证**: 在安全环境中验证代码
- **增量贡献**: 新算子必须对 IC 有正向贡献
- **可解释**: 参数范围和原理清晰

## 来源
- `factorminer/operators/auto_inventor.py`
