# GPU Backend (GPU 加速)

## 概述
GPU Backend 模块提供 CUDA GPU 加速支持，用于并行因子评估，包含设备管理、张量转换和批量执行。

## DeviceManager

单例风格的设备选择助手：

```python
class DeviceManager:
    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = self._select_device()
        return self._device
    
    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # Apple Silicon
            return torch.device("mps")
        return torch.device("cpu")
    
    @property
    def is_gpu(self) -> bool:
        return self.device.type in ("cuda", "mps")
```

## 设备优先级

1. **CUDA**: NVIDIA GPU（最快）
2. **MPS**: Apple Silicon GPU
3. **CPU**: 备用

## 张量转换

```python
def to_gpu(arr: np.ndarray) -> torch.Tensor:
    """NumPy 数组转换为 GPU 张量"""
    return torch.from_numpy(arr).to(device)

def to_cpu(tensor: torch.Tensor) -> np.ndarray:
    """GPU 张量转换回 NumPy 数组"""
    return tensor.cpu().numpy()
```

## 批量执行

```python
def batch_eval(
    expressions: List[ExpressionTree],
    data: Dict[str, np.ndarray],
    batch_size: int = 32
) -> List[np.ndarray]:
    """批量评估表达式树（GPU 加速）"""
```

## 自动 CPU 回退

当 GPU 不可用时自动回退到 CPU：

```python
try:
    result = gpu_compute(expression, data)
except RuntimeError:
    result = cpu_compute(expression, data)  # 自动回退
```

## 性能对比

| Backend | 1000 因子 | 10000 因子 |
|---------|-----------|------------|
| CPU (numpy) | ~30s | ~300s |
| GPU (CUDA) | ~2s | ~20s |
| Apple MPS | ~5s | ~50s |

## 配置

```yaml
evaluation:
  backend: "gpu"       # gpu/numpy/c
  gpu_device: "cuda:0" # 设备 ID
```

## 限制

- GPU extra 仅支持 Linux（cupy-cuda12x）
- macOS 仅支持 MPS 或 CPU
- 内存限制：大型因子矩阵可能超出 GPU 显存

## 来源
- `factorminer/operators/gpu_backend.py`
