"""MOABB Trial 数据读取与模型输入维度适配。"""
"""
创建数据集对象
make_dataset() → LocalBNCI2014_001

从本地读取 Trial
check_local_data() → _get_single_subject_data() → paradigm.get_data()

转换神经网络输入维度
adapt_input() → (B, 1, C, T) 或 (B, C, T)
"""
from pathlib import Path
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.datasets.bnci.bnci_2014 import _load_data_001_2014
from moabb.paradigms import MotorImagery


class LocalBNCI2014_001(BNCI2014_001):
    """沿用 MOABB 官方数据集读取逻辑，固定使用项目内已下载的数据。"""
    # 接收受试者列表和本地数据根目录。
    def __init__(self, subjects, data_root):
        super().__init__(subjects=subjects)
        self.data_root = Path(data_root)
    # 覆盖官方预取逻辑，避免此处预取 NEMAR 数据。
    def _prefetch_nemar_sourcedata(self, subjects, verbose=None):
        # 已逐个检查本地源文件，禁止 MOABB 在 auto 模式下预取 NEMAR。
        return None
    # 检查本地文件，随后调用 _load_data_001_2014() 加载当前受试者数据。
    
    def _get_single_subject_data(self, subject):
        check_local_data(self, [subject])
        # 返回值： MOABB 的嵌套字典，主要按照 Session 和 Run 组织数据，每个 Run 通常对应一个 MNE Raw 对象。 
        # 此处返回的是原始连续记录的组织结构，不是 (B,C,T) 的神经网络输入。
        return _load_data_001_2014(
            subject=subject,
            path=str(self.data_root),
            force_update=False,
            update_path=False,
            only_filenames=False,
            verbose=False,
        )


def make_dataset(subjects, data_root):
    """创建 BCI IV 2a 数据集；其他 MOABB 数据集可在此扩展。"""
    return LocalBNCI2014_001(
        subjects=subjects, data_root=data_root
    )


def check_local_data(dataset, subjects):
    """缺少任一会话文件时立即停止，避免 MOABB 尝试下载。"""
    folder = dataset.data_root / "MNE-bnci-data" / "~bci" / "database" / "001-2014"
    missing = [
        folder / f"A{subject:02d}{session}.mat"
        for subject in subjects
        for session in ("T", "E")
        if not (folder / f"A{subject:02d}{session}.mat").is_file()
    ]
    if missing:
        raise FileNotFoundError("本地 MOABB 数据缺失：\n" + "\n".join(map(str, missing)))


def load_trial_info(paradigm, dataset, subject):
    """
    读取一名受试者的完整 Trial，返回预检所需的形状与标签信息。
    epochs：切好的 Trial 数据及采样信息；
    labels：每个 Trial 的类别标签；
    metadata：每个 Trial 对应的受试者、会话等信息。
    """
    epochs, labels, metadata = paradigm.get_data(
        dataset, subjects=[subject], return_epochs=True
    )
    return {
        "shape": (len(epochs), len(epochs.ch_names), len(epochs.times)), # (Trial 数, 通道数, 每个 Trial 的时间点数)
        "sfreq": float(epochs.info["sfreq"]),
        "classes": sorted(str(label) for label in np.unique(labels)), # 出现的类别名称
        "counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        },
        "sessions": {
            str(session): int(count)
            for session, count in metadata["session"].value_counts().items()
        },
    }


def adapt_input(X, model_name):
    """MOABB 的 (B,C,T) 转为当前模型需要的维度。"""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"预期 MOABB Trial 形状 (B,C,T)，实际为 {X.shape}")

    
    canonical = X[:, None, :, :] # 新增一个维度

    if model_name in ("CustomEEGNet", "CustomEEGTCNet", "CustomEEGNex"):
        return canonical # 返回四个维度
    
    return canonical[:, 0, :, :] #返回三个维度
