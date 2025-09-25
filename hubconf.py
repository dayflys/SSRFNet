# hubconf.py

import torch
import torch.nn as nn

# 필수: 이 레포에서 제공하는 엔트리포인트 함수 목록
dependencies = ["torch", "torchvision"]

def _fix_state_dict(state_dict):
    """모델의 state_dict 키를 수정하여 호환성을 유지합니다."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("student_model."):
            new_key = k[len("student_model.") :]
        elif k.startswith("classifier."):
            new_key = k[len("classifier.") :]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def _ensure_sys_path():
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent
    extra_paths = [
        repo_root,
        repo_root / "experiments" / "eval_only" / "test_code",
        repo_root / "experiments" / "train" / "train_code",
    ]

    for path in extra_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_ensure_sys_path()

try:  # 허브에서 로드할 때 data_augmentation 경로를 미리 확보
    import data_augmentation  # noqa: F401
except ModuleNotFoundError:
    pass


def SSRFNet_svmixer(pretrained=False, **kwargs):
    _ensure_sys_path()
    from experiments.eval_only.test_code.models.svmixer import SVMixer

    model = SVMixer(12, 149, 1024)

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/dayflys/SSRFNet/releases/download/v1.0.1/ssrfnet_eer0.60_sv_mixer_state_dict.pt",
            map_location="cpu"
        )
        ckpt = _fix_state_dict(ckpt)
        model.load_state_dict(ckpt)
    return model


def SSRFNet_backend(pretrained=False, **kwargs):
    _ensure_sys_path()
    from experiments.eval_only.test_code.models.redimnet import ReDimNetWrap

    model = ReDimNetWrap(
        F=64,
        C=16,
        embed_dim=192,
        insert_feature_num=4,
        num_hidden_layers=12,
        merge_layer_num=4
    )

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/dayflys/SSRFNet/releases/download/v1.0.1/ssrfnet_eer0.60_classifier_state_dict.pt",
            map_location="cpu"
        )
        ckpt = _fix_state_dict(ckpt)
        model.load_state_dict(ckpt)
    return model

