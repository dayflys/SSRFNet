# hubconf.py

import torch
import torch.nn as nn

# 필수: 이 레포에서 제공하는 엔트리포인트 함수 목록
dependencies = ["torch", "torchvision"]

def small_svmixer(pretrained=False, **kwargs):
    from experiments.eval_only.test_code.models.svmixer import SVMixer

    model = SVMixer(12, 149, 1024)

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/dayflys/SSRFNet/releases/download/v1.0.1/ssrfnet_eer0.60_sv_mixer_state_dict.pt",
            map_location="cpu"
        )
        model.load_state_dict(ckpt)
    return model
    


def backend_module(pretrained=False, **kwargs):
    from experiments.eval_only.test_code.models.redimnet import ReDimNetWrap

    model = ReDimNetWrap(
        F=64, 
        C=16, 
        embed_dim=192,
        insert_feature_num= 4,
        num_hidden_layers = 12,
        merge_layer_num = 4
    )

    if pretrained:
        # 사전 학습된 가중치를 불러오기 (예: 릴리스에 올려둔 파일에서 다운로드)
        ckpt = torch.hub.load_state_dict_from_url(
            "https://github.com/dayflys/SSRFNet/releases/download/v1.0.1/ssrfnet_eer0.60_classifier_state_dict.pt",
            map_location="cpu"
        )
        model.load_state_dict(ckpt)
    return model
