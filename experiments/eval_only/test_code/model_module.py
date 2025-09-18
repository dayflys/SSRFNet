import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from thop import profile
from utils import compute_eer, compute_min_dcf

class ModelModule(pl.LightningModule):
    def __init__(self, args, student_model, classifier):
        super().__init__()
        self.student_model = student_model
        self.classifier = classifier

        self.lr = args['lr']
        self.lr_patience = args['lr_patience']
        self.lr_gamma = args['lr_gamma']
        self.weight_decay = args['weight_decay']
        self.num_seg = args['num_seg']

        self._val_outputs = []
    
    def set_trials(self, name, trials):
        self.test_name = name
        self.trials = trials

    def forward(self, x):
        x = self.student_model(x)
        x = self.classifier(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        """
        체크포인트 로딩 시 불일치하는 가중치를 필터링합니다.
        """
        # 현재 모델의 state_dict
        model_state_dict = self.state_dict()
        
        # 불일치하는 키들을 필터링
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Skipping {key}: shape mismatch - checkpoint: {value.shape}, model: {model_state_dict[key].shape}")
            else:
                print(f"Skipping {key}: not found in model")
        
        # 필터링된 state_dict로 로딩
        return super().load_state_dict(filtered_state_dict, strict=False)

    def on_validation_epoch_start(self):
        self._val_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x_seg, keys = batch
        B = x_seg.size(0)
        x_seg = x_seg.to(dtype=torch.float32, device=self.device, non_blocking=True)
        x_seg = x_seg.view(B * self.num_seg, -1)
        x_seg_ssl = self.student_model(x_seg)
        emb = self.classifier(x_seg_ssl,x_seg)
        emb = emb.view(B, self.num_seg, -1)
        self._val_outputs.append({"keys": keys, "embeddings": emb})

    def on_validation_epoch_end(self):
        # 1. 모든 키와 embedding을 모음
        keys_list = []
        embeddings_list = []
        for batch_out in self._val_outputs:
            keys_list.extend(batch_out["keys"])
            embeddings_list.append(batch_out["embeddings"])

        all_keys_tensor = torch.stack(keys_list, dim=0)
        all_embeddings_tensor = torch.cat(embeddings_list, dim=0)

        # 2. 다중 GPU 환경이면 모든 프로세스의 결과 gather, 아니면 그대로 사용
        if self.trainer.world_size > 1:
            gathered_keys = self.all_gather(all_keys_tensor)
            gathered_embeddings = self.all_gather(all_embeddings_tensor)
        else:
            gathered_keys = all_keys_tensor
            gathered_embeddings = all_embeddings_tensor

        gathered_keys = gathered_keys.cpu().view(-1)
        gathered_embeddings = gathered_embeddings.cpu().view(-1, self.num_seg, gathered_embeddings.size(-1))

        # 3. 각 키(예: 스피커 ID)에 해당하는 임베딩을 embedding_dict에 저장 (메모리 효율적)
        embedding_dict = {}
        for k, emb in zip(gathered_keys, gathered_embeddings):
            embedding_dict[int(k.item())] = emb

        # 메모리 정리
        del gathered_keys, gathered_embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 4. 트라이얼을 chunk 단위로 처리 (메모리 효율성을 위해 크기 줄임)
        chunk_size = 10000  # 더 작은 chunk 크기로 메모리 사용량 줄임
        all_scores = []
        all_labels = []
        trial_chunk = []
        
        total_trials = len(self.trials)
        processed_trials = 0
        
        for trial in self.trials:
            # trial은 key1, key2, label 속성을 가진 객체라고 가정
            trial_chunk.append((trial.key1, trial.key2, trial.label))
            processed_trials += 1
            
            if len(trial_chunk) >= chunk_size:
                scores_chunk, labels_chunk = self.process_trial_chunk(trial_chunk, embedding_dict)
                all_scores.append(scores_chunk)
                all_labels.extend(labels_chunk)
                trial_chunk = []
                # 메모리 정리
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 진행 상황 출력 (매 10%마다)
                if processed_trials % (total_trials // 10) == 0:
                    print(f"Validation progress: {processed_trials}/{total_trials} trials processed")
        
        # 나머지 처리
        if len(trial_chunk) > 0:
            scores_chunk, labels_chunk = self.process_trial_chunk(trial_chunk, embedding_dict)
            all_scores.append(scores_chunk)
            all_labels.extend(labels_chunk)

        # 5. 모든 chunk의 결과 합치기
        if all_scores:
            scores = torch.cat(all_scores, dim=0)
        else:
            scores = torch.tensor([])
        
        # 6. 평가 지표 계산
        if len(scores) > 0:
            eer = compute_eer(scores, all_labels)
            min_dcf = compute_min_dcf(scores, all_labels)
        else:
            eer = 0.0
            min_dcf = 0.0

        self.log(f"{self.test_name}_EER", eer, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{self.test_name}_minDCF", min_dcf, prog_bar=True, on_epoch=True, sync_dist=True)


    def process_trial_chunk(self, trial_chunk, embedding_dict):
        """
        trial_chunk: list of tuples (key1, key2, label)
        embedding_dict: 키-임베딩 딕셔너리
        
        각 트라이얼에 대해 키에 해당하는 임베딩을 가져와서,
        self.num_seg 개의 segment에 대해 cosine similarity를 계산하고
        평균을 내어 최종 점수를 반환합니다.
        """
        batch_chunk = len(trial_chunk)
        if batch_chunk == 0:
            return torch.tensor([]), []
            
        # key1, key2, label 추출
        cos_sims_1 = [embedding_dict[key1] for key1, key2, label in trial_chunk]
        cos_sims_2 = [embedding_dict[key2] for key1, key2, label in trial_chunk]
        labels = [label for key1, key2, label in trial_chunk]
        
        # (batch_chunk, num_seg, D) 형태로 스택
        buffer_seg_1 = torch.stack(cos_sims_1, dim=0).view(batch_chunk, self.num_seg, -1)
        buffer_seg_2 = torch.stack(cos_sims_2, dim=0).view(batch_chunk, self.num_seg, -1)

        # 메모리 효율적인 cosine similarity 계산
        # 각 segment 쌍에 대해 cosine similarity 계산
        scores_chunk = torch.zeros(batch_chunk, device=buffer_seg_1.device)
        
        for i in range(batch_chunk):
            # 각 트라이얼에 대해 segment 간 cosine similarity 계산
            seg1 = buffer_seg_1[i]  # (num_seg, D)
            seg2 = buffer_seg_2[i]  # (num_seg, D)
            
            # 모든 segment 쌍에 대해 cosine similarity 계산
            cos_sim_matrix = F.cosine_similarity(
                seg1.unsqueeze(1),  # (num_seg, 1, D)
                seg2.unsqueeze(0),  # (1, num_seg, D)
                dim=2
            )  # (num_seg, num_seg)
            
            # 평균 계산
            scores_chunk[i] = cos_sim_matrix.mean()
        
        return scores_chunk, labels

    def configure_optimizers(self):
        params = list(self.student_model.parameters())
        if isinstance(self.criterion_sv, torch.nn.Module):
            params += list(self.criterion_sv.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode="min", factor=self.lr_gamma, patience=self.lr_patience, verbose=True),
            'monitor': 'EER',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]