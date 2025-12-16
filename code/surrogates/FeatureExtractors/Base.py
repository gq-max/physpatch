import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Callable, Dict
import torch.nn.functional as F
import numpy as np

class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

class EnsembleFeatureExtractor_ours(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor], k=8):
        super(EnsembleFeatureExtractor_ours, self).__init__()
        self.extractors = nn.ModuleList(extractors) 
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        features = {}  
        features_local = {}
        for i, model in enumerate(self.extractors):
            # features[i] = model(x).squeeze()
            x_global, x_local = model.global_local_features(x.to(x.device))
            features[i] = x_global.squeeze()
            svd_feature = self.get_svd_feature(x_local[0], x.device)
            features_local[i] = svd_feature
        return features, features_local
    
    def encode_img(self, x: Tensor) -> Tensor:
        features = {}  
        for i, model in enumerate(self.extractors):
            features[i] = model(x).squeeze()
        return features
    
    
    def get_svd_feature(self, embedding_image, device):
        # emdedding_image:[N, D]
        U_o, S_o, _ = torch.linalg.svd(embedding_image, full_matrices=False) # [N, D], [D], [D, D]

        U_o_k = U_o[:, :self.k] # [N, k]
        S_o_k = S_o[:self.k] # [k]
        feat_o_reduced = U_o_k @ torch.diag(S_o_k) # [N, k] @ [k, k] → [N, k]
        return feat_o_reduced.to(device)
    
class EnsembleFeatureLoss_ours(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], k=8):
        super(EnsembleFeatureLoss_ours, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.image_full_feature = []
        self.k = k
    
    @torch.no_grad()
    def set_full_feature(self, x: Tensor):
        self.image_full_feature.clear()
        for model in self.extractors:
            self.image_full_feature.append(model(x).to(x.device))
    
    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            # self.ground_truth.append(model(x).to(x.device))
            x_global, x_local = model.global_local_features(x.to(x.device))
            svd_feature = self.get_svd_feature(x_local[0], x.device)
            self.ground_truth.append(x_global)
            self.ground_truth_local.append(svd_feature)
            

    def __call__(self, feature_dict: Dict[int, Tensor], feature_local_dict: Dict[int, Tensor], feature_full_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        loss_local = 0
        loss_full = 0
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index]
            gt = self.ground_truth[index]
            gt_full = self.image_full_feature[index]
            feature_local = feature_local_dict[index]
            feature = feature_dict[index]
            feature_full = feature_full_dict[index]
            # feature_local = feature_local / feature_local.norm(dim=1, keepdim=True)
            # gt_local = gt_local / gt_local.norm(dim=1, keepdim=True)
            loss_local += F.mse_loss(feature_local, gt_local)
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            loss_full += torch.mean(torch.sum(feature_full * gt_full, dim=1))
        
        loss = loss / len(self.extractors)
        loss_local=loss_local / len(self.extractors)
        loss_full = loss_full / len(self.extractors)
        print("loss:", loss, "loss_local:", loss_local, "loss_full:", loss_full)
        loss = loss + loss_local + loss_full*0.1
        return loss
    
    def tgt_ori_loss(self, feature_dict: Dict[int, Tensor], feature_local_dict: Dict[int, Tensor], feature_dict_ori: Dict[int, Tensor], feature_local_dict_ori: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        loss_local = 0
        loss_ori = 0
        loss_local_ori = 0
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index]
            gt = self.ground_truth[index]
            feature_local = feature_local_dict[index]
            feature = feature_dict[index]
            feature_local_ori = feature_local_dict_ori[index]
            feature_ori = feature_dict_ori[index]
            feature_local = feature_local / feature_local.norm(dim=1, keepdim=True)
            gt_local = gt_local / gt_local.norm(dim=1, keepdim=True)
            loss_local += F.mse_loss(feature_local, gt_local)
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            loss_local_ori += F.mse_loss(feature_local_ori, gt_local)
            loss_ori += torch.mean(torch.sum(feature_ori * gt, dim=1))
            
            
        loss = loss / len(self.extractors)
        loss_local=loss_local / len(self.extractors)
        loss_ori = loss_ori / len(self.extractors)
        loss_local_ori = loss_local_ori / len(self.extractors)
        loss = loss + loss_local*0.1 - loss - 0.1 * loss_local_ori
        return loss
    
    def get_svd_feature(self, embedding_image, device):
        # emdedding_image:[N, D]
        U_o, S_o, _ = torch.linalg.svd(embedding_image, full_matrices=False) # [N, D], [D], [D, D]
        U_o_k = U_o[:, :self.k] # [N, k]
        S_o_k = S_o[:self.k] # [k]
        feat_o_reduced = U_o_k @ torch.diag(S_o_k) # [N, k] @ [k, k] → [N, k]
        return feat_o_reduced.to(device)
    
class EnsembleFeatureLoss_ours_auto(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], k=8):
        super(EnsembleFeatureLoss_ours_auto, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.image_full_feature = []
        self.k = k
        self.previous_loss_list=[]
    
    @torch.no_grad()
    def set_full_feature(self, x: Tensor):
        self.image_full_feature.clear()
        for model in self.extractors:
            self.image_full_feature.append(model(x).to(x.device))
    
    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            # self.ground_truth.append(model(x).to(x.device))
            x_global, x_local = model.global_local_features(x.to(x.device))
            svd_feature = self.get_svd_feature(x_local[0], x.device)
            self.ground_truth.append(x_global)
            self.ground_truth_local.append(svd_feature)
            

    def __call__(self, feature_dict: Dict[int, Tensor], feature_local_dict: Dict[int, Tensor], feature_full_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        loss_full_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index]
            gt = self.ground_truth[index]
            gt_full = self.image_full_feature[index]
            feature_local = feature_local_dict[index]
            feature = feature_dict[index]
            feature_full = feature_full_dict[index]
            # feature_local = feature_local / feature_local.norm(dim=1, keepdim=True)
            # gt_local = gt_local / gt_local.norm(dim=1, keepdim=True)
            loss_local_list.append(F.mse_loss(feature_local, gt_local))
            loss_list.append(torch.mean(torch.sum(feature * gt, dim=1)))
            loss_full_list.append(torch.mean(torch.sum(feature_full * gt_full, dim=1)))
        total_losses = [
            loss_list[i] + 1 * loss_local_list[i]  + 0.1 * loss_full_list[i]
            for i in range(len(self.extractors))
        ]
        if len(self.previous_loss_list) == 0: 
            self.previous_loss_list = [l.detach() for l in loss_full_list]
        weights = []
        for i in range(len(self.extractors)):
            ratio = loss_full_list[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  
       
        for i in range(len(self.extractors)): 
            self.previous_loss_list[i] = loss_full_list[i].detach()
        

        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        print("loss_full:", sum(loss_full_list) / len(loss_full_list))
       
        return total_loss
    
    def get_svd_feature(self, embedding_image, device):
        # emdedding_image:[N, D]
        U_o, S_o, _ = torch.linalg.svd(embedding_image, full_matrices=False) # [N, D], [D], [D, D]

        U_o_k = U_o[:, :self.k] # [N, k]
        S_o_k = S_o[:self.k] # [k]

        feat_o_reduced = U_o_k @ torch.diag(S_o_k) # [N, k] @ [k, k] → [N, k]
        return feat_o_reduced.to(device)


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors) 

    def forward(self, x: Tensor) -> Tensor:
        features = {}  
        for i, model in enumerate(self.extractors):
            features[i] = model(x).squeeze() 
        return features
    
    def encode_text(self, texts: List[str]) -> Dict[int, Tensor]:
        text_features = {}
        for i, model in enumerate(self.extractors):
            if hasattr(model, "encode_text"):
                text_features[i] = model.encode_text(texts).squeeze()  # shape: [N, D]
        return text_features


class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.all_extractors = nn.ModuleList(extractors)
        self.extractors = nn.ModuleList(extractors)
        self.last_loss = None
        self.image_ground_truth = []
        self.image_full_target = []
        self.text_ground_truth = []
        self.text_ground_truth_neg = []
        self.image_full_feature = []
    
    @torch.no_grad()
    def set_full_feature(self, x: Tensor):
        self.image_full_feature.clear()
        for model in self.extractors:
            self.image_full_feature.append(model(x).to(x.device))

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.image_ground_truth.clear()
        for model in self.extractors:
            self.image_ground_truth.append(model(x).to(x.device))
    
    @torch.no_grad()
    def set_full_target(self, x: Tensor):
        self.image_full_target.clear()
        for model in self.extractors:
            self.image_full_target.append(model(x).to(x.device))
    def global_loss(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.image_full_feature[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            
        loss = loss / len(self.extractors)

        return loss


    def __call__(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.image_ground_truth[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            
        loss = loss / len(self.extractors)

        return loss
