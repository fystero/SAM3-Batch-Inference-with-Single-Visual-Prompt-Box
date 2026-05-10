import os
import sam3
from PIL import Image
import json
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import  normalize_bbox
import torch
from utils import get_files_from_folder,pr_ap_single_img,augment_one,save_results
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
from tqdm import tqdm

class SelfLearner:
    def __init__(self, input_floder, init_folder, checkpoint_path,is_reinfer=False, device=None, conf_threshold=0.6, output_folder= None, use_label_embeding=False):
        self.image_file_pathes = get_files_from_folder(input_floder,"jpg")#修改你的图片后缀名
        self.output_folder = output_folder
        self.use_label_embedings = use_label_embeding
        self.is_reinfer = is_reinfer
        self.processor = None
        self.model = None
        self.conf_threshold = conf_threshold
        self.init_model(checkpoint_path, device)
        self.DEVICE = self.processor.device
        self.hippo = {"embedings":None,"embedings_score":None}
        with torch.no_grad():
            self.dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.DEVICE)
        self.init_prompt(init_folder)
        self.results1 = {}
        self.results2 = {}
        self.curr_image_name = ''
        
    def dummy_state(self,inference_state):
        inference_state["backbone_out"].update(self.dummy_text_outputs)
        inference_state["geometric_prompt"] = self.model._get_dummy_prompt()
        return inference_state

    def init_model(self,checkpoint_path, device):
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        self.model = build_sam3_image_model(bpe_path=bpe_path,checkpoint_path=checkpoint_path,device=device)
        self.processor = Sam3Processor(self.model, confidence_threshold=self.conf_threshold,resolution=1120)

    def read_init_files(self,init_folder):
        label_pathes = get_files_from_folder(init_folder,'JSON')
        file_data = []
        # 读取 JSON 文件
        for json_file in label_pathes:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                try:
                    with open(json_file, 'r', encoding='gbk') as f:
                        data = json.load(f)
                except UnicodeDecodeError:
                    try:
                        with open(json_file, 'r', encoding='iso-8859-1') as f:
                            data = json.load(f)
                    except UnicodeDecodeError:
                        print(f"无法解码文件 {json_file}，请检查文件编码。")
                        return []
            rects = []
            labels = []
            for shape in  data['shapes']:
                if shape["shape_type"]!="rectangle":
                    continue
                rects.append(shape['points'])
                labels.append(1.0 if shape['label']=='1' else 0.0)
            file_data.append((os.path.splitext(json_file)[0]+'.jpg',rects,labels))#后缀名改好
        return file_data
    
    def update_hippo(self,new_embedings,new_scorces):
        self.hippo['embedings'] = torch.concat([self.hippo['embedings'],new_embedings[-2:]],dim=0) ####
        self.hippo['embedings_score'] = torch.concat([self.hippo['embedings_score'],new_scorces[-2:]])
        if len(self.hippo['embedings']) > 100:###
            idx = torch.randperm(100, device=self.DEVICE)[:100]###
            self.hippo['embedings'] = torch.index_select(self.hippo['embedings'], 0, idx)
            self.hippo['embedings_score'] = torch.index_select(self.hippo['embedings_score'], 0, idx)    

    def init_prompt(self,init_folder):
        file_datas = self.read_init_files(init_folder)
        embedings = []
        embedings_label = []
        for image_path, boxes,labels in file_datas:
            image = Image.open(image_path)
            width, height = image.size
            inference_state = self.processor.set_image(image,True)###是否编码位置嵌入
            self.processor.reset_all_prompts(inference_state)
            inference_state['boxes_embeds'] = None
            box_input_cxcywh = box_xyxy_to_cxcywh(torch.tensor(boxes).view(-1,4))
            norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, width, height).tolist()
            input_boxes = torch.tensor(norm_boxes_cxcywh, device=self.DEVICE, dtype=torch.float32).view(-1, 1, 4)
            if self.use_label_embedings:
                input_boxes_label = torch.tensor(labels,dtype=torch.long,device=input_boxes.device).view(-1,1)
            else:
                input_boxes_label= None
            input_feats = inference_state['backbone_out']["backbone_fpn"][-1]
            boxes_embeds = self.encode_box(input_boxes, input_feats,input_boxes_label)
            embedings_label+=labels
            embedings.append(boxes_embeds)
            torch.cuda.empty_cache()
        self.hippo["embedings"] = torch.concat(embedings, dim=0).to(self.DEVICE)
        self.hippo["embedings_score"] = torch.tensor(embedings_label, dtype=torch.float32).to(self.DEVICE)

    def encode_box(self,input_boxes, input_feats, input_boxes_label=None):
        with torch.no_grad():
            B,C,H,W=input_feats.shape
            input_feats = input_feats.flatten(2).permute(2, 0, 1)
            input_feats = self.model.geometry_encoder.img_pre_norm(input_feats).permute(1, 2, 0).view(B, C, H, W)
            return  self.model.geometry_encoder.encode_boxes(input_boxes,input_feats,input_boxes_label)
        
    def first_infer(self,image,save_path):
        boxes_embeds = self.hippo["embedings"]
        # boxes_embeds = self.denoise_pca(boxes_embeds,0.95)
        inference_state = self.processor.set_image(image,True)###是否编码位置嵌入
        self.processor.reset_all_prompts(inference_state)    
        inference_state = self.dummy_state(inference_state)
        indices = torch.randperm(len(boxes_embeds))[:30]#### 取前20个提示嵌入

        use_boxes_embeds = boxes_embeds[indices]

        inference_state['boxes_embeds'] = use_boxes_embeds

        inference_state['is_segment'] = True ###是否绘制实例分割
        inference_state = self.processor.forward_grounding(inference_state)
        # if save_path:
        #     save_results(image, save_path, inference_state, conf_thresd=0.5)
        confs =inference_state['scores'].cpu().float().numpy().reshape(-1,1)
        bboxes=inference_state['boxes'].cpu().numpy()
        infer_results = np.concatenate([bboxes,confs],axis=-1)
        self.results1[f'{self.curr_image_name}'] = infer_results.tolist()
        return inference_state, confs.reshape(-1)
    
    def set_test_inference_state(self, image, first_boxes):
        test_image,trans_boxes = augment_one(image,first_boxes)
        test_inference_state = self.processor.set_image(test_image,True)###是否编码位置嵌入
        self.processor.reset_all_prompts(test_inference_state) 
        test_inference_state = self.dummy_state(test_inference_state)
        test_inference_state['trans_boxes'] = trans_boxes
        test_inference_state['is_segment'] = False
        for i in range(2):
            del test_inference_state['backbone_out']['backbone_fpn'][0]
            del test_inference_state['backbone_out']['vision_pos_enc'][0]
        return test_inference_state  
      
    def compute_scores(self, prompt,prompt_mask, hs,hs_mask):
        if not hs_mask.any():
            return torch.tensor(0.0)
        with torch.no_grad():
            prompt_mask = prompt_mask.clone()
            hs = hs[hs_mask].unsqueeze(0).unsqueeze(0)
            out_logits = self.model.dot_prod_scoring(hs, prompt, prompt_mask)
            outputs_class = out_logits.sigmoid()
            return outputs_class
            
    def get_new_boxes_embeds(self,inference_state,confs):
        rank = confs.argsort()
        height = inference_state['original_height']       
        width = inference_state['original_width']
        inference_boxes = inference_state['boxes']
        input_feats = inference_state['backbone_out']["backbone_fpn"][-1]
        box_input_cxcywh = box_xyxy_to_cxcywh(inference_boxes.view(-1,4))
        input_boxes = normalize_bbox(box_input_cxcywh, width, height).view(-1, 1, 4)
        if self.use_label_embedings:
            input_boxes_label = torch.ones((len(input_boxes),),dtype=torch.long,device=input_boxes.device).view(-1,1)
        else:
            input_boxes_label= None
        boxes_embeds = self.encode_box(input_boxes,input_feats,input_boxes_label)
        return boxes_embeds[rank],torch.tensor(confs[rank],device=self.DEVICE)

    def reinfer(self,inference_state,test_inference_state):
        height = inference_state['original_height']       
        width = inference_state['original_width']
        inference_boxes = inference_state['boxes']
        # inference_scores = inference_state['scores'].tolist()
        prompt, prompt_mask = inference_state['prompts_embeds'],inference_state['prompts_masks']
        input_feats = inference_state['backbone_out']["backbone_fpn"][-1]

        box_input_cxcywh = box_xyxy_to_cxcywh(inference_boxes.view(-1,4))
        input_boxes = normalize_bbox(box_input_cxcywh, width, height).view(-1, 1, 4)
        if self.use_label_embedings:
            input_boxes_label = torch.ones((len(input_boxes),),dtype=torch.long,device=input_boxes.device).view(-1,1)
        else:
            input_boxes_label= None
        boxes_embeds = self.encode_box(input_boxes,input_feats,input_boxes_label)
        scores = []
        for boxe_embed in boxes_embeds:
            test_inference_state['boxes_embeds'] = boxe_embed.unsqueeze(0)
            test_inference_state['conf_threshold'] = 0.6 ###二次推理置信度阈值
            test_inference_state = self.processor.forward_grounding(test_inference_state)
            hs = test_inference_state["queries"]
            hs_mask = test_inference_state['out_probs_all']>0.35 ###得分阈值
            test_boxes = test_inference_state['boxes'].tolist()
            test_scores = test_inference_state['scores'].tolist()
            outputs_class = self.compute_scores(prompt,prompt_mask,hs,hs_mask)
            final_p, final_r, ap = pr_ap_single_img(test_boxes,test_scores,test_inference_state['trans_boxes'])
            similarity = outputs_class.mean()
            scores.append(similarity.item()*ap)
            # scores.append(similarity.item())
        scores = torch.tensor(scores,device=boxes_embeds.device)
        rank =scores.argsort()
        scores = scores[rank]
        boxes_embeds = boxes_embeds[rank]
        ind = self.selecter(scores)
        return boxes_embeds[ind], scores[ind]
    
    def selecter(self,scores):
        scores = scores.cpu().numpy()
        if len(scores) == 0:
            return []
        db = DBSCAN(eps=0.05, min_samples=3).fit(scores.reshape(-1,1))###密度聚类阈值
        cluster_id = set(db.labels_)
        max_center = 0
        max_center_id = -1
        for cid in set(cluster_id):
            if cid == -1:
                continue
            mask = db.labels_ == cid
            center = scores[mask].mean()
            if center > max_center:
                max_center = center
                max_center_id = cid
        min_mask = scores>0.15 ###最低得分阈值
        ind =np.arange(len(scores),dtype=np.int16)
        return_mask = (db.labels_ == max_center_id)&min_mask
        return_ind = ind[return_mask]
        return return_ind if len(return_ind) <= 15 else return_ind[-15:] ###最多返回15个提示嵌入

    def result_infer(self,image,inference_state,better_boxes_embeds, save_path):

        self.processor.reset_all_prompts(inference_state)

        inference_state['boxes_embeds'] = better_boxes_embeds

        inference_state['is_segment'] = True ###是否绘制实例分割
        inference_state = self.dummy_state(inference_state)
        inference_state = self.processor.forward_grounding(inference_state)
        if save_path:
            save_results(image, save_path, inference_state, conf_thresd=0.5)###是否保存结果图片，可调整结果图片的置信度阈值
        bboxes = inference_state['boxes'].cpu().numpy()
        confs = inference_state['scores'].cpu().float().numpy().reshape(-1,1)
        infer_results = np.concatenate([bboxes,confs],axis=-1)
        self.results2[f'{self.curr_image_name}'] = infer_results.tolist()

    def denoise_pca(self,X,ratio):
        X = X.reshape(-1,X.shape[-1]).float().cpu()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pca = PCA(n_components=ratio)
        X_pca = pca.fit_transform(X)
        X_final = pca.inverse_transform(X_pca)
        X_final = scaler.inverse_transform(X_final)
        return torch.tensor(X_final,device=self.DEVICE).view(-1,1,X.shape[-1])
    
    def __call__(self):
        for image_file_path in tqdm(self.image_file_pathes,desc="Processing"):
            org_image = Image.open(image_file_path)
            self.curr_image_name = os.path.splitext(os.path.basename(image_file_path))[0]
            if self.output_folder:
                first_output_path = os.path.join(self.output_folder,f'{self.curr_image_name}_0.JPG')
                second_output_path = os.path.join(self.output_folder,f'{self.curr_image_name}_1.JPG')
            else:
                first_output_path= None
                second_output_path= None
            first_inference_state, confs = self.first_infer(org_image.copy(),first_output_path)
            if self.is_reinfer:
                test_inference_state = self.set_test_inference_state(org_image.copy(), first_inference_state['boxes'].tolist())
                best_querys,query_scorces = self.reinfer(first_inference_state,test_inference_state)
                self.result_infer(org_image.copy(), first_inference_state, best_querys, second_output_path)
            else:
                best_querys,query_scorces = self.get_new_boxes_embeds(first_inference_state,confs)
                # self.result_infer(org_image.copy(), first_inference_state, best_querys, second_output_path)
            self.update_hippo(best_querys,query_scorces)
            del best_querys,query_scorces,first_inference_state
            torch.cuda.empty_cache()
        return self
    
if __name__=="__main__":

    dataset_floder = r'.\datasets\images'#所有需要推理的图片
    init_folder = r'.\datasets\init'#laleme标注好的用于提示的json文件和图片
    checkpoint_path = r".\sam3_4.pth"#权重
    device = 'cuda'#设备
    output_folder = r'.\datasets\outputs' #输出文件夹
    conf_threshold=0.6#置信度阈值
    leaner = SelfLearner(dataset_floder, init_folder, checkpoint_path,True, device ,conf_threshold, output_folder)
    leaner()
