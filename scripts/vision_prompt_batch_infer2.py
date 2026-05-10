import os
import sam3
from PIL import Image
import cv2
import json
import random
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
from utils import get_files_from_folder,pr_ap_single_img,augment_one,save_results,get_bbox_opencv
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
from tqdm import tqdm

def weighted_image_fusion(img1, img2, weight1=0.2):
    """
    将两张图片按照指定权重相加融合
    
    参数:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径
        weight1: 第一张图片的权重（0-1之间）
        output_path: 输出图片保存路径
    
    返回:
        fused_image: 融合后的图像
    """

    
    if img1 is None or img2 is None:
        raise ValueError("无法读取图片，请检查文件路径")
    
    # 确保两张图片尺寸相同
    if img1.shape != img2.shape:
        # 将第二张图片调整为第一张图片的尺寸
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 计算第二张图片的权重
    weight2 = 1.0 - weight1
    
    # 执行加权融合
    fused_image = cv2.addWeighted(img1, weight1, img2, weight2, 0)
    
    
    return fused_image

def segment_image_by_polygons(image_path, polygons_list):
    """
    根据多边形坐标列表分割图像
    
    参数:
        image_path: 输入图片路径
        polygons_list: 多边形坐标列表，每个元素为N×2的数组，表示一个多边形的顶点坐标
    
    返回:
        segmented_image: 分割后的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像，请检查文件路径")
    
    # 创建与原图相同大小的掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 为每个多边形创建掩码
    for polygon in polygons_list:
        # 确保多边形坐标格式正确
        polygon = np.array(polygon, dtype=np.int32)
        if polygon.shape[0] < 3:
            continue  # 至少需要3个点才能构成多边形
        
        # 在掩码上绘制填充的多边形
        cv2.fillPoly(mask, [polygon], 255)
    
    # 应用掩码：保留多边形内部区域
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 将多边形外部区域填充为黑色
    # 创建黑色背景
    black_background = np.zeros_like(image)
    # 将掩码区域从原图复制到黑色背景
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    segmented_image = cv2.add(black_background, segmented_image)
    
    return segmented_image

class SelfLearner:
    def __init__(self, input_floder, init_folder, checkpoint_path,is_reinfer=False, device=None, conf_threshold=0.5, output_folder= None,use_label_embeding=False):
        self.image_file_pathes = get_files_from_folder(input_floder,"JPG")
        self.output_folder = output_folder
        self.use_label_embedings = use_label_embeding
        self.is_reinfer = is_reinfer
        self.processor = None
        self.model = None
        self.conf_threshold = conf_threshold
        self.init_model(checkpoint_path, device)
        self.DEVICE = self.processor.device
        self.hippo = {"prompt_images":[],"prompt_boxes":[],'prompt_masks':[],'prompt_scores':[]}
        with torch.no_grad():
            self.dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.DEVICE)
        self.init_prompt(init_folder)
        self.results = {}

        self.curr_image_name = ''
        
    def dummy_state(self,inference_state):
        inference_state["backbone_out"].update(self.dummy_text_outputs)
        inference_state["geometric_prompt"] = self.model._get_dummy_prompt()
        return inference_state

    def init_model(self,checkpoint_path, device):
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        self.model = build_sam3_image_model(bpe_path=bpe_path,checkpoint_path=checkpoint_path,device=device)
        self.processor = Sam3Processor(self.model, confidence_threshold=self.conf_threshold)

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
            polygons = []
            labels = []
            for shape in  data['shapes']:
                if shape["shape_type"]!="polygon":
                    continue
                polygons.append(shape['points'])
                rects.append(get_bbox_opencv(shape['points']))
                labels.append(1.0 if shape['label']=='1' else 0.0)
            file_data.append((os.path.splitext(json_file)[0]+'.jpg',rects,polygons,labels))
        return file_data
    

    def init_prompt(self,init_folder):
        file_datas = self.read_init_files(init_folder)

        for image_path, boxes,polygons,labels in file_datas:
            
            segmented_image = segment_image_by_polygons(image_path,polygons)
            self.hippo["prompt_images"].append(segmented_image)
            self.hippo["prompt_boxes"].append(boxes)
            self.hippo["prompt_scores"].append(labels)

    def encode_box(self,input_boxes, input_feats, input_boxes_label=None):
        with torch.no_grad():
            B,C,H,W=input_feats.shape
            input_feats = input_feats.flatten(2).permute(2, 0, 1)
            input_feats = self.model.geometry_encoder.img_pre_norm(input_feats).permute(1, 2, 0).view(B, C, H, W)
            return  self.model.geometry_encoder.encode_boxes(input_boxes,input_feats,input_boxes_label)
        
    def first_infer(self,image):
        prompt_images = self.hippo["prompt_images"]
        height,width = image.shape[0],image.shape[1]
        indices = torch.randperm(len(prompt_images))[0].item()
        use_prompt_image = prompt_images[indices]
        fused_image = weighted_image_fusion(image,use_prompt_image,0.5)
        fused_image = cv2.cvtColor(fused_image,cv2.COLOR_BGR2RGB)
        inference_state = self.processor.set_image(Image.fromarray(fused_image),False)
        self.processor.reset_all_prompts(inference_state)    
        inference_state = self.dummy_state(inference_state)
        use_prompt_image_box = self.hippo["prompt_boxes"][indices]
        use_prompt_image_label = self.hippo["prompt_scores"][indices]
        box_input_cxcywh = box_xyxy_to_cxcywh(torch.tensor(use_prompt_image_box, device=self.DEVICE, dtype=torch.float32).view(-1,4))
        norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, width, height).view(-1, 1, 4)
        if self.use_label_embedings:
            input_boxes_label = torch.tensor(use_prompt_image_label,dtype=torch.long,device=self.DEVICE).view(-1,1)
        else:
            input_boxes_label= None
        input_feats = inference_state['backbone_out']["backbone_fpn"][-1]
        boxes_embeds = self.encode_box(norm_boxes_cxcywh, input_feats,input_boxes_label)
        return boxes_embeds
    
    def set_test_inference_state(self, image, first_boxes):
        test_image,trans_boxes = augment_one(image,first_boxes)
        test_inference_state = self.processor.set_image(test_image)
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

    def reinfer(self,image,boxes_embeds,save_path):

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        inference_state = self.processor.set_image(Image.fromarray(image),True)
        self.processor.reset_all_prompts(inference_state)    
        inference_state = self.dummy_state(inference_state)
        inference_state['boxes_embeds'] = boxes_embeds
        inference_state['is_segment'] = True
        inference_state = self.processor.forward_grounding(inference_state)
        if save_path:
            save_results(Image.fromarray(image), save_path, inference_state, conf_thresd=0.5)
        bboxes = inference_state['boxes'].cpu().numpy()
        confs = inference_state['scores'].cpu().float().numpy().reshape(-1,1)
        infer_results = np.concatenate([bboxes,confs],axis=-1)
        self.results[f'{self.curr_image_name}'] = infer_results.tolist()
        return inference_state
    
    def __call__(self):
        for image_file_path in tqdm(self.image_file_pathes,desc="Processing"):
            org_image = cv2.imread(image_file_path)
            self.curr_image_name = os.path.splitext(os.path.basename(image_file_path))[0]
            if self.output_folder:
                first_output_path = os.path.join(self.output_folder,f'{self.curr_image_name}_0.JPG')
                second_output_path = os.path.join(self.output_folder,f'{self.curr_image_name}_1.JPG')
            else:
                first_output_path= None
                second_output_path= None
            boxes_embeds = self.first_infer(org_image.copy())
            inference_state = self.reinfer(org_image.copy(),boxes_embeds,second_output_path)
            torch.cuda.empty_cache()
        return self
    
if __name__=="__main__":

    dataset_floder = r'C:\Users\25000\Desktop\sam3\dataset\images'
    init_folder = r'C:\Users\25000\Desktop\sam3\dataset\init'
    checkpoint_path = r"E:\Code\SAM3-Adapter-Pytorch\sam3_4.pth"#"E:\Code\SAM3-Adapter-Pytorch\sam3_4.pth"
    device = 'cuda'
    output_folder = r'C:\Users\25000\Desktop\sam3\dataset\outputs' 
    leaner = SelfLearner(dataset_floder, init_folder, checkpoint_path,True, device,0.5, output_folder)
    leaner()
