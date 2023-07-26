import sys
import argparse
import os
import numpy as np
import torch
from torch import autograd
from tqdm import tqdm
from PIL import Image
#from hilaCAM_lora.lib.CLIP.clip import *
from vision_language_models_are_bows.model_zoo import get_model
from gradcam.CLIP_explainability import interpret


#sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'clip_ancor', 'VL-CheckList'))
from vl_checklist.utils import chunks, add_caption
from vl_checklist.data_loader import DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#def perturbation_image(self, item, cam_image, cam_text, pert_steps, is_positive_pert=False):
#def perturbation_image(self, item, cam_image, pert_steps, pert_acc, is_positive_pert=False):
def perturbation_image(clip_model, img, text_inputs, cam_image, pert_steps, pert_acc, is_positive_pert=False):
    with torch.no_grad():
        if is_positive_pert:
            cam_image = cam_image * (-1)
        
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        total_num_patches = cam_image.shape[0]
        #print(f'total_num_patches: {total_num_patches}')


        ##############################################
        ##############################################
        for step_idx, step in enumerate(pert_steps):
            # find top step boxes
            num_top_patches = int((1 - step) * total_num_patches)
            #print(f'step_idx: {step_idx}, step: {step}, num_top_patches: {num_top_patches}')
            _, top_patches_indices = cam_image.topk(k=num_top_patches, dim=-1)
            #print(f'top_patches_indices: {top_patches_indices}')
            top_patches_indices = top_patches_indices.cpu().data.numpy()
            top_patches_indices.sort()

            #clip_model.visual.forward_patches - for classification
                #later for classification with fewer image tokens
            image_features = clip_model.visual.forward_patches(img,top_patches_indices)
            
            #normalize image_features and text_features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            #print(f'similarity.shape: {similarity.shape}, similarity: {similarity}')

            pos_score = similarity[0][0]
            neg_score = similarity[0][1]
            #print(f'pos_score: {pos_score}, neg_score: {neg_score}')
            if pos_score > neg_score:
                pert_acc[step_idx] += 1

        #exit(0)
        return pert_acc

    

def load_model(base_name="ViT-B/32", lora_r=-1, weight_name=""):
    clip_model, preprocess = clip.load(base_name, jit=False, lora=lora_r)
    #clip_model = clip_model.cuda()
    clip_model = clip_model.to(device=device)
    clip_model = clip_model.float()
    
    for param in clip_model.parameters():
        param.data = param.data.float()
        if param.grad:
          param.grad.data = param.grad.data.float() 

    if lora_r <= 0:
      #clip.model.convert_weights(self.clip_model)
      model.convert_weights(clip_model)

    if weight_name:
      clip_model.load_state_dict(torch.load(weight_name)['state_dict'])
        
    input_resolution = clip_model.visual.input_resolution
    context_length = clip_model.context_length
    vocab_size = clip_model.vocab_size

    print("=========")
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Image Preprocessing:", preprocess)
    print("=========")

    return clip_model, preprocess

#def main(args):
def main():
    #model_pert = ModelPert(args.COCO_path, use_lrp=True)
    #ours = GeneratorOurs(model_pert)
    #baselines = GeneratorBaselines(model_pert)
    #oursNoAggAblation = GeneratorOursAblationNoAggregation(model_pert)
    #vqa_dataset = vqa_data.VQADataset(splits="valid")
    #vqa_answers = utils.get_data(VQA_URL)
    #method_name = args.method

    #items = vqa_dataset.data
    #random.seed(1234)
    #r = list(range(len(items)))
    #random.shuffle(r)
    #pert_samples_indices = r[:args.num_samples]
    #iterator = tqdm([vqa_dataset.data[i] for i in pert_samples_indices])

    #test_type = "positive" if args.is_positive_pert else "negative"
    test_type = "positive"
    #modality = "text" if args.is_text_pert else "image"
    #batch_size = args.batch_size
    batch_size = 1
    print(f"running {test_type} pert test for image modality. batch_size: {batch_size}")

    #################################
        
    ####
    #vit_name = 'ViT-B/32'
    #self.load_model(base_name=vit_name, weight_name=checkpoint_file)
    #load with lora lib
    #lora=1
    #clip_model, preprocess = load_model(base_name=vit_name, lora_r=lora)


    CLIPWrapperModel, preprocess = get_model(model_name="NegCLIP", device=device)

    
    #checkpoint trained with get_caption_tree6(flan T5) LoRA 0.5 * DT loos  + 0.5 * contrastive loss (tree_loss_weight = 0.5)
    #filename = '/mnt5/nir/CLIP/interpret/DT_0.5_flan_t5_LoRA_CLIP_contrast_0.5_checkpoint_epoch_0012.pt.tar'

    print(f'testing NegCLIP with no extra training')

    #print(f'loading model state from filename: {filename}')
    #checkpoint = torch.load(filename) 
    #clip_model.load_state_dict(checkpoint['state_dict'])

    clip_model = CLIPWrapperModel.model
    clip_model = clip_model.to(device=device)
    clip_model = clip_model.float()

    #VL-Checklist dataloader
    #data_types
    #TYPES: ["Relation/action", "Relation/spatial", "Attribute/color", "Attribute/material", "Attribute/size", "Attribute/action", "Attribute/state"]
    #TYPES: ["Object/Location/center", "Object/Location/margin", "Object/Location/mid", "Object/Size/large", "Object/Size/medium", "Object/Size/small"]
    #TEST_DATA: ["vg"]
    data_names = ["vg"]
    #data_type = "Relation/action"
    #data_type = "Relation/spatial"
    #data_type = "Attribute/color"
    #TYPES = ["Attribute/material", "Attribute/size", "Attribute/action", "Attribute/state"]
    TYPES = ["Relation/action", "Relation/spatial", "Attribute/color", "Attribute/material", "Attribute/size", "Attribute/action", "Attribute/state"]

    positive_negative_pert_list = [True, False]
    task = 'itc'
    #positive_cam = True
    #positive_cam = False
    #relevancy_map_str = "positive" if positive_cam else "negative"
    print(f'use diff relevancy map from without normalizing first')
    for data_type in TYPES:
        for is_positive_pert in positive_negative_pert_list:
        
            print(f'starting perturbation_image test for data_type: {data_type} with is_positive_pert: {is_positive_pert}')
            print(f'cam_image = cam_image.abs() after normalization')
            d = DataLoader(data_names, data_type, task)
            for name in d.data:
                #print(f'name: {name}')

                iterator = tqdm(chunks(d.data[name], batch_size), desc="Progress", ncols=100,
                                            total=int(len(d.data[name]) / batch_size))
                #for batch in tqdm(chunks(d.data[name], self.batch_size), desc="Progress", ncols=100,
                #                              total=int(len(d.data[name]) / self.batch_size)):
                            #for batch in tqdm(chunks(d.test_data[name], self.batch_size), desc="Progress", ncols=100,
                            #                total=int(len(d.test_data[name]) / self.batch_size)):
                                

                pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
                pert_acc = [0] * len(pert_steps)
                
                for index, batch in enumerate(iterator):

                    image_paths = [z["path"] for z in batch]
                    texts_pos = [z['texts_pos'][0] for z in batch]
                    texts_neg = [z['texts_neg'][0] for z in batch]

                    #print(f'images.shape: {images.shape}, texts_pos.shape: {texts_pos.shape}, texts_neg.shape: {texts_neg.shape}')
                    #print(f'\n\nimage_paths: {image_paths}, \ntexts_pos: {texts_pos}, \ntexts_neg: {texts_neg}')

                    texts = [texts_pos[0], texts_neg[0]]

                    img = preprocess(Image.open(image_paths[0])).unsqueeze(0).to(device)
                    #img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    #texts = ["a man with eyeglasses"]
                    tokenized_text = clip.tokenize(texts).to(device)

                    R_text, R_image = interpret(model=clip_model, image=img, texts=tokenized_text, device=device)
                    #cam_image_idx = 0 if positive_cam else 1
                    positive_cam_image = R_image[0]
                    negative_cam_image = R_image[1]

                    #try diff without normaliziing first
                    #positive_cam_image = (positive_cam_image - positive_cam_image.min()) / (positive_cam_image.max() - positive_cam_image.min())
                    #negative_cam_image = (negative_cam_image - negative_cam_image.min()) / (negative_cam_image.max() - negative_cam_image.min())
                  
                    
                    cam_image = positive_cam_image - negative_cam_image
                    cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min())
                    #print(f'cam_image before abs: \n{cam_image}')
                    
                    cam_image = cam_image.abs()
                    #print(f'cam_image after abs: \n{cam_image}')
                    

                    #cam_text = (cam_text - cam_text.min()) / (cam_text.max() - cam_text.min())
                
                    #curr_pert_result = perturbation_image(item, cam_image, cam_text, args.is_positive_pert)
                    #pert_acc = perturbation_image(item, cam_image, pert_steps, pert_acc, args.is_positive_pert)
                    #pert_acc = perturbation_image(item, cam_image, pert_steps, pert_acc, True)
                    #pert_acc = perturbation_image(clip_model, img, tokenized_text, cam_image, pert_steps, pert_acc, True)   
                    #pert_acc = perturbation_image(clip_model, img, tokenized_text, cam_image, pert_steps, pert_acc, False)
                    
                    pert_acc = perturbation_image(clip_model, img, tokenized_text, cam_image, pert_steps, pert_acc, is_positive_pert)   
                    curr_pert_result = [round(res / (index+1) * 100, 2) for res in pert_acc]
                    iterator.set_description(f"Acc: {curr_pert_result}")

if __name__ == "__main__":
    #main(args)
    main()