# Imports
#pip install transformers
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git

import argparse
import os
import numpy as np
import torch
from torch import autograd
import torchvision 
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader, Dataset  
from tqdm import tqdm
#import clip  
from torchvision.datasets import CocoCaptions
from lora.lib.CLIP.clip import *
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
from gradcam.CLIP_explainability import get_image_text_relevance
#import datasets

learning_rate = 0.001
batch_size = 64
num_epochs = 3
device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='CLIP COCO LoRA finetune')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--checkpoint_file', default='', type=str,
                    help='filename to save checkpoint to')


parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:2375', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for ei ther single node or '
                         'multi node data parallel training')
                                     
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--num-cycles', default=10, type=int)




#imagenet_train_dataset = ImageNet(root="/mnt5/nir/dataset/ImageNet", split='train', transform=preprocess)
    
#imagenet_test_dataset = ImageNet(root="/mnt5/nir/dataset/ImageNet", split='val', transform=preprocess)                     

class CocoCaptionsDataset(CocoCaptions):
  
  def __init__(self, root, annFile, transform):
    super().__init__(root, annFile, transform)

  def _load_image(self, id: int) -> Image.Image:
    path = self.coco.loadImgs(id)[0]["file_name"]
    full_path = os.path.join(self.root, path)
    return Image.open(full_path).convert("RGB") , full_path

  def __getitem__(self, index):

    id = self.ids[index]
    image , full_path = self._load_image(id)
    #target contains 5 captions per image - use only the first caption
    target = self._load_target(id)[0]

    if self.transforms is not None:
      image, target = self.transforms(image, target)

    return full_path, image, target
    
  



class DecisionTreeClipModel(nn.Module):
  def __init__(self, lora=-1, checkpoint_file=""):
    super(DecisionTreeClipModel, self).__init__()
    
    vit_name = 'ViT-B/32'
    #self.load_model(base_name=vit_name, weight_name=checkpoint_file)
    self.clip_model, self.preprocess = self.load_model(base_name=vit_name, lora_r=lora)
    #self.freeze_visual_head()
    self.clip_model = self.clip_model.float()


 

  def load_model(self, base_name="ViT-B/32", lora_r=-1, weight_name=""):
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

  def calc_label_path_dict(self):
    
    num_imagenet_classes = len(self.hier.LEAF_IDS)

    for class_num in range(num_imagenet_classes):
      path, class_num = self.GetPathToLeaf(class_num)
      if path:
        self.good_labels.append(class_num)
        #to get label path from node id path
        #label_path = [hier.HIER_NODE_ID_TO_NUM[id] for id in path]
        self.label_path_dict[class_num] = path
  

  def get_encoded_prompts(self, node_names):
    #make sure prompts are in the same order of HIER_NODE_NUM(which is the node label)
    #num_nodes = len(all_nodes)
    #node_names = [node.prompt for node in all_nodes ]

    prompts = [f'a photo of {name}' for name in node_names]

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)
    #tokenized_prompts.double()
    #print(f'tokenized_prompts.type(): {tokenized_prompts.type()}')
    
    with torch.no_grad():
        encoded_prompts = self.clip_model.encode_text(tokenized_prompts).to(device=device)
        encoded_prompts /= encoded_prompts.norm(dim=-1, keepdim=True)   

    #encoded_prompts = encoded_prompts.double()
    #move to gpu only in forward
    #encoded_prompts = encoded_prompts.to('cpu')
    return encoded_prompts


  #def interpret_prediction(self, x, caption):
  #  self.classify_example(x, caption, print_path=True, break_on_incorrect=True)


  def get_caption_tree_figure(self, caption, file_name="tree_fig.png"):
    root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree(caption)
    return self.get_caption_tree_figure(all_tree_nodes, edges, caption, node_texts, file_name)

  def get_caption_tree_figure(self, all_tree_nodes, edges, caption, node_texts, file_name="tree_fig.png"):
    
    file_path = plotly_plot_tree(all_tree_nodes, edges, caption, node_texts, file_name)
    tree_fig = Image.open(file_path)
    return tree_fig

  def get_image_figure(self, image, permute=True):
    if permute:
      image = np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0))
    
    image *= (1.0/image.max())
    image=(image+1)/2
    
    pil_image=Image.fromarray((image * 255).astype(np.uint8))
    return pil_image

  def get_caption_tree_and_image_figure(self, image, all_tree_nodes, edges, caption, node_texts, image_dir = "classification_samples", tree_file_name="tree_fig.png"):
    img_fig = self.get_image_figure(image)
    tree_fig = self.get_caption_tree_figure(all_tree_nodes, edges, caption, node_texts, tree_file_name)

    if not os.path.exists(image_dir):
      os.mkdir(image_dir)

    merged_img = get_concat_h_blank(img_fig, tree_fig)

    merged_file = f"sample_{self.sample_num}.png"
    merged_path = image_dir + "/" + merged_file

    merged_img.save(merged_path)
    

  def get_grad_cam(self, x, img_path):
    input_grad = x.grad.data.cpu()
    img_arr = x.detach().clone().cpu()
    #x.shape: torch.Size([1, 3, 224, 224]), input_grad.shape: torch.Size([1, 3, 224, 224])
    #print(f'\nget_grad_cam x.shape: {x.shape}, input_grad.shape: {input_grad.shape}\n')

    ######
    pil_img = Image.open(img_path)

    transform = Compose([
                         #Resize(224),
                         #CenterCrop(224),
                         ToTensor(),
                         #Normalize(mean=[0.485, 0.456, 0.406],
                         #          std=[0.229, 0.224, 0.225])
                         ])

    
    img_tensor_ = transform(pil_img)
    #img_tensor_ = ToTensor(pil_img)
    img_tensor = img_tensor_.unsqueeze(0)
    #img_arr = img_tensor.clone()
    print(f'\nimg_tensor.shape: {img_tensor.shape}')
    ######

    # pool the gradients across the channels
    pooled_gradients = torch.mean(input_grad, dim=[0, 2, 3])
    #pooled_gradients.shape: torch.Size([3])
    #print(f'pooled_gradients.shape: {pooled_gradients.shape}')

    # weight the channels by corresponding gradients
    for i in range(3):
      #img_arr[:, :, i] *= pooled_gradients[i]
      img_arr[:, i, :, :] *= pooled_gradients[i]
      img_tensor[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the image
    heatmap = torch.mean(img_arr, dim=1).squeeze()
    #heatmap = torch.mean(img_tensor, dim=1).squeeze()
    #heatmap.shape: torch.Size([224, 224])
    #print(f'heatmap.shape: {heatmap.shape}')

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    dpi = 400
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.matshow(heatmap.squeeze())

    #TODO: move to a function
    dir_path = "heatmap"

    if not os.path.exists(dir_path):
      os.mkdir(dir_path)

    #sample_num = 1
    file_name = f"heatmap_{self.sample_num}.png"
    save_path = dir_path + "/" + file_name
    ###

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    print(f'img.shape: \n{img.shape}')
    #heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    file_name = f"resized_heatmap_{self.sample_num}.png"
    save_path = dir_path + "/" + file_name
    cv2.imwrite(save_path, heatmap)

    superimposed_img = heatmap * 0.4 + img
    #superimposed_img = heatmap * 0.4 + img_arr.squeeze().permute(1,2,0).numpy()

    #TODO: move to a function
    dir_path = "gradcam"

    if not os.path.exists(dir_path):
      os.mkdir(dir_path)

    #sample_num = 1
    file_name = f"gradcam_{self.sample_num}.png"
    gradcam_save_path = dir_path + "/" + file_name

    file_name = f"img_{self.sample_num}.png"
    img_save_path = dir_path + "/" + file_name

    ###

    cv2.imwrite(gradcam_save_path, superimposed_img)
    pil_img.save(img_save_path)
    #cv2.imwrite(img_save_path, pil_img)


  def classify_example(self, img_path, x, caption, print_all=False, print_incorrect=True):
    
    root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree(caption)

    probs_matrix, _ = self.get_probs_matrix(x, true_label_path, all_tree_nodes)
    prob_values, predictions = probs_matrix.topk(1)
    #print(f'\nprob_values: {prob_values}\npredictions: {predictions}\n')
    #print(f'\ntrue_label_path: {true_label_path}\n')
    
    if print_all or print_incorrect:
      #print('\nprediction path:')
      #print('predicted caption (probability) - correct/incorrect\n')
      str_to_print = '\nprediction path:\n'
      str_to_print += 'predicted caption (probability) - correct/incorrect\n\n'
      for i, predicted_node_num in enumerate(predictions):
        is_correct = (predicted_node_num == true_label_path[i])
        correct_str = "correct" if is_correct else "incorrect"
        predicted_caption = all_tree_nodes[predicted_node_num].prompt
        
        #print(f'{predicted_caption} ({100 * prob_values[i].item():.2f}%) - {correct_str}')
        str_to_print += f'{predicted_caption} ({100 * prob_values[i].item():.2f}%) - {correct_str}\n'

        if print_all:
           print(str_to_print)

        if not is_correct:
          if print_incorrect and (not print_all):
            print(str_to_print)
            
          correct_node_num = true_label_path[i]
          correct_caption = all_tree_nodes[correct_node_num].prompt
          correct_prob = probs_matrix[i][correct_node_num].item()

          #print(f'path_index: {i}, all_tree_nodes[correct_node_num].node_num: {all_tree_nodes[correct_node_num].node_num}, correct_node_num: {correct_node_num}')
          #print(f'all_tree_nodes[correct_node_num].prompt is: {all_tree_nodes[correct_node_num].prompt}')

          print(f'correct caption is: {correct_caption} - with probability: ({100 * correct_prob:.2f}%)')
          expanded_captions = expand_caption(predicted_caption, correct_caption)
          encoded_prompts = self.get_encoded_prompts(expanded_captions)
          similarity = self.get_cosine_similarity(x,encoded_prompts)
          probs = similarity.softmax(dim=-1)
          #print(f'\n\nexpanded_captions: \n{expanded_captions}')
          #print(f'\n\nencoded_prompts: {encoded_prompts}')
          #print(f'\n\nsimilarity: {similarity}')
          #print(f'\n\nprobs: {probs}')
          #print(f'\n\nencoded_prompts.shape: {encoded_prompts.shape}')
          #print(f'\n\nsimilarity.shape: {similarity.shape}')
          #print(f'\n\nprobs.shape: {probs.shape}')

          probs = probs.flatten()
          #print(f'\n\nprobs.shape: {probs.shape}, probs: {probs}')
          #print(f'\n\nlen(expanded_captions): {len(expanded_captions)}, \nexpanded_captions: {expanded_captions}')
          #print(f'\n\probs.flatten().shape: {probs.flatten().shape}')
  
          print(f'\nexpand node prediction: ')
          for idx, capt in enumerate(expanded_captions):
            print(f'caption: \'{capt}\', prob: ({100 * probs[idx].item():.2f}%)')

          #self.train() 
          with torch.set_grad_enabled(True):
            #HilaCAM
            #texts = [predicted_caption, correct_caption]
            #get_image_text_relevance(img_path, texts, self.clip_model, self.preprocess)
            x = x.to(device=device)
            x.requires_grad = True
            loss_fn = nn.NLLLoss(reduction='none')
            ###################
          
            log_probabilities = self.forward(x,true_label_path, all_tree_nodes)
            node_num_labels = torch.tensor(true_label_path,device=device)

            loss = loss_fn(log_probabilities, node_num_labels)
            #print(f'i: {i}, log_probabilities.shape: {log_probabilities.shape}, node_num_labels.shape: {node_num_labels.shape}, loss.shape: {loss.shape}')
            #incorrect_loss = loss[i, correct_node_num]
            incorrect_loss = loss[i]
            incorrect_loss.backward()
            #input_grad = x.grad.data.cpu().numpy()
            input_grad = x.grad.data
            #input_grad.shape: torch.Size([1, 3, 224, 224])
            #print(f'input_grad.shape: {input_grad.shape}')
            grad_arr = torch.abs(input_grad).mean(dim=1).permute(1, 2, 0)
            grad_arr /= grad_arr.quantile(0.98)
            grad_arr = torch.clamp(grad_arr, 0, 1)
            grad_arr = grad_arr.cpu().numpy()
            #input_grad_scaled = 
            x.requires_grad = False
            x_arr = to_array(x)
            #x_arr = x.squeeze()
            #x_arr = x_arr.cpu().permute(1, 2, 0).detach().numpy()
            grad_img_arr = x_arr * grad_arr
            #grad_img_arr = x * grad_arr
            #grad_img_arr = x * input_grad
            grad_pil_img = self.get_image_figure(grad_img_arr, permute=False)
            #grad_pil_img = self.get_image_figure(grad_img_arr)
          

            dir_path = "gradients_samples"

            if not os.path.exists(dir_path):
              os.mkdir(dir_path)

            #sample_num = 1
            file_name = f"gradient_{self.sample_num}.png"
            save_path = dir_path + "/" + file_name

            grad_pil_img.save(save_path)
            #self.eval()
            ###################
            self.get_grad_cam(x, img_path)
            self.get_caption_tree_and_image_figure(x, all_tree_nodes, edges, caption, node_texts)
            
            self.sample_num += 1
            exit(0)
            break

    return predictions, torch.tensor(true_label_path, device=device)

  """
  def classify_example(self, x, print_path=False):
    curr_node = self.root_id
    #node_path_list = []

    decendants = self.descendants_dict[curr_node]
    if print_path:
      print('prediction path:')
      print('prediction (probability)\n')

    while (decendants):
    
      encoded_prompts = self.decendants_prompts_dict[curr_node]
      similarity = self.get_cosine_similarity(x,encoded_prompts)
      probs = similarity.softmax(dim=-1)
      #print(f'similarity: {similarity}, probs: {probs}')
      value, index = probs[0].topk(1)
      curr_node = decendants[index]
      if print_path:
        print(f'{self.hier.HIER_NODE_NAME[curr_node].split(", ")[0]} ({100 * value.item():.2f}%)')
        #node_path_list.append((curr_node,value))
      decendants = self.descendants_dict[curr_node]
    
    return self.hier.LEAF_ID_TO_NUM[curr_node]
    """

  def get_cosine_similarity(self, x, encoded_prompts):
    images = x.to(device=device)
    image_features = self.clip_model.encode_image(images)
    
    #print(f'get_cosine_similarity image_features.type(): {image_features.type()}, image_features: \n\n{image_features}')
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    encoded_prompts = encoded_prompts.to(device=device)
    similarity = 100.0 * image_features @ encoded_prompts.t() #batch_size X num nodes
    #print(f'get_cosine_similarity similarity: \n\n{similarity}')

    #encoded_prompts = encoded_prompts.to(device='cpu')
    return similarity


  def get_probs_matrix(self, images, true_label_path, all_tree_nodes):
    
    node_names = [node.prompt for node in all_tree_nodes ]
    self.encoded_prompts = self.get_encoded_prompts(node_names)
    #print(f'\nencoded_prompts.grad_fn: {self.encoded_prompts.grad_fn}\n')

    similarity = self.get_cosine_similarity(images,self.encoded_prompts)
    #print(f'\nsimilarity.grad_fn: {similarity.grad_fn}\n')
    exponent_similarity = similarity.exp()
    #print(f'\nexponent_similarity.grad_fn: {exponent_similarity.grad_fn}\n')

    #duplicate each row of exponent_similarity by len of its coresponding path
    duplicated_exponent_similarity = self.duplicate_matrix_rows(exponent_similarity,true_label_path)
    #print(f'\nduplicated_exponent_similarity.grad_fn: {duplicated_exponent_similarity.grad_fn}\n')

    #print(f'\n\n\nlen(unrolled_node_ids): {len(unrolled_node_ids)}')
    #print(f'exponent_similarity.shape: {exponent_similarity.shape}')
    #print(f'duplicated_exponent_similarity.shape: {duplicated_exponent_similarity.shape}')

    #exit(0)

    #DT_CLIP_model(images,true_label_path, all_tree_nodes)

    path_mask_matrix = self.get_path_mask_matrix(duplicated_exponent_similarity, true_label_path, all_tree_nodes)
    #print(f'\npath_mask_matrix.grad_fn: {path_mask_matrix.grad_fn}\n')

    softmax_numerator = duplicated_exponent_similarity*path_mask_matrix
    #print(f'\nsoftmax_numerator.grad_fn: {softmax_numerator.grad_fn}\n')
    
    softmax_denominator = softmax_numerator.sum(dim=1,keepdim=True)
    #print(f'\nsoftmax_denominator.grad_fn: {softmax_denominator.grad_fn}\n')
    probs = softmax_numerator / softmax_denominator

    return probs, path_mask_matrix

  def forward(self, images, true_label_path, all_tree_nodes):

    probs_matrix, path_mask_matrix = self.get_probs_matrix(images, true_label_path, all_tree_nodes)
    #print(f'\nprobs_matrix.grad_fn: {probs_matrix.grad_fn}\n')
    #print(f'\npath_mask_matrix.grad_fn: {path_mask_matrix.grad_fn}\n')

    #log of zero probabilities will return -inf but will not be used 
    #as only the log probability of the ground truth is used in NLL Loss
    #RuntimeError: Function 'LogBackward0' returned nan values in its 0th output.
    #log_probs = probs.log()
    probs_matrix[path_mask_matrix.bool()] = probs_matrix[path_mask_matrix.bool()].log()
    #print(f'\nprobs_matrix.grad_fn: {probs_matrix.grad_fn}\n')

    #return log_probs
    return probs_matrix


  def duplicate_matrix_rows(self, x, true_label_path):
    
    #res = torch.cat([x[idx].repeat(len(self.label_path_dict[l.item()]),1) for idx, l in enumerate(labels)] )
    path_len = len(true_label_path)
    res = x.repeat(path_len,1)
    return res
    """
    >>> t = torch.tensor([[1, 2, 3], [4, 4, 4]])                                                        
    >>> t
    tensor([[1, 2, 3],
        [4, 4, 4]])
    >>> s = torch.cat([t[i].repeat(5,1) for i in range(len(t))] )
    >>> s
    tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4]])
    """

  def exponent_matrix(self, x):
    res = torch.exp(x)
    return res 

    """
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> ex = torch.exp(t)
    >>> ex
    tensor([[2.7183e+00, 7.3891e+00, 2.0086e+01],
            [5.4598e+01, 1.4841e+02, 4.0343e+02],
            [1.0966e+03, 2.9810e+03, 8.1031e+03]])  
    """

  def get_path_mask_matrix(self, similarity_matrix, true_label_path, all_tree_nodes):
    
    mask = torch.zeros_like(similarity_matrix)
    
    for idx, node_num in enumerate(true_label_path):
      true_label_node = all_tree_nodes[node_num]
      parent_node = true_label_node.parent
      siblings = parent_node.children
      row_mask = list(siblings.keys())
      mask[idx][row_mask] = 1

    #for idx, node_id in enumerate(unrolled_node_ids):
    #  siblings = self.get_siblings_node_ids(node_id)
    #  row_mask = [self.hier.HIER_NODE_ID_TO_NUM[s] for s in siblings]
    #  mask[idx][row_mask] = 1
    
    #TODO: just mask similarity_matrix with similarity_matrix[idx][~row_mask] = 0 ??

    return mask

    """
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> p = torch.zeros_like(t)
    >>> p
    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]) 
    >>> row_mask = torch.tensor([0, 2])
    >>> row_mask
    tensor([0, 2])
    >>> p[0][row_mask] = 1
    >>> p
    tensor([[1, 0, 1],
            [0, 0, 0],
            [0, 0, 0]])
    """

  def get_probabilities_matrix(self, masked_exponents_matrix):
    
    denominator = masked_exponents_matrix.sum(dim=1,keepdim=True)
    res = masked_exponents_matrix / denominator
    
    return res    

    """
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> su = t.sum(dim=1,keepdim=True)
    >>> su
    tensor([[ 6],
            [15],
            [24]])
    >>> su.shape
    torch.Size([3, 1])
    >>> res = t / su
    >>> res
    tensor([[0.1667, 0.3333, 0.5000],
            [0.2667, 0.3333, 0.4000],
            [0.2917, 0.3333, 0.3750]])

    """

  def get_log_probabilities_matrix(self, probabilities_matrix):
    
    #zero probabilities will return -inf but will not be used 
    #as only the log probability of the ground truth is used in NLL Loss
    res = probabilities_matrix.log()
    
    return res    

    """
    >>> t = torch.tensor([[0, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[0, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    >>> res = t.log()
    >>> res
    tensor([[  -inf, 0.6931, 1.0986],
            [1.3863, 1.6094, 1.7918],
            [1.9459, 2.0794, 2.1972]])

    """

  def get_unrolled_node_id_labels(self, labels):
    
    #print(f'self.label_path_dict: \n\n{self.label_path_dict}')
    #paths_lst = [self.label_path_dict[l.item()] for l in labels ]
    #print(f'\n\n\npaths_lst: \n\n{paths_lst}')
    #numpy_paths_lst = [np.array(self.label_path_dict[l.item()]) for l in labels ]
    #print(f'\n\n\nnumpy_paths_lst: \n\n{numpy_paths_lst}')
    
    unrolled_labels = np.concatenate([np.array(self.label_path_dict[l.item()]) for l in labels ])
    #print(f'\n\n\nunrolled_labels: \n\n{unrolled_labels}')
    unrolled_labels = list(unrolled_labels)
    #print(f'\n\n\nunrolled_labels: \n\n{unrolled_labels}')

    #unrolled_labels = torch.cat([torch.tensor(self.label_path_dict[l.item()]) for l in labels ])
    total_labels_length = 0
    for l in labels:
      total_labels_length += len(self.label_path_dict[l.item()])
    #print(f'len(unrolled_labels): {len(unrolled_labels)}, total labels_path length: {total_labels_length}')
    #exit(0)
    return unrolled_labels    

    """
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> labels = torch.cat([t[i] for i in range(len(t)) ])
    >>> labels
    tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> labels.shape
    torch.Size([9])
    """

  def make_prediction(self, x, tot_prob, curr_node_id, final_preds):
   
    descendants = list(self.hier.get_direct_descendants(curr_node_id))

    #check if leaf node 
    if not descendants:
        label = self.hier.LEAF_ID_TO_NUM[curr_node_id]
        #name = self.hier.HIER_NODE_NAME[curr_node_id]
        final_preds[:, label] = tot_prob        
        return final_preds

    encoded_prompts = self.decendants_prompts_dict[curr_node_id]
    #print(f'x.type(): {x.type()} encoded_prompts.type(): {encoded_prompts.type()}')
    similarity = 100.0 * x @ encoded_prompts.t() #batch_size X num descendants
    curr_prob = self.softmax(similarity)
    curr_prob = similarity.softmax(dim=-1)
    #print(f'curr_prob: {curr_prob}')
    
    for idx, next_node_id in enumerate(descendants):
        next_node_prob = curr_prob[:, idx]
        #print(f'idx: {idx}, next_node_prob: {next_node_prob}')
        next_tot_prob = next_node_prob * tot_prob
        #print(f'next_tot_prob: {next_tot_prob}')
        final_preds = self.make_prediction(x, next_tot_prob, next_node_id, final_preds)

    return final_preds

"""
  def forward_full_tree_path(self, x):
    images = x.to(device=device)

    image_features = self.clip_model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.double()
    #print(f'image_features.type(): {image_features.type()}')

    batch_size = image_features.shape[0]
    #initial_label_prob = torch.ones(batch_size,1).to(device=device)
    initial_label_prob = torch.ones(batch_size, dtype=torch.double).to(device=device)
    preds = torch.zeros((batch_size,self.num_classes), dtype=torch.double)
    final_preds = self.make_prediction(image_features, initial_label_prob, self.root_id, preds)
    #no need to do softmax, performed inside CrossEntropyLoss criterion
    #final_preds = self.softmax(final_preds)
    return final_preds.to(device=device) 
"""

def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    #dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def CLIP_COCO_train(model, num_epochs, data_loader, img_criterion, txt_criterion, optimizer, save_frequency=5, checkpoint_name='CLIP_LoRA_COCO_checkpoint_epoch_'):
  model.train()
  loaded_epoch = 0
  #loss = None
  
  #torch.save({
  #          'epoch': loaded_epoch,
  #          'model_state_dict': model.state_dict(),
  #          'optimizer_state_dict': optimizer.state_dict(),
  #          'loss': loss
  #        }, PATH)

  #PATH = '/mnt5/nir/CLIP/interpret/DT_CLIP_CC3M_checkpoint_epoch_0020.pt.tar'
  #PATH = 'DT_CLIP_COCO_checkpoint_epoch_0001.pt.tar'
  #PATH = 'DT_LoRA_CLIP_COCO_checkpoint_epoch_0007.pt.tar'
  
  #checkpoint = torch.load(PATH)
  #DT_CLIP_model.load_state_dict(checkpoint['state_dict'])
  #optimizer.load_state_dict(checkpoint['optimizer'])
  #loaded_epoch = checkpoint['completed_epoch']
  ##loss = checkpoint['loss']
  
  print(f'CLIP_COCO_train: loaded epoch: {loaded_epoch}')

  for epoch in range(loaded_epoch,num_epochs):
    print(f'start CLIP_COCO_train epoch#: {epoch+1}')

    losses = []
    total_examples = 0
    
    for batch_idx, (full_path, images, captions) in enumerate(tqdm(data_loader)):

      #with autograd.detect_anomaly():
        
      images = images.to(device=device)
      
      #print(f'captions: {captions}')
      #print(f'captions[0]: {captions[0]}')
      #print(f'captions[0][0]: {captions[0][0]}')
      
      prompts = [f'a photo of {cap}' for cap in captions]
      #print(f'\ncaptions: {captions}\n')
      #print(f'\nprompts: {prompts}\n')


      tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)
      #tokenized_prompts.shape: torch.Size([128, 77])
      #print(f'\ntokenized_prompts.shape: {tokenized_prompts.shape}')

      #tokenized_captions = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)

      logits_per_image, logits_per_text = model(images, tokenized_prompts)
      ground_truth = torch.arange(len(images), dtype=torch.long).to(device=device)

      loss_img = img_criterion(logits_per_image, ground_truth)
      loss_txt = txt_criterion(logits_per_text, ground_truth)

      total_loss = (loss_img + loss_txt)/2
      
      losses.append(total_loss.item())
      num_examples = len(images)
      
      total_examples += num_examples
      
      #print(f'batch loss: {total_loss.item()/num_examples}, curr_loss: {total_loss.item()}, num_examples: {num_examples}')

      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
      
    
    completed_epoch = epoch + 1
    print(f'epoch loss: {sum(losses) / total_examples}')


    if (completed_epoch == num_epochs or 
                (
                    save_frequency > 0 and completed_epoch % save_frequency == 0
                )):
      
        # Saving checkpoints.
        checkpoint_dict = {
              "completed_epoch": completed_epoch,
              "state_dict": DT_CLIP_model.state_dict(),
              "optimizer": optimizer.state_dict(),
        }

        # Save only clip_model checkpoints
        checkpoint_dict = {
              "completed_epoch": completed_epoch,
              "state_dict": DT_CLIP_model.clip_model.state_dict(),
              "optimizer": optimizer.state_dict(),
        }  

        filename = checkpoint_name + f"{completed_epoch:04d}.pt.tar"
        print(f'saving model state to filename: {filename}')
        torch.save(checkpoint_dict, filename)



def check_accuracy(loader, clip_model):
    num_correct = 0
    num_samples = 0
    clip_model.eval()

    with torch.no_grad():
        #for batch in tqdm(loader):
        
        for full_path, images, captions in tqdm(loader):
           
            images = images.to(device=device)
            
            #########################
            #########################

            prompts = [f'a photo of {cap}' for cap in captions]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)
      
            logits_per_image, logits_per_text = model(images, tokenized_prompts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            labels = torch.arange(len(images), dtype=torch.long).to(device=device)
            
            _, predictions = probs.max(1)

            #########################
            #########################
                       
            num_correct += (predictions == labels).sum()
            num_samples += labels.size(0)
            #num_samples += 1
            #print(f'num_correct: {num_correct}, num_samples: {num_samples}, accuracy: {100*num_correct/num_samples:.2f}')

    clip_model.train()
    return num_correct/num_samples


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def load_and_save_clip_model_state_dict(DT_CLIP_model):
  PATH = 'DT_LoRA_CLIP_COCO_checkpoint_epoch_0005.pt.tar'
    
  checkpoint = torch.load(PATH)
  DT_CLIP_model.load_state_dict(checkpoint['state_dict'])
  print(f'loaded DT_CLIP_model checkpoint: {PATH}')

 
  # Save only clip_model checkpoints
  checkpoint_dict = {
    "completed_epoch": checkpoint["completed_epoch"],
    "state_dict": DT_CLIP_model.clip_model.state_dict(),
    "optimizer": checkpoint["optimizer"],
  }  

  filename = 'LoRA_CLIP_COCO_checkpoint_epoch_0005.pt.tar'
  print(f'saving DT_CLIP_model.clip_model state to filename: {filename}')
  torch.save(checkpoint_dict, filename)


def load_model(base_name="ViT-B/32", lora_r=-1, weight_name=""):
  clip_model, preprocess = clip.load(base_name, jit=False, lora=lora_r)
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


def main():

    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))    
    cur_dir = os.path.realpath(os.curdir)

    batch_size = args.batch_size
    #batch_size = 1
    num_epochs = args.epochs
    print(f'num_epochs: {args.epochs}')
    print(f'batch_size: {batch_size}')

    # Load the model
    vit_name = 'ViT-B/32'
    lora = 1
    #clip_model, preprocess = clip.load('ViT-B/32', device=device)
    clip_model, preprocess = load_model(base_name=vit_name, lora_r=lora)
    clip_model = clip_model.float()
        
    train_dataset = CocoCaptionsDataset(root="/mnt5/yoavkurtz/datasets/coco2017/train2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_train2017.json", transform=preprocess)
    test_dataset = CocoCaptionsDataset(root="/mnt5/yoavkurtz/datasets/coco2017/val2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_val2017.json", transform=preprocess)
   
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    save_frequency = 1

    # Loss and optimizer
    img_criterion = nn.CrossEntropyLoss()
    txt_criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()

    learning_rate = 3e-6
    #learning_rate = 1e-6
    weight_decay = 0.1

    params = [p for p in clip_model.parameters() if p.requires_grad]
    #num_params = sum([np.prod(p.size()) for p in params])
    #num_params: 87849216
    #print(f'num_params: {num_params}')
    #exit(0)
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.Adam(params, lr=learning_rate)

    # optionally resume from a checkpoint
    start_epoch = 0
    checkpoint_path = args.checkpoint_file 
    if not checkpoint_path:
        checkpoint_path = 'CLIP_LoRA_COCO_checkpoint_epoch_'

    #training loop
    print(f'starting training loop on train dataset')
    CLIP_COCO_train(clip_model, num_epochs, train_loader, img_criterion, txt_criterion, optimizer, save_frequency, checkpoint_path)

    
    #PATH = 'CLIP_LoRA_COCO_checkpoint_epoch_0005.pt.tar'
    
    #checkpoint = torch.load(PATH, map_location=torch.device(device))
    #DT_CLIP_model.load_state_dict(checkpoint['state_dict'])
    #print(f'loaded checkpoint: {PATH}')
    #print(f'starting test set check_accuracy')
    #test_accuracy = check_accuracy(test_loader, clip_model)
    
    #print(f"Accuracy on test set: {test_accuracy*100:.2f}%")
    


if __name__ == "__main__":
    main()
