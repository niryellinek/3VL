import argparse
import logging
import math
import os
import yaml
from PIL import Image
import numpy as np
#import clip
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageNet
from tqdm import tqdm, trange
from CLIP_linear_probe import LinearProbeClipModel, linear_probe_train
import pandas as pd
import spacy
import json
#import _pickle as cPickle
import pickle
import random
from nltk.corpus import wordnet as wn
import igraph
from igraph import Graph, EdgeSeq, plot
import plotly.graph_objects as go

nlp = spacy.load("en_core_web_sm")

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Create a CSV of CLIP encodes images and traget values')

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
                                     


class Node():
    def __init__(self, node_num=0, prompt=None, parent=None, prompt_encoding=None):
        ''' constructor ''' 
        
        self.node_num = node_num
        self.prompt = prompt
        self.prompt_encoding = prompt_encoding
        self.parent = parent
        self.children = dict()


with open('full_adjectives2.json', 'rb') as fp:
  adjectives = json.load(fp)

with open('full_nouns.json', 'rb') as fp:
  nouns = json.load(fp)

with open('full_verbs.json', 'rb') as fp:
  verbs = json.load(fp)

def get_antonym(word):
  #print(f'get_antonym word: {word}','\n')
  antonyms = []

  for syn in wn.synsets(word):
    #print(f'syn: {syn}','\n')
    for l in syn.lemmas():
      #print(f'lemma: {l}','\n')
      if l.antonyms():
        #print(f'l.antonyms(): {l.antonyms()}','\n')
        antonyms.append(l.antonyms()[0].name())
  
  antonyms = set(antonyms)
  antonyms.discard(word)
  antonyms = list(antonyms)
  #print(f'antonyms: {antonyms}\n')

  if not antonyms:
    #print(f'no antonyms found for word: {word}\n')
    return ''
  
  return random.choice(antonyms)

def get_co_hyponym(word):
  co_hyponyms = get_co_hyponym_list(word)

  if not co_hyponyms:
    #print(f'no co_hyponyms found for word: {word}\n')
    return ''

  #print(f'co_hyponyms: {co_hyponyms}\n')
  return random.choice(co_hyponyms)

def get_co_hyponym_list(word):
  #print(f'get_co_hyponym word: {word}','\n')
  co_hyponyms = []

  for syn in wn.synsets(word):
    #print(f'syn: {syn}','\n')
    hypernym_lst = syn.hypernyms()
    #print(f'hypernym_lst: {hypernym_lst}','\n')

    if hypernym_lst:
      hypernym = hypernym_lst[0]
      hyponyms = hypernym.hyponyms()
      #print(f'hyponyms: {hyponyms}','\n')

      for hypo in hyponyms:
        hypo_names = [l.name() for l in hypo.lemmas()]
        #print(f'hypo_names: {hypo_names}','\n')
        co_hyponyms.extend(hypo_names)
  
  co_hyponyms = set(co_hyponyms)
  co_hyponyms.discard(word)
  co_hyponyms = list(co_hyponyms)

  return co_hyponyms

def get_rand_word(token_text, token_pos):
  if not token_pos in ['NOUN', 'VERB', 'ADJ']:
    print(f'get_rand_word token_pos ({token_pos}) of token: {token_text} is not in [NOUN, VERB, ADJ]')
    return ''
  
  if token_pos in ['NOUN', 'VERB']:
    return get_co_hyponym(token_text)
  
  return get_antonym(token_text)

"""
def replace_root_verb(doc):
  print(f'\nreplace_root_verb orig_text: {doc}','\n')

  orig_text_lst = doc.text.split()

  for token in doc:
    print(f'token.text: {token.text}, index: {token.i}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
    if token.pos_ == 'VERB':
      rand_verb = get_co_hyponym(token.text)
      print(f'rand_verb: {rand_verb}')
"""

def get_new_sentences(orig_text, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in orig_text]
  
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  for token in orig_text:
    #print(f'token.text: {token.text}, index: {token_index}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #print(f'token.text: {token.text}, token_index: {token_index}, token.i: {token.i}, pos_: {token.pos_}\n')
    #new_text_lst.append(token.text)
    if token.pos_ in pos_to_replace:
      
      #print(f'token {token.text} index: {token_index} is {token.pos_}\n')
      #print(f'orig_text_lst: {orig_text_lst}\n')
      
      rand_word = get_rand_word(token.text, token.pos_)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'new_sentences: {new_sentences}\n')
  return new_sentences

def get_new_noun_phrases(noun_phrase):
  #print(f'get_new_noun_phrases')
  return get_new_sentences(noun_phrase, pos_to_replace=['NOUN', 'ADJ'])

def get_new_verb_phrases(verb_phrase):
  #print(f'get_new_verb_phrases')
  return get_new_sentences(verb_phrase, pos_to_replace=['VERB'])


def expand_token(token, num_expansaions=3):
  #print(f'\nexpand_token: {token.text}, pos: {token.pos_}\n')
  
  res = []
  nltk_pos = None
  if token.pos_ == 'ADJ':
    nltk_pos = 'a'
  elif token.pos_ == 'NOUN':
    nltk_pos = 'n'
  elif token.pos_ == 'VERB':
    nltk_pos = 'v'
  #print(f'nltk_pos: {nltk_pos}\n')
  new_token_captions = []
  #TODO: get synset with the correct POS 
  #TODO: get synset with the correct POS also in training get co_hyponyn and antonym
  #for syn in wn.synsets(token.text):
  #syn = wn.synsets(token.text, pos=token.pos_)
  #if sys[]
  #for syn in wn.synsets(token.text, pos=[nltk_pos]):
  
  #syn_list = wn.synsets(token.text, pos=[nltk_pos])
  syn_list = wn.synsets(token.text, pos=nltk_pos)
  if syn_list:

    syn = syn_list[0]
    """
    print(f'syn: {syn}','\n')
    print(f'syn.lexname(): {syn.lexname()}','\n')
    split = syn.lexname().split('.')
    print(f'syn.lexname().split(): {split}','\n')
    supersense = split[1]
    print(f'supersense: {supersense}','\n')
    supersense_syn = wn.synsets(supersense, pos=[nltk_pos])
    print(f'supersense_syn: {supersense_syn}','\n')
    if supersense_syn:
      print(f'supersense_syn[0].hyponyms(): {supersense_syn[0].hyponyms()}','\n')

    
    print(f'syn.definition(): {syn.definition()}','\n')
    print(f'syn.examples(): {syn.examples()}','\n')
    print(f'syn.member_holonyms(): {syn.member_holonyms()}','\n')
    print(f'syn.substance_holonyms(): {syn.substance_holonyms()}','\n')
    print(f'syn.part_holonyms(): {syn.part_holonyms()}','\n')
    print(f'syn.part_holonyms(): {syn.part_holonyms()}','\n')
    print(f'syn.member_meronyms(): {syn.member_meronyms()}','\n')
    print(f'syn.substance_meronyms(): {syn.substance_meronyms()}','\n')
    print(f'syn.part_meronyms(): {syn.part_meronyms()}','\n')
    print(f'syn.topic_domains(): {syn.topic_domains()}','\n')
    print(f'syn.in_topic_domains(): {syn.in_topic_domains()}','\n')
    print(f'syn.region_domains(): {syn.region_domains()}','\n')
    print(f'syn.in_region_domains(): {syn.in_region_domains()}','\n')
    print(f'syn.usage_domains(): {syn.usage_domains()}','\n')
    print(f'syn.in_usage_domains(): {syn.in_usage_domains()}','\n')
    print(f'syn.attributes(): {syn.attributes()}','\n')
    print(f'syn.entailments(): {syn.entailments()}','\n')
    print(f'syn.causes(): {syn.causes()}','\n')
    print(f'syn.also_sees(): {syn.also_sees()}','\n')
    print(f'syn.verb_groups(): {syn.verb_groups()}','\n')
    print(f'syn.similar_tos(): {syn.similar_tos()}','\n')
    print(f'syn.root_hypernyms(): {syn.root_hypernyms()}','\n')
    """
    ########################
    ########################
    if token.pos_ in ['NOUN', 'VERB']:
      co_hyponyms = get_co_hyponym_list(token.text)
      #TODO: add co co_hyponyms ? (all grandsons of grand father), add more relations?

      if not co_hyponyms:
        print(f'no co_hyponyms found for word: {token.text}\n')
        return res

      num_words = min(len(co_hyponyms),num_expansaions)
      res = random.sample(co_hyponyms, num_words)

      #print(f'result: {res}')
    ########################
    ########################

    if token.pos_ == 'ADJ':
      similar_to_names = []
      similar_to = syn.similar_tos()

      for sim in similar_to:
        similar_to_names.extend( [l.name() for l in sim.lemmas()] )
      
      #print(f'similar_to_names: {similar_to_names}','\n')
  
      similar_to_names = set(similar_to_names)
      similar_to_names.discard(token.text)
      similar_to_names = list(similar_to_names)

      if not similar_to_names:
        print(f'no similar_to_names found for word: {token.text}\n')
        return res

      num_words = min(len(similar_to_names),num_expansaions)
      res = random.sample(similar_to_names, num_words)
      
      attributes = syn.attributes()
      if attributes:
        attributes_names = []
        for attr in attributes:
          attributes_names.extend( [l.name() for l in attr.lemmas()] )
        
        attr = random.choice(attributes_names)
        #print(f'attr: {attr}')
        res = [w if attr in w else f'{w} {attr}' for w in res]

      #print(f'result: {res}')
    
    return res


def get_more_caption_tokens(predicted_token, correct_token):
  #print(f'get_more_caption_tokens')
  #def common_hypernyms(self, other)
  #def lowest_common_hypernyms(self, other, simulate_root=False, use_min_depth=False):
  #def path_similarity(self, other, verbose=False, simulate_root=True):
  #def res_similarity(self, other, ic, verbose=False):
  #def jcn_similarity(self, other, ic, verbose=False):
  #def lin_similarity(self, other, ic, verbose=False):
  #new_tokens_texts = []
  predicted_expansion = expand_token(predicted_token)
  correct_expansion = expand_token(correct_token)
  #new_tokens_texts.extend(predicted_expansion)
  #new_tokens_texts.extend(correct_expansion)
  return predicted_expansion, correct_expansion

def get_new_texts_list(orig_text, tok_index, new_words):
  res = []
  if not new_words:
    return res
  orig_text_lst = [token.text for token in orig_text]

  for word in new_words:
    new_text_lst = orig_text_lst.copy()
    new_text_lst[tok_index] = word
    new_text = ' '.join(new_text_lst)
    res.append(new_text)

  return res

def expand_caption(predicted_caption, correct_caption):
  predicted_caption = predicted_caption.lower()
  correct_caption = correct_caption.lower()
  
  predicted_doc = nlp(predicted_caption)
  correct_doc = nlp(correct_caption)

  #orig_predicted_text_tokens_lst = [token.text for token in predicted_caption]
  #orig_correct_text_tokens_lst = [token.text for token in correct_caption]
  token_index = 0
  new_captions = []
  #new_predicted_captions = []
  #new_correct_captions = []
  for predicted_token, correct_token in zip(predicted_doc, correct_doc):
    #print(f'\npredicted_token.text: {predicted_token.text}, correct_token.text: {correct_token.text}')
    #print(f'\npredicted_token.pos_: {predicted_token.pos_}, correct_token.pos_: {correct_token.pos_}')
    if predicted_token.text != correct_token.text:
      new_predicted_tokens_texts, new_correct_tokens_texts = get_more_caption_tokens(predicted_token, correct_token)
      
      new_predicted_captions = get_new_texts_list(predicted_doc, token_index, new_predicted_tokens_texts)
      new_correct_captions = get_new_texts_list(correct_doc, token_index, new_correct_tokens_texts)
      new_captions.extend(new_predicted_captions)
      new_captions.extend(new_correct_captions)

    token_index += 1

  new_captions.append(predicted_caption)
  new_captions.append(correct_caption)
  new_captions = list(set(new_captions))
  return new_captions

#  for token in caption:
#    if word == token.text:



def make_annotations(x_pos , y_pos, texts, font_size=10, font_color='rgb(250,250,250)'):
    L=len(x_pos)
    if len(texts)!=L:
        raise ValueError('The lists x_pos and texts must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=texts[k], 
                x=x_pos, y=y_pos,
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations

def get_box_text(text, max_line_width=20, break_every=3):
  #TODO use textwrap.TextWrapper(width=50) ??
  words = text.split(' ')
  ###
  box_text = ""
  word_count = 0
  line_width = 0
  for word in words:
    word_width = len(word)

    if line_width + word_width > max_line_width:
      #box_text += "\n"
      box_text += "<br>"
      line_width = 0
    
    box_text += word + " "
    line_width += word_width + 1
    #word_count += 1
    #if break_every == word_count:
    #  #box_text += "\n"
    #  box_text += "<br>"
    #  word_count = 0

  #print(f'\n\nget_box_text - \ntext: \n{text}, \nbox_text: \n{box_text}')
  return box_text


def get_caption_tree(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
  #sub_sentences = []

  #for token in doc:
  #  print(token.text, token.dep_, token.head.text, token.pos_)

  #class Node():
  #  def __init__(self, node_num=0, prompt=None, prompt_encoding=None):

  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  node_texts.append("")
  curr_path_node = root
  next_path_node = None
  for chunk in doc.noun_chunks:
    #print(f'chunk.text: {chunk.text}, root: {chunk.root.text}, root.dep_: {chunk.root.dep_}, root.head: {chunk.root.head.text},  root.head.dep_: {chunk.root.head.dep_}\n')
    #TODO: f'a photo of {chunk.text}'
    #sub_sentences.append(chunk.text)
    #add child node under curr_node
    true_label_path.append(node_index)
    edges.append([curr_path_node.node_num, node_index])
    next_path_node = Node(node_num=node_index, prompt=chunk.text, parent=curr_path_node)
    node_texts.append(get_box_text(chunk.text))
    all_tree_nodes.append(next_path_node)
    
    curr_path_node.children[node_index] = next_path_node
    node_index += 1

    new_noun_phrases = get_new_noun_phrases(chunk)
    for new_text in new_noun_phrases:
      tmp_node = Node(node_num=node_index, prompt=new_text, parent=curr_path_node)
      node_texts.append(get_box_text(new_text))
      all_tree_nodes.append(tmp_node)
      curr_path_node.children[node_index] = tmp_node
      edges.append([curr_path_node.node_num, node_index])
      node_index += 1

    curr_path_node = next_path_node
    #if  chunk.root.dep_ == 'nsubj':
    #  print(chunk.root.head.text,  chunk.root.head.dep_, chunk.root.head.pos_)
    #  print(f'chunk.root.head.lefts: {[token.text for token in chunk.root.head.lefts]}')
    #  print(f'chunk.root.head.rights: {[token.text for token in chunk.root.head.rights]}')
  
  #sub_sentences.append(original_sentence)
  true_label_path.append(node_index)
  next_path_node = Node(node_num=node_index, prompt=original_caption, parent=curr_path_node)
  node_texts.append(get_box_text(original_caption))
  all_tree_nodes.append(next_path_node)
  curr_path_node.children[node_index] = next_path_node
  edges.append([curr_path_node.node_num, node_index])
  node_index += 1
  
  new_sentences = get_new_verb_phrases(doc)
  for new_sentence in new_sentences:
      tmp_node = Node(node_num=node_index, prompt=new_sentence, parent=curr_path_node)
      #node_texts.append(new_sentence)
      node_texts.append(get_box_text(new_sentence))
      ###
      all_tree_nodes.append(tmp_node)
      curr_path_node.children[node_index] = tmp_node
      edges.append([curr_path_node.node_num, node_index])
      node_index += 1
  
  #######################
  #######################
  #plotly_plot_tree(all_tree_nodes, edges, original_caption, node_texts)
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts

def plotly_plot_tree(all_tree_nodes, edges, original_caption, node_texts, image_file_name=""):


  #######################
  #######################
  nr_vertices = len(all_tree_nodes)
  v_label = list(map(str, range(nr_vertices)))
  #G = Graph.Tree(nr_vertices, 2) # 2 stands for children number
  G = Graph(n=nr_vertices, edges=edges)
  #lay = G.layout('rt', [0])
  lay = G.layout_reingold_tilford(root=[0])
  #plot(G, layout=lay)

  
  position = {k: lay[k] for k in range(nr_vertices)}
  Y = [lay[k][1] for k in range(nr_vertices)]
  M = max(Y)

  es = EdgeSeq(G) # sequence of edges
  E = [e.tuple for e in G.es] # list of edges

  L = len(position)
  Xn = [position[k][0] for k in range(L)]
  Yn = [2*M-position[k][1] for k in range(L)]
  Xe = []
  Ye = []
  for edge in E:
    Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

  labels = v_label
  #######################
  #######################
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=Xe,
                   y=Ye,
                   mode='lines',
                   line=dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'
                   ))
  fig.add_trace(go.Scatter(x=Xn,
                  y=Yn,
                  #mode='markers',
                  mode='text',
                  #mode='markers+text',
                  name=original_caption,
                  marker=dict(symbol='circle-dot',
                                #size=18,
                                size=12,
                                #color='#6175c1',    #'#DB4551',
                                color='#dfe3f2',    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                  #text=labels,
                  text=node_texts,
                  #textposition="bottom center",
                  #textposition="middle center",
                  textposition="top center",
                  hoverinfo='text',
                  opacity=0.8
                  ))
  #######################
  #######################
  axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

  fig.update_layout(title=original_caption,
              annotations=make_annotations(Xn, Yn, node_texts),
              font_size=10,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              width=1000,
              #margin=dict(l=50, r=40, b=85, t=100),
              #margin=dict(l=50, r=40, b=55, t=100),
              #margin=dict(l=50, r=40, b=35, t=100),
              margin=dict(l=50, r=40, b=35, t=50),
              #margin=dict(l=40, r=40, b=185, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)'
  )
  fig.update_layout(title_text=original_caption)
  #fig.update_layout(title_xanchor="center")
  #fig.update_layout(title_xanchor="right")
  fig.update_layout(title_xanchor="left")
  fig.update_layout(title_x=0.3)
  #fig.update_layout(title_font_size=<VALUE>)
  fig.update_traces(automargin=True, selector=dict(type='pie'))

  #fig.update_annotations(width=1)

  #######################
  #######################
  
  image_dir = "plotly_images"

  if not os.path.exists(image_dir):
    os.mkdir(image_dir)
  
  if image_file_name:
    file_path = image_dir + "/" + image_file_name
  else:
    file_path = image_dir + "/" + "default_fig.png"

  fig.write_image(file_path)
  return file_path
  #fig.show()
  
  #######################
  #######################

def main():
  #original_sentence = "flowers in the shape of wind turbines"
  #original_sentence = "neo soul artist speaks onstage during tv programme"
  #original_sentence = "a boxer dog leaping through the air on a snow filled country lane"
  #original_sentence = "with the right equipment , mowing the lawn is pretty much a game"
  #get_sub_sentences(original_sentence)
  #print(spacy.explain('acl'))

  #predicted_caption = "a big sheep"
  #correct_caption = "a small sheep"
  #predicted_caption = "full ski racks"
  #correct_caption = "empty ski racks"
  #predicted_caption = "A lunch shoebox"
  #correct_caption = "A lunch bowl"
  predicted_caption = "A group of friends playing a motion mouse video game"
  correct_caption = "A group of friends playing a motion controlled video game"
  #expanded_captions = expand_caption(predicted_caption, correct_caption)
  #print(f'\nexpanded_captions are:\n\n {expanded_captions}')
  get_caption_tree(correct_caption)
        
   
        
if __name__ == "__main__":
    main()


    


