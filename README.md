# 3VL: using Trees to teach Vision & Language models compositional concepts

> Nir Yellinek<sup>⋆</sup>, Leonid Karlinsky<sup>†</sup>, Raja Giryes<sup>⋆</sup>
> 
> ⋆ Tel-Aviv University, † MIT-IBM Watson AI Lab
>
> Vision & Language models (VLMs) have proved effective
at aligning image and text representations, producing superior
zero-shot results when transferred to many downstream
tasks. However, these representations suffer some key shortcomings
in Compositional Language Concepts (CLC) understanding
such as recognizing objects’ attributes, states, and relations
between different objects. Moreover, VLMs typically
have poor interpretability, making it challenging to debug and
mitigate compositional-understanding failures. In this work,
we introduce the Tree-augmented Vision & Language (3VL)
model architecture and training technique. By expanding the
text of an arbitrary image-text pair into a hierarchical tree
structure using language analysis tools, 3VL allows inducing
this structure into the visual representation learned by
the model, enhancing its compositional reasoning and interpretability.
Overall, we provide a novel approach for improving
performance and explainablity of VLMs.

<p align="center">
<img src="teaser.jpg" width="800px"/>  
<br>
Tree-augmented Vision & Language (3VL) model architecture and training
technique allows for rich exploration of the text space using several levels of incremental text augmentation from coarse to fine-grained. 
</p>

## Caption tree generation
1) For each image caption pair we first parse the caption using spaCy to get all noun phrases and part of speech tags.
   
2) Then, we reconstruct the full caption hierarchically to get a positive sub-caption for each level in the tree
   in the following way:

   for the caption: "several people standing in a green field together while flying kytes"
   - The first level of the tree will contain the first noun phrase as its positive text
     (i.e. "several people").
   - The second level of the tree will contain the text of the first and second noun phrases concatenated with some connecting word like 'and'
     (i.e. "several people and a green field").
   - The Third level of the tree will contain the text of the original caption from the start until the end of the second noun phrase
     (e.g. "several people standing in a green field").
   - if more noun phrases exist in the original caption then in a similar way the next levels of the tree will contain the text of previous nouns
     phrases concatenated to the current noun phrase with a word like 'and', and the original caption from the start until the end of the current noun phrase
   - finally, the last level of the tree will contain the text of the full original caption
      (i.e. "several people standing in a green field together while flying kytes").
     
4) Next, for each positive sub-caption we generate one negative caption for each Noun, Adjective, Adposition, and Verb in the sub-caption.

   **_Note_** that we do not replace again words that appeared in previous tree levels. So information from a previous level flows without change.

   for the above example we generate the following negative captions:
   - In the first level we generate "one people", "several animals" (for the positive text "several people").
   - In the second level we generate "several people and a blue field", "several people and a green forest"
      (for the positive text "several people and a green field").
   - In the third level we generate "several people gathered in a green field", "several people standing out a green field" (for the positive text "several people standing in a green field").
   - In the fourth level we generate "several people standing in a green field together while soaring kytes", "several people standing in a green field together while flying sales"
(for the positive text "several people standing in a green field together while flying kytes").
     **_Note_** that our automated negatives generation method can generate grammatical errors sometimes

### Negatives generation

In each negative generation we replace one word of the positive caption, either a Noun, Adjective, Adposition, or Verb. 
Where each negative word is generated as:
1. An opposite (Antonym) of the positive word using FLAN-T5 with prompt (e.g. "find an opposite for the word:
...”)
2.  If an opposite is not found then we generate a co-hyponym<sup>*</sup> of the positive word using WordNet
3.  If a co-hyponym[^1] <sup>*</sup> is not found then we generate a word to fill the masked positive word using    FLAN-T5 (the token 'extra_id_0' replaces the positive word in prompt).

<sup>*</sup>  A hyponym is a word that have a more specific meaning than its hypernym (the direct ancestor in the wordnet tree). For example, ’apple’ is a hyponym of ’fruit’, and ’fruit’ is the hypernym of ’apple’. 
Co-hyponyms are words that share the same hypernym in the wordnet tree (e.g. ’apple’ and ’banana’ are co-hyponyms as they share the hypernym ’fruit’; the words ’car’ and ’motorcycle’, ’blue’ and ’yellow’ are co-hyponyms as well.

[^1]: A hyponym is a word that have a more specific meaning than its hypernym (the direct ancestor in the wordnet tree). For example, ’apple’ is a hyponym of ’fruit’, and ’fruit’ is the hypernym of ’apple’. 
Co-hyponyms are words that share the same hypernym in the wordnet tree (e.g. ’apple’ and ’banana’ are co-hyponyms as they share the hypernym ’fruit’; the words ’car’ and ’motorcycle’, ’blue’ and ’yellow’ are co-hyponyms as well.

***see above example for caption tree creation and negatives generation for the caption "several people standing in a green field together while flying kytes".***

  
## Tree based training

<p align="center">
<img src="docs/tree_training.jpg" width="800px"/>  
<br>
For each image-caption pair, we first create a caption tree. Then, for each level of the tree we calculate the cosine similarity scores between the image and all captions at that level and calculate the Cross Entropy Loss. The final tree loss is the sum of losses over all tree levels.
</p>


## Results


