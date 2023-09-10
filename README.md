# 3VL: using Trees to teach Vision & Language models compositional concepts

> Nir Yellinek⋆, Raja Giryes⋆, Leonid Karlinsky †  
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
1) For each image caption pair we first parse the ground truth caption using spaCy to get all noun phrases and the part of speech of each word in the caption.
2) Then, starting with the noun phrases, we reconstruct the full caption hierarchically to get a positive sub-caption for each level in the tree
   in the following way(we use as an example the caption "several people standing in a green field together while flying kytes"):

   
   - The first level of the tree will contain the first noun phrase as its positive text (i.e. "several people").
   - The second level of the tree will contain the text of the first and second noun phrases concatenated with some connecting word like 'and'
     (i.e. "several people and a green field").
   - The Third level of the tree will contain the text of the original caption from the start until the end of the second noun phrase
     (e.g. "several people standing in a green field").
   - if more noun phrases exist in the original caption then in a similar way the next levels of the tree will contain the text of previous nouns
     phrases concatenated to the current noun phrase with a word like 'and', and the original caption from the start until the end of the current noun phrase
   - finally, the last level of the tree will contain the text of the full original caption
      (i.e. "several people standing in a green field together while flying kytes").
3) Next, for each positive sub-caption we generate one negative caption for each Noun, Adjective, Adposition, and Verb in the sub-caption.
   *Note that we do not replace again words that appeared in previous tree levels. So information from a previous level flows without change.

   - for the above example we generate the following negative captions in each tree level
   - In the first level we generate "<p style="color:red;"one people</p>", "<p style="color:red;"several animals</p>"
   - In the second level we generate "several people and a blue field", "several people and a green forest"
   - In the third level we generate "several people gathered in a green field", "several people standing out a green field"
   - In the fourth level we generate "several people standing in a green field together while soaring kytes", "several people standing in a green field together while flying sales"
   - *Note that our negatives generation methid can generate grammatical errors sometimes

## Tree based training

<p align="center">
<img src="docs/tree_training.jpg" width="800px"/>  
<br>
For each image-text pair, we create a caption tree and for
each level of the tree we calculate the cosine similarity between
the image and all captions at that level. We apply softmax
and calculate the Cross Entropy Loss. We then sum the
cross-entropy loss over all the tree levels to get the tree loss. 
</p>
