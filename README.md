# Learning Representations for Items:

This repo contains an example of the implementation of Mikolov's paper, "Distributed Representations of Words and Phrases and their Compisitionality", for items that appear in the same context.  
Assuming that we are able to generate a dataset that contains a set of target, context items, this template can be used to train the NN that will learn an embedding for each item. These embeddings can then be used to return other similar items or in clustering analysis to generate segmentations of the population based on the items they interacted with.

The input dataset needs to contain the input, output items thare seen in the same context. Each data point represents an occurence of this item/context item. These could be set of purchases: for each transaction, we have the set of items purchased together or web searches:for each web search session, the set of items searched in the same session. The co-occurence of these items signals that they are perceived as similar/substitute or complementary products. 
