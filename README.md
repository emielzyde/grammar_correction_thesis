# Automated Grammar Correction #

Grammatical error correction (GEC), the task of correcting noisy, ungrammatical data, is a challenging but crucial natural language processing (NLP) task. It can be used, for example, to provide feedback to non-native English learners attempting to learn English. As is common in machine learning, this task benefits from the availability of a large corpus of erroneous sentences and their grammatically-correct counterparts. Constructing such datasets with high-quality error annotations is, however, labour-intensive and the lack of sufficient data may hamper existing methods.

Building on work by Xie et al. (2018) and Kasewa et al. (2018), amongst others, we will investigate which methods of generating additional (artificial data are most effective in addressing the data sparsity bottleneck in grammatical error correction. In doing so, we will propose and evaluate novel methods of

applying noise to clean, grammatically-correct data. Such methods will allow us to transform clean text, which is abundantly available, into additional training examples for GEC. Initially, we will seek to replicate the results of Xie et al. (2018). Thereafter, we propose several extensions (yet to be decided on).  To evaluate the performance of our models, we will seek to identify several (heuristic) metrics, measuring the extent of grammatical incorrectness of a

transformed sentence and the extent to which it deviates from its source sentence. Furthermore, we will measure the extent to which the additional data affects the performance on the task of error detection, whilst holding the error
detection model fixed. We do not aim to maximise the overall performance of an end-to-end GEC pipeline, but rather seek to investigate which data augmentation methods could most effectively address the data bottleneck.

Overall, the aim of the project is to present novel, efficient and scalable systems of generating large amounts of artificial noisy data. Combined with developments in the field of error detection, this can contribute to increased performance and applicability of GEC systems.