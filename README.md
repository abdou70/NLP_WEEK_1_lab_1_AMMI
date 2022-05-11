# NLP_WEEK_1_lab_1_AMMI
## FIRST PART
*The* goal of this lab is to implement a language identifier (LID).

Our first model will be based on Naive Bayes.
The next function is used to load the data. Each line of the data consist of a label (corresponding to a language), followed by some text, written in that language. Here is an example of data:

```__label__de Zur Namensdeutung gibt es mehrere Varianten.```
Next, we will start implementing the Naive Bayes method. This technique is based on word counts, and we thus need to start by implementing a function to count the words and labels of our training set.

`n_examples` is the total number of examples

`n_words_per_label` is the total number of words for a given label

`label_counts` is the number of times a given label appears in the training data

`word_counts` is the number of times a word appears with a given label

Next, using the word and label counts from the previous function, we can implement the prediction function.

Here, `mu` is a regularization parameter (Laplace smoothing), and `sentence` is the list of words corresponding to the test example.
The next function will be used to evaluate the Naive Bayes model on a validation set. It computes the accuracy for a particular regularization parameter `mu`.
## SECOND PART
In this second part of the lab, we will implement a language identifier trained on the same data, but using Logistic Regression instead of Naive Bayes.
This function is used to build the dictionary, or vocabulary, which is a mapping from strings (or words) to integers (or indices). This will allow to build vector representations of documents. 

This function is used to load the training dataset, and build vector representations of the training examples. In particular, a document or sentence is represented as a bag of words. Each example correspond to a sparse vector ` x` of dimension `V`, where `V` is the size of the vocabulary. The element `j` of the vector `x` is the number of times the word `j` appears in the document.

First, we implement the softmax function. Don't forget numerical stability!

Now, let's implement the main training loop, by using stochastic gradient descent. The function will iterate over the examples of the training set. For each example, we will first compute the loss, before computing the gradient and performing the update.
The next function will predict the most probable label corresponding to example `x`, given the trained classifier `w`.
Finally, this function will compute the accuracy of a trained classifier `w` on a validation set.
