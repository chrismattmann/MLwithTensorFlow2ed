Machine Learning with TensorFlow, 2nd Edition
=============================================
This is the code repository for the 2nd edition of [Manning Publications'](http://manning.com/)
[Machine Learning with TensorFlow](https://www.manning.com/books/machine-learning-with-tensorflow-second-edition?a_aid=5700fc87&a_bid=1e05f0bb) 
written by [Chris Mattmann](http://twitter.com/chrismattmann/).

The code in this repository is mostly [Jupyter Notebooks](http://jupyter.org/) that correspond
to the numbered listings in each chapter of the book. The code has beeen tested with 
[TensorFlow 1.14.0](https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs) but there 
is an experimental branch for [TensorFlow 2.x](http://github.com/chrismattmann/tree/tensorflow2).
Please be patient as we port the examples to TF 2.x.

## Pre-requisites

Though the book has [TensorFlow](http://twitter.cm/Tensorflow) in the name, the book is also
just as machine about generalized machine learning and its theory, and the suite of frameworks
that also come in handy when dealing with machine learning. The requirements for running the 
notebooks are below. You should PIP install them using your favorite Python. The examples from
the book have been shown to work in Python 2.7, and Python 3.7. I didn't have time to test all 
of them but we are happy to receive PRs for things we've missed.

 * TensorFlow 
 * Jupyter
 * Pandas - for data frames and easy tabular data manipulation
 * NumPy, SciPy
 * Matplotlib
 * [NLTK](http://nltk.org/) - for anything text or NLP (such as Sentiment Analysis from Chapter 6)
 * TQDM - for progress bars
 * SKLearn - for various helper functions
 * [Bregman Toolkit](https://github.com/bregmanstudio/BregmanToolkit/) (for audio examples in Chapter 7)
 * [Tika](http://github.com/chrismattmann/tika-python)
 * [Ystockquote](https://github.com/cgoldberg/ystockquote)

## Data Requirements

You will generate lots of data when running the notebooks in particular building models. But to train and
build those models you will also need data. I have created an easy [DropBox](http://dropbox.com/) folder
for you to pull input data for use in training models from the book. Access the DropBox folder 
[here](https://www.dropbox.com/sh/abjqqcwuzx2mttd/AADIM01H44Y-tdAHXUHt5ZWFa?dl=0).

The pointers below let you know what data you need for what chapters, and where to put it. Unless otherwise
not specified, the data should be placed into the `data` folder. Note that as you are running the notebooks
the notebooks will generate TF models and write them and `checkpoint` files to the `models/` folder.

## Data Input requirements

### Chapter 4
 - `data/311.csv`

### Chapter 6
 - `data/word2vec-nlp-tutorial/labeledTrainData.tsv`
 - `data/word2vec-nlp-tutorial/testData.tsv`
 - `data/aclImdb/test/neg/`
 - `data/aclImdb/test/pos/`

### Chapter 7
 - `data/audio_dataset/`

### Chapter 8
 - `data/User Identification From Walking Activity/`

### Chapter 10
 - `data/mobypos.txt`

### Chapter 12
 - `data/cifar-10-batches-py`
 - `data/MNIST_data/` (if you try the MNIST extra example)

Questions, comments?
===================
Send them to [Chris A. Mattmann](mailto:chris.mattmann@gmail.com).

Contributors
============
* Chris A. Mattmann
* Rob Royce (`tensorflow2` branch)
* Philip Southam (Dockerfile build)

License
=======
[Apache License, version 2](http://www.apache.org/licenses/LICENSE-2.0)
