# MLwithTensorFlow2-2ed
---

Exercises and listings for ["Machine Learning with TensorFlow 2nd Edition"](https://www.manning.com/books/machine-learning-with-tensorflow-second-edition) using TensorFlow v2. The original listings, which use TensorFlow v1 are available on [Chris Mattmann's GitHub page](https://github.com/chrismattmann/MLwithTensorFlow2ed)

## Methodology
---

Tensorflow v2 has introduced a variety of breaking changes. Some of these changes affect workflow, where others require adopting entirely new paradigms. [Eager Execution](https://www.tensorflow.org/guide/eager), for example, requires a change from Declarative to Imperative Programming. We no longer use `Placeholder`'s, and we rely on different libraries to accomplish tasks that have been deprecated in v2. The examples, exercises, and listings in the text and on Chris' GitHub page will be translated from TensorFlow to TensorFlow2 using the following methodologies:

- We use the official [TF v1 to TF v2 migration guide](https://www.tensorflow.org/guide/migrate) wherever possible.
- When the migration guide does not suffice, we attempt to _replicate results_ attained in the text and in Chris' github repository.



For anyone interested in how a more elaborate project in Tensorflow would be migrated from v1 to v2, we encourage you to check out the migration guide linked above, and also see whether the official [upgrade script](https://www.tensorflow.org/guide/upgrade) would work for your situation. Note that we are not attempting to use the upgrade script in this repository for two reasons:

- 1 ["The conversion script automates as much as possible, but there are still syntactical and stylistic changes that cannot be performed by the script."](https://www.tensorflow.org/guide/upgrade)
- 2 There is value for the author (of this repository) in fully examining the changes from TF v1 to TF v2 (i.e., this is a learning experience).

## Contributions
---

Contributions are more then welcome. This repository and the contents there within are in the public domain, subject to the terms and conditions (if any) laid out by Manning Publications (distributor), Chris Mattmann (author), or any other binding agreements between the user of this repository and the proprietor of the source material.

## Disclaimer
---

The users of this repository shall have no expectation in terms of correctness or thoroughness. The author(s) have attempted to correctly translate the original source material, but there are many reasons why the v2 source might be wildly different from that of the v1 source. If there are issues with the code and/or documentation, please submit a pull request and/or contact the owner of this repository.

## Asking for help
---

We will make reasonable attempts to address issues in the code, but please be aware that the viability of a solution will be judged by the contents of its output. The Jupyter Notebooks herein capture the output of the execution and persist that output when pushed to Github. We _do not_ plan to update this repository as new Tensorflow versions are released. This repository is meant to satisfy the demand for v2 translations and not to be an upto date implementation.
