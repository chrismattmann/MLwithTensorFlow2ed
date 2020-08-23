#!/usr/bin/env bash

mkdir -p data
mkdir -p data/cache

echo "Downloading Ch4 data..."
curl -L "https://www.dropbox.com/s/naw774olqkve7sc/311.csv?dl=0" -o data/311.csv
echo "...done"

echo "Downloading Ch6 data..."
mkdir  data/word2vec-nlp-tutorial/
curl -L "https://www.dropbox.com/s/oom8kp7c3mbvyhv/labeledTrainData.tsv?dl=0" -o data/word2vec-nlp-tutorial/labeledTrainData.tsv
curl -L "https://www.dropbox.com/s/cjhix4njcjkehb1/testData.tsv?dl=0" -o data/word2vec-nlp-tutorial/testData.tsv
curl -L "https://www.dropbox.com/s/3ahq4z9376xu7yc/aclImdb_v1.tar.gz?dl=0" -o aclImdb_v1.tar.gz
tar xvzf aclImdb_v1.tar.gz
mv aclImdb* data
echo "...done"

echo  "Downloading Ch7  data..."
curl -L "https://www.dropbox.com/sh/ch7qnta0gumqjyd/AADGbuvN4pesQOtKzq3WoWuOa?dl=1" -o audio_dataset.zip
mv audio_dataset.zip data && pushd data
mkdir audio_dataset && mv audio_dataset.zip audio_dataset && pushd audio_dataset
unzip audio_dataset.zip && mv audio_dataset.zip ..
popd
popd
curl -L "https://www.dropbox.com/s/r7sbwckqvqy7uil/TalkingMachinesPodcast.wav?dl=0" -o data/TalkingMachinesPodcast.wav
echo "...done"

echo "Downloading Ch8 data..."
curl -L "https://www.dropbox.com/s/39l7unjv11hvouv/User%20Identification%20From%20Walking%20Activity.zip?dl=0" -o "data/User Identification From Walking Activity.zip"
pushd data
unzip User*.zip &&  rm -rf __MACOSX/
popd
echo "...done"

echo "Downloading Ch 10 data..."
curl -L "https://www.dropbox.com/s/5l2jt866795kjra/mobypos.txt?dl=0" -o data/mobypos.txt
echo  "...done"

echo "Downloading Ch 12 and Ch14 and Ch15 (CIFAR-10) data..."
curl -L "https://www.dropbox.com/s/gn965hqdxrytyq3/cifar-10-python.tar.gz?dl=0" -o data/cifar-10-python.tar.gz
pushd data
tar xvzf cifar-10-python.tar.gz
popd
echo "...done"

echo "Downloading Ch15 data (VGG Face)..."
curl -L "https://www.dropbox.com/s/yq5med963kfl23w/vgg_face_dataset.tar.gz?dl=0" -o data/vgg_face_dataset.tar.gz
pushd data
tar xvf vgg_face_dataset.tar.gz
popd
mkdir -p data/vgg_face
curl -L "https://www.dropbox.com/s/xk70h7w35fm8we8/vgg-face-small.tar.gz?dl=0" -o data/vgg_face/vgg-face-small.tar.gz
pushd data/vgg_face
tar xvzf vgg-face-small.tar.gz && mv vgg-face-small.tar.gz ..
popd
curl -L "https://www.dropbox.com/s/yiacsezcmjhfn9c/vgg_face_full_urls.csv?dl=0" -o data/vgg_face_full_urls.csv
curl -L "https://www.dropbox.com/s/tpg3aekqixhzhew/vgg_face_full.csv?dl=0" -o  data/vgg_face_full.csv
curl -L "https://www.dropbox.com/s/jn6jcu26tx13fzz/vgg-models.zip?dl=0" -o data/vgg-models.zip
pushd data
unzip vgg-models.zip
popd
curl -L "https://www.dropbox.com/s/k21cpcl43gmpfzv/vgg_face_weights.h5?dl=0" -o models/vgg_face_weights.h5
echo "...done"

echo "Downloading Ch16 data..."
curl -L  "https://www.dropbox.com/s/xa73rwzint8bbr3/international-airline-passengers.csv?dl=0" -o data/international-airline-passengers.csv
echo "...done"

echo "Downloading Ch18 data..."
curl -L "https://www.dropbox.com/s/9mwr58i9fsxgexd/seq2seq.tar.gz?dl=0" -o  data/seq2seq.tar.gz
pushd data
tar xvzf seq2seq.tar.gz
popd
echo "...done"

echo "Downloading Ch19 data..."
mkdir -p data/cloth_folding_rgb_vids
curl -L "https://www.dropbox.com/sh/jjozbrynieo7hf2/AAB3mHI5DcWFIl9kEmbNPIFXa?dl=0" -o data/cloth_folding_rgb_vids/cloth_folding_vids.zip
pushd data/cloth_folding_rgb_vids
unzip cloth_folding_vids.zip
popd

