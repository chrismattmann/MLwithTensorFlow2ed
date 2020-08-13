#!/usr/bin/env bash
set -e
set -x
mkdir -p data
mkdir -p data/cache

echo "Downloading Ch4 data..."
curl -L "https://www.dropbox.com/s/11331ycu0fmpb5h/311.csv.xz?dl=0" -o data/311.csv.xz
curl -L "https://www.dropbox.com/s/offtc9mul4eqbz2/311.csv.xz.sha256?dl=0" -o data/311.csv.xz.sha256
pushd data
sha256sum -c 311.csv.xz.sha256 && rm 311.csv.xz.sha256
unxz 311.csv.xz
popd
echo "...done"

echo "Downloading Ch6 data..."
curl -L "https://www.dropbox.com/s/f98keuexrnvc5qo/word2vec-nlp-tutorial.tar.xz?dl=0" -o word2vec-nlp-tutorial.tar.xz
curl -L "https://www.dropbox.com/s/6b7q40pdfohi8ri/word2vec-nlp-tutorial.tar.xz.sha256?dl=0" -o word2vec-nlp-tutorial.tar.xz.sha256
sha256sum -c word2vec-nlp-tutorial.tar.xz.sha256 && rm word2vec-nlp-tutorial.tar.xz.sha256
tar -C data -xJvf word2vec-nlp-tutorial.tar.xz && rm word2vec-nlp-tutorial.tar.xz
curl -L "https://www.dropbox.com/s/hbryifzukwa5uh5/aclImdb_v1.tar.xz?dl=0" -o aclImdb_v1.tar.xz
curl -L "https://www.dropbox.com/s/78vf3bkcqu6gn95/aclImdb_v1.tar.xz.sha256?dl=0" -o aclImdb_v1.tar.xz.sha256
sha256sum -c aclImdb_v1.tar.xz.sha256 && rm aclImdb_v1.tar.xz.sha256
tar -C data -xJvf aclImdb_v1.tar.xz && rm aclImdb_v1.tar.xz
echo "...done"

echo  "Downloading Ch7  data..."
curl -L "https://www.dropbox.com/s/qenizussmpf6o90/audio_dataset.tar.xz?dl=0" -o audio_dataset.tar.xz
curl -L "https://www.dropbox.com/s/zfqnrjmqd85oynn/audio_dataset.tar.xz.sha256?dl=0" -o audio_dataset.tar.xz.sha256
sha256sum -c audio_dataset.tar.xz.sha256 && rm audio_dataset.tar.xz.sha256
tar -C data -xJvf audio_dataset.tar.xz && rm audio_dataset.tar.xz
curl -L "https://www.dropbox.com/s/uuv5nk6hqx5yv5n/TalkingMachinesPodcast.wav.xz?dl=0" -o data/TalkingMachinesPodcast.wav.xz
curl -L "https://www.dropbox.com/s/v07jp2js999ciyi/TalkingMachinesPodcast.wav.xz.sha256?dl=0" -o data/TalkingMachinesPodcast.wav.xz.sha256
pushd data
sha256sum -c TalkingMachinesPodcast.wav.xz.sha256 && rm TalkingMachinesPodcast.wav.xz.sha256
unxz TalkingMachinesPodcast.wav.xz
popd
echo "...done"

echo "Downloading Ch8 data..."
curl -L "https://www.dropbox.com/s/um35z0jc338mi4p/user-identification-from-walking-activity.tar.xz?dl=0" -o user-identification-from-walking-activity.tar.xz
curl -L "https://www.dropbox.com/s/shdaavem2q47laj/user-identification-from-walking-activity.tar.xz.sha256?dl=0" -o user-identification-from-walking-activity.tar.xz.sha256
sha256sum -c user-identification-from-walking-activity.tar.xz.sha256 && rm user-identification-from-walking-activity.tar.xz.sha256
tar -C data -xJvf user-identification-from-walking-activity.tar.xz && rm user-identification-from-walking-activity.tar.xz
echo "...done"

echo "Downloading Ch 10 data..."
curl -L "https://www.dropbox.com/s/lfmbvahvak0a1yr/mobypos.txt.xz?dl=0" -o data/mobypos.txt.xz
curl -L "https://www.dropbox.com/s/vyzqx1fn71jhahz/mobypos.txt.xz.sha256?dl=0" -o data/mobypos.txt.xz.sha256
pushd data
sha256sum -c mobypos.txt.xz.sha256 && rm mobypos.txt.xz.sha256
unxz mobypos.txt.xz
popd
echo  "...done"

echo "Downloading Ch 12 and Ch14 and Ch15 (CIFAR-10) data..."
curl -L "https://www.dropbox.com/s/58ffbllifzt1n8p/cifar-10-python.tar.xz?dl=0" -o cifar-10-python.tar.xz
curl -L "https://www.dropbox.com/s/hjdpljhwn2rlb64/cifar-10-python.tar.xz.sha256?dl=0" -o cifar-10-python.tar.xz.sha256
sha256sum -c cifar-10-python.tar.xz.sha256 && rm cifar-10-python.tar.xz.sha256
tar -C data -xJvf cifar-10-python.tar.xz && rm cifar-10-python.tar.xz
echo "...done"

echo "Downloading Ch15 data (VGG Face)..."
curl -L "https://www.dropbox.com/s/nut408obn99la12/vgg_face_dataset.tar.xz?dl=0" -o vgg_face_dataset.tar.xz
curl -L "https://www.dropbox.com/s/9hspp5mr8qpbata/vgg_face_dataset.tar.xz.sha256?dl=0" -o vgg_face_dataset.tar.xz.sha256
sha256sum -c vgg_face_dataset.tar.xz.sha256 && rm vgg_face_dataset.tar.xz.sha256
tar -C data -xJvf vgg_face_dataset.tar.xz && rm vgg_face_dataset.tar.xz
curl -L "https://www.dropbox.com/s/5nv0pw367wlfkrc/vgg_face-small.tar.xz?dl=0" -o vgg_face-small.tar.xz
curl -L "https://www.dropbox.com/s/rbj6m8gf4xjebb0/vgg_face-small.tar.xz.sha256?dl=0" -o vgg_face-small.tar.xz.sha256
sha256sum -c vgg_face-small.tar.xz.sha256 && rm vgg_face-small.tar.xz.sha256
tar -C data -xJvf vgg_face-small.tar.xz && rm vgg_face-small.tar.xz
curl -L "https://www.dropbox.com/s/tyukhq0r3cgk5xx/vgg_face_full_urls.csv.xz?dl=0" -o data/vgg_face_full_urls.csv.xz
curl -L "https://www.dropbox.com/s/ea503l12x02wtse/vgg_face_full_urls.csv.xz.sha256?dl=0" -o data/vgg_face_full_urls.csv.xz.sha256
curl -L "https://www.dropbox.com/s/i6wncx0fs1k51nf/vgg_face_full.csv.xz?dl=0" -o  data/vgg_face_full.csv.xz
curl -L "https://www.dropbox.com/s/kapt1u0dcrxmmta/vgg_face_full.csv.xz.sha256?dl=0" -o data/vgg_face_full.csv.xz.sha256
pushd data
sha256sum -c vgg_face_full_urls.csv.xz.sha256 && rm vgg_face_full_urls.csv.xz.sha256
unxz vgg_face_full_urls.csv.xz
sha256sum -c vgg_face_full.csv.xz.sha256 && rm vgg_face_full.csv.xz.sha256
unxz vgg_face_full.csv.xz
popd
curl -L "https://www.dropbox.com/s/tl27ja252tpi9sy/vgg-models.tar.xz?dl=0" -o vgg-models.tar.xz
curl -L "https://www.dropbox.com/s/bfmr80r7jaq0v88/vgg-models.tar.xz.sha256?dl=0" -o vgg-models.tar.xz.sha256
sha256sum -c vgg-models.tar.xz.sha256 && rm vgg-models.tar.xz.sha256
tar -C data -xJvf vgg-models.tar.xz && rm vgg-models.tar.xz
echo "...done"

echo "Downloading Ch16 data..."
curl -L  "https://www.dropbox.com/s/8a207h2klogtpep/international-airline-passengers.csv?dl=0" -o data/international-airline-passengers.csv
curl -L  "https://www.dropbox.com/s/j41jldmcglrsgbo/international-airline-passengers.csv.sha256?dl=0" -o data/international-airline-passengers.csv.sha256
pushd data
sha256sum -c international-airline-passengers.csv.sha256 && rm international-airline-passengers.csv.sha256
popd
echo "...done"

echo "Downloading Ch17 data..."
curl -L "https://www.dropbox.com/s/hr28xhat69kmu8j/LibriSpeech.tar.bz2.partaa?dl=0" -o LibriSpeech.tar.bz2.partaa
curl -L "https://www.dropbox.com/s/namsr3tacyxctyj/LibriSpeech.tar.bz2.partab?dl=0" -o LibriSpeech.tar.bz2.partab
curl -L "https://www.dropbox.com/s/x2hjw2p4jay08nz/LibriSpeech.tar.bz2.partac?dl=0" -o LibriSpeech.tar.bz2.partac
curl -L "https://www.dropbox.com/s/jy70yoqmpzqoi05/LibriSpeech.tar.bz2.partad?dl=0" -o LibriSpeech.tar.bz2.partad
curl -L "https://www.dropbox.com/s/qwzj2d4cuovohz1/LibriSpeech.tar.bz2.partae?dl=0" -o LibriSpeech.tar.bz2.partae
curl -L "https://www.dropbox.com/s/bndq8zp4udhhd6g/LibriSpeech.tar.bz2.partaf?dl=0" -o LibriSpeech.tar.bz2.partaf
curl -L "https://www.dropbox.com/s/ar0qx5rjoto5iow/LibriSpeech.tar.bz2.partag?dl=0" -o LibriSpeech.tar.bz2.partag
curl -L "https://www.dropbox.com/s/fhcen1irrtez1v7/LibriSpeech.tar.bz2.partah?dl=0" -o LibriSpeech.tar.bz2.partah
curl -L "https://www.dropbox.com/s/5fydv42rcelwt9j/LibriSpeech.tar.bz2.partai?dl=0" -o LibriSpeech.tar.bz2.partai
curl -L "https://www.dropbox.com/s/dt3v1x2pkbulzgx/LibriSpeech.tar.bz2.partaj?dl=0" -o LibriSpeech.tar.bz2.partaj
curl -L "https://www.dropbox.com/s/2fe9a8g8fmjd2ao/LibriSpeech.tar.bz2.partak?dl=0" -o LibriSpeech.tar.bz2.partak
curl -L "https://www.dropbox.com/s/362rllp8fy3xvzb/LibriSpeech.tar.bz2.partal?dl=0" -o LibriSpeech.tar.bz2.partal
curl -L "https://www.dropbox.com/s/kxoow98bdqblswq/LibriSpeech.tar.bz2.partam?dl=0" -o LibriSpeech.tar.bz2.partam
curl -L "https://www.dropbox.com/s/u7s8itp8ocyqodn/LibriSpeech.tar.bz2.partan?dl=0" -o LibriSpeech.tar.bz2.partan
curl -L "https://www.dropbox.com/s/5u5g6bm3cruhiep/LibriSpeech.tar.bz2.partao?dl=0" -o LibriSpeech.tar.bz2.partao
curl -L "https://www.dropbox.com/s/6e5njq4x5756ttx/LibriSpeech.tar.bz2.partap?dl=0" -o LibriSpeech.tar.bz2.partap
curl -L "https://www.dropbox.com/s/pdxp9jb27xz4wpx/LibriSpeech.tar.bz2.partaq?dl=0" -o LibriSpeech.tar.bz2.partaq
curl -L "https://www.dropbox.com/s/92jqmcyu412nbcu/LibriSpeech.tar.bz2.sha256?dl=0" -o LibriSpeech.tar.bz2.sha256
sha256sum -c LibriSpeech.tar.bz2.sha256 && rm LibriSpeech.tar.bz2.sha256
cat LibriSpeech.tar.bz2.parta* | tar -C data --use-compress-program lbunzip2 -xvf -
rm -f LibriSpeech.tar.bz2.parta*
echo "...done"

echo "Downloading Ch18 data..."
curl -L "https://www.dropbox.com/s/gmhhij7uand8e42/seq2seq.tar.xz?dl=0" -o  seq2seq.tar.xz
curl -L "https://www.dropbox.com/s/1oxf6etpff8fsin/seq2seq.tar.xz.sha256?dl=0" -o  seq2seq.tar.xz.sha256
sha256sum -c seq2seq.tar.xz.sha256 && rm seq2seq.tar.xz.sha256
tar -C data -xJvf seq2seq.tar.xz && rm seq2seq.tar.xz
echo "...done"

echo "Downloading Ch19 data..."
curl -L "https://www.dropbox.com/s/66uhiglbibstrsd/cloth_folding_rgb_vids.tar.xz?dl=0" -o cloth_folding_rgb_vids.tar.xz
curl -L "https://www.dropbox.com/s/siufwdahl5g6muo/cloth_folding_rgb_vids.tar.xz.sha256?dl=0" -o cloth_folding_rgb_vids.tar.xz.sha256
sha256sum -c cloth_folding_rgb_vids.tar.xz.sha256 && rm cloth_folding_rgb_vids.tar.xz.sha256
tar -C data -xJvf cloth_folding_rgb_vids.tar.xz && rm cloth_folding_rgb_vids.tar.xz
echo "...done"

