# Install DensePhrases (ThangPM: Will support it later)
git clone -b v1.1.0 https://github.com/princeton-nlp/DensePhrases.git
cd DensePhrases
# yes yes | ./config.sh
rm -rf slides/
python setup.py develop

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader stopwords
python -m spacy download en
