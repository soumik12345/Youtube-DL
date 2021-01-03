export KAGGLE_USERNAME=$1
export KAGGLE_KEY=$2
kaggle datasets download -d jerzydziewierz/bee-vs-wasp
unzip -q bee-vs-wasp.zip
rm bee-vs-wasp.zip
