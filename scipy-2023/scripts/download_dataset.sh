#! /usr/bin/env sh
# downloads a dataset with wget
URL=$1
OUTPUT_FILEPATH=$2

echo "Downloading dataset from ${URL}"
wget $URL -O $OUTPUT_FILEPATH
