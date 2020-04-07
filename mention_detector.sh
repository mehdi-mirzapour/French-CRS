#!/bin/bash

source ~/neural-end-to-end-coref/.virtenv/bin/activate
python ~/neural-end-to-end-coref/caren/formats/raw_text.py $1/mention_detector_files/raw_text.txt  $1/mention_detector_files/built-ngrams.json
python ~/neural-end-to-end-coref/caren/detmentions.py ~/neural-end-to-end-coref/1.model $1/mention_detector_files/built-ngrams.json $1/mention_detector_files/mentions.json
deactivate
