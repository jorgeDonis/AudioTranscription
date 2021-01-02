#!/bin/bash

MUSICXMLFILES=../data_SCORE_XML/*

for file in  $MUSICXMLFILES
do
	new_filename="../data_SCORE_HUMDRUM/$(basename "$file" | sed 's/\(.*\)\..*/\1/').txt"
	./xml2hum $file > $new_filename
done


