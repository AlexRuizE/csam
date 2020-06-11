#!/bin/sh

input_path=$1

docker run -v ${input_path}:/var/local csam

# Strips the container's directory out of the output
CWD=`pwd`
cd $1/output
FILE=`ls -Art | tail -n 1`
cat $FILE | sed 's|\/var\/local\/\(.*\)|'$input_path'\/\1|' > $FILE.tmp
rm $FILE
mv $FILE.tmp $FILE
