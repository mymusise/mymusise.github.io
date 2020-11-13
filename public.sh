#! /bin/bash

git add --all
git commit -m "update"
git push 

rm -rf /tmp/docs/*

hugo -D

git checkout master

cp -r /tmp/docs/* ./

git add --all
git commit -m "update"
git push

