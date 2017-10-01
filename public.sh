#! /bin/bash

git add --all
git commit -m "update"
git push 

hugo -D

git checkout master

cp -r /tmp/docs/* ./

git add --all
git commit -m "update"
git push

