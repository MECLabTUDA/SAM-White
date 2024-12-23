#!/bin/bash

TARGET=~/HESSENBOX-DA/SAM_whitepaper/

mkdir -p "$TARGET"
cp -r out/* "$TARGET"
rm -r "$TARGET"/*/*.pt
