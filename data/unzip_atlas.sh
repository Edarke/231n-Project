#!/usr/bin/env bash

find . -name "*.gz" | xargs -P 5 -I fileName sh -c 'gunzip "fileName"'
# remove compressed files
# find . -depth -name '*.gz' -exec rm {} \;
