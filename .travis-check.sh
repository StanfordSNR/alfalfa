#!/bin/bash -ex

cd /home/user/alfalfa
./autogen.sh
./configure
make -j3 distcheck V=1 || (cat alfalfa-0.1/_build/src/tests/test-suite.log && exit 1)
