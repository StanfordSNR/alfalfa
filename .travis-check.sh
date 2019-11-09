#!/bin/bash -ex

cd /home/user/alfalfa
./autogen.sh
./configure --enable-debug
make -j3 distcheck V=1 || (cat alfalfa-0.1/_build/sub/src/tests/test-suite.log && exit 1)
