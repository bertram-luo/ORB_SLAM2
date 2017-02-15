cd build
make -j4 && cd .. && ./Examples/Monocular/mono_fdulab_tracking Vocabulary/ORBvoc.txt ./Examples/Monocular/fdulab201608272217.yaml ~/master/experiments/data/lab201608272217human.mov
