cd build
make clean
cd ..
rm -rf build
mkdir build
cd build
cmake ..
make -j 4
cd ..
