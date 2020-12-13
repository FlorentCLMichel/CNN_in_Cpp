cd C++/
g++ -O3 -I ~/miniconda3/include/python3.8 -fpic -c -o CNN3.o CNN3.cpp
g++ -o CNN3.so -shared CNN3.o -lboost_python3 -lboost_numpy3
mv CNN3.so ../
rm CNN3.o
cd ../
