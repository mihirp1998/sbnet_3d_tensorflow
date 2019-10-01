export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-9.0/bin:$PATH
sudo rm /usr/bin/g++ /usr/bin/gcc
sudo ln -s /usr/bin/gcc-5 /usr/bin/gcc
sudo ln -s /usr/bin/g++-5 /usr/bin/g++

for f in avx512fintrin.h avx512pfintrin.h avx512vlintrin.h; do
   curl -H "User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36" -o $f "https://gcc.gnu.org/viewcvs/gcc/branches/gcc-5-branch/gcc/config/i386/${f}?view=co&revision=245536&content-type=text%2Fplain&pathrev=245536"
done && sudo mv avx512*intrin.h  /usr/lib/gcc/x86_64-linux-gnu/5/include/