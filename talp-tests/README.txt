1. make LDFLAGS="-L/opt/apps/ucx/1.17.0/lib /opt/apps/ucx/1.17.0/lib/libucs.so -Wl,-rpath,/opt/apps/ucx/1.17.0/lib"
To successfully install DLB with the current available modules

2. export LD_LIBRARY_PATH=/opt/apps/ucx/1.17.0/lib:$LD_LIBRARY_PATH
   export LIBRARY_PATH=/opt/apps/ucx/1.17.0/lib:$LIBRARY_PATH

these environment variables may need to be set

3. mpic++ -o helloworld helloworld.cpp -L /home1/09830/vikaskurapati/seissol_env/local/lib -ldlb -Wl,-rpath,/home1/09830/vikaskurapati/seissol_env/local/lib
to compile program with TALP support

4. export DLB_ARGS="--talp"
preload="<DLB_PREFIX>/lib/libdlb_mpi.so"
mpirun <opts> env LD_PRELOAD="$preload" ./app

to run with talp support

but for vista
ibrun env LD_PRELOAD="$preload" ./mpi-openmp

