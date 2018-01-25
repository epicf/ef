../../ef.out Bush.conf &
PID=$!
sleep 2
kill $PID
python3 4D_interp.py
../../ef.out out_test_0000000.h5
python3 plot.py
