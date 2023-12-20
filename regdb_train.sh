for trail in 1 2 3 4 5 6 7 8 9 10
#for trail in 9 10
do
  python train.py --dataset regdb --gpu 8 --trial $trail --method awg -b 10 --lr 0.01 --mess "1. restore dataset, 2. lr 0.01"
done
echo 'Done!'