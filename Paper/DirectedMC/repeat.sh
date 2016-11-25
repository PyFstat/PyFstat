for ((n=0;n<100;n++))
do
echo $n
python generate_data.py --quite --no-template-counting
done
