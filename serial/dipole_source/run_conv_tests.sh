for i in 0 1 2 3
do
for j in 0 1
do
	./dipole_source -p $i -graded $j >> ~/temp_output/output.txt
done
done

