for v in '0.1' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0' '1.1' '1.2'
do
    ((a = 0))
    while ((a < 1))
    do
	qsub -v P='angspeed',V=$v scan.sh
	((a += 1))
    done
done

