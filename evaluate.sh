#!/bin/sh -eux

make

echo `date` > files/score.out

for i in `seq -w 1 20`
do
    ./graph_generator.out files/random_${i}.in 0 $i
    ./hokudai_procon1 < files/random_${i}.in > files/random_${i}.out
    echo random_${i} >> files/score.out
    ./score_evaluator.out files/random_${i}.in files/random_${i}.out >> files/score.out
done
for i in `seq -w 1 20`
do
    ./graph_generator.out files/complete_${i}.in 1 $i
    ./hokudai_procon1 < files/complete_${i}.in > files/complete_${i}.out
    echo complete_${i} >> files/score.out
    ./score_evaluator.out files/complete_${i}.in files/complete_${i}.out >> files/score.out
done
