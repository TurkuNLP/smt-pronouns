

# $1 is architecture file
# $2 is the model file
# $3 is then lang pair

####So, let's first predict and tag the dev_set
python output_set.py $1 $2 /home/jmnybl/git_checkout/smt-pronouns/train_data/all.$3.filtered.withids /home/jmnybl/git_checkout/smt-pronouns/dev_data/TEDdev.$3.data.filtered.withids $3-dev
####Let's print scores for the dev
perl /home/jmnybl/wmt16_pronoun/WMT16_CLPP_scorer.pl $3-dev_gold $3-dev_pred $3

##Okay and now the same for the test

#Let's convert the format first
python make_test_set_correct_format.py /home/jmnybl/git_checkout/smt-pronouns/test_data/WMT2016.$3.data.filtered.final.withids | cut -f 1-6 > temp_test_set
python output_set.py $1 $2 /home/jmnybl/git_checkout/smt-pronouns/train_data/all.$3.filtered.withids temp_test_set $3-test
####Let's print scores for the test to see if it goes through the script
perl /home/jmnybl/wmt16_pronoun/WMT16_CLPP_scorer.pl $3-test_pred $3-test_pred $3
mkdir test_outputs
cp $3-test_pred ./test_outputs
