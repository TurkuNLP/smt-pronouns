#!/bin/bash

# en-fr train

echo "*** train ***"

for lang in "en-fr" "en-de" "fr-en" "de-en"
    do
    echo $lang

    if [ "$lang" = "en-fr" ] || [ "$lang" = "fr-en" ]
    then
        idfile="en-fr"
    fi

    if [ "$lang" = "en-de" ] || [ "$lang" = "de-en" ]
    then
        idfile="de-en"
    fi

    if [ "$lang" = "fr-en" ]
    then 
        filtered=""
    else
        filtered=".filtered"
    fi


    paste -d"\t" raw_data/Europarl.$lang.data$filtered raw_data/Europarl.$idfile.doc-ids | cut -f 1-6 > train_data/Europarl.$lang.data$filtered.withids
    paste -d"\t" raw_data/IWSLT15.$lang.data$filtered raw_data/IWSLT15.$idfile.doc-ids > train_data/IWSLT15.$lang.data$filtered.withids
    paste -d"\t" raw_data/NCv9.$lang.data$filtered raw_data/NCv9.$idfile.doc-ids | cut -f 1-6 > train_data/NCv9.$lang.data$filtered.withids
    
    cat train_data/Europarl.$lang.data$filtered.withids train_data/IWSLT15.$lang.data$filtered.withids train_data/NCv9.$lang.data$filtered.withids > train_data/all.$lang$filtered.withids

done 


echo "*** dev ***"

for lang in "en-fr" "en-de" "fr-en" "de-en"
    do
    echo $lang

    if [ "$lang" = "en-fr" ] || [ "$lang" = "fr-en" ]
        then
            idfile="en-fr"
        fi
    if [ "$lang" = "en-de" ] || [ "$lang" = "de-en" ]
        then
            idfile="de-en"
        fi
    
    if [ "$lang" = "fr-en" ]
    then 
        filtered=""
    else
        filtered=".filtered"
    fi

    paste -d"\t" raw_data/TEDdev.$lang.data$filtered raw_data/TEDdev.$idfile.doc-ids > dev_data/TEDdev.$lang.data$filtered.withids
    

done


echo "*** test ***"

for lang in "en-fr" "en-de" "fr-en" "de-en"
    do
    echo $lang

    if [ "$lang" = "fr-en" ]
    then 
        filtered=""
    else
        filtered=".filtered"
    fi


    zcat raw_test_data/$lang/WMT2016.$lang.data$filtered.final.gz | paste -d"\t" - raw_test_data/$lang/WMT2016.$lang.doc-ids | cut -f 1-6 > test_data/WMT2016.$lang.data$filtered.final.withids
    

done

