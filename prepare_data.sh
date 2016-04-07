#!/bin/bash

# en-fr train
paste -d"\t" Europarl.en-fr.data.filtered Europarl.en-fr.doc-ids | cut -f 1-6 > train_data/Europarl.en-fr.data.filtered.withids
paste -d"\t" IWSLT15.en-fr.data.filtered IWSLT15.en-fr.doc-ids > train_data/IWSLT15.en-fr.data.filtered.withids
paste -d"\t" NCv9.en-fr.data.filtered NCv9.en-fr.doc-ids | cut -f 1-6 > train_data/NCv9.en-fr.data.filtered.withids

cat train_data/Europarl.en-fr.data.filtered.withids train_data/IWSLT15.en-fr.data.filtered.withids train_data/NCv9.en-fr.data.filtered.withids > train_data/all.en-fr.filtered.withids

# en-fr dev
paste -d"\t" TEDdev.en-fr.data.filtered TEDdev.en-fr.doc-ids > dev_data/TEDdev.en-fr.data.filtered.withids

#paste -d"\t" Europarl.en-de.data.filtered Europarl.de-en.doc-ids | cut -f 1-6 > train_data/Europarl.en-de.data.filtered.withids
