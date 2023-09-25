## Co-occurrence analysis for

This repository is a set of functions used to derive significant species pairs or potential co-occurrences from presence/absence data.
To this end, we use a text analysis algorithm ([Dunning et al., 1993](https://dl.acm.org/doi/10.5555/972450.972454); [Wahl and Gries, 2018](https://link.springer.com/chapter/10.1007/978-3-319-92582-0_5)) that identifies pairs of species which co-occur more frequently than expected by their individual presence patterns.
The algorithm assigns an association score; called likelihood ratio; to each possible species pair. This likelihood ratio compares the probability of two species
co-occurring to the probability of one species occurring without the other or when both species are absent using Shannon's entropy.
We further distinguish between strong co-occurrences and strong one-sided occurrences and co-absences by comparing the observed co-occurrence frequency with the 
product of the individual species presences ([Evert, 2009](https://api.semanticscholar.org/CorpusID:13224169)).

The functions in this script where used in [Hofmann et al., 2021](https://www.sciencedirect.com/science/article/pii/S0079661121000203) to derive species networks, 
and in [Benedetti et al., 2021](https://www.nature.com/articles/s41467-021-25385-x) to characterize the changes in potential species co-occurrences with climate change.


