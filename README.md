# music-recommender-systems
Machine learning course project at WUT 2023

# Results
For the business criterium of at least 1 recommended on playlist track to be fully played by user, the models scored on AB tests as follows:

| **model VAE** | **model classifier** |
| :---: | :---: |
| 70% | 30% |

The results indicate following conclusions:
* _The more effective approach with usage of ANN is with Variational Autoenkoder latent space search_
* Further architectures must be verified

# Discussion
The VAE model is most similar to representation learning. This way, the recommended tracks are similar to tracks listened by the user. The effects were acceptable, however the moethod may in a long term result in user being trapped in sort of **echo chamber**, as the tracks will be similar only to tracks he listened.

On the other hand, the classifier model attempted to solve a problem via classifying the preference of user towards each track. As the input is based not only on the hstory of listened tracks, but also user preference, there is a hope of classifying the track out of users mainstream listening preference as interesting. Nevertheless, this approach has many downsides: (1) very expensive & long training, (2) necessity of running the whole model with each and every track we are thinking of proposing, (3) constant results for given users, (4) the model is very prone to overfitting. As a result of above, the buisness criteria accuracy achived by this model were unacceptable.

# Conclusion
The results indicate the better approach to the given problem by usage of ANN is representation learning crossed with K-Nearest Neighbours algorithm then track preference classification.
