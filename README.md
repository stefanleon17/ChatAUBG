# ChatAUBG

This project consists of a basic corpus-based chatbot, trained using a dataset obtained by scraping text data off of the AUBG website (aubg.edu). It uses a Markov-Chain-Based algorithm to generate text, and makes use of a combined unigram-bigram classification algorithm to classify user input into one of the webpages considered, subsequently using the text present on that webpage to bias the generation algorithm, leading, in theory, to more specific and accurate replies. Final results are mixed, with some of the functionalities performing as hypothesized, and others performing more unpredictably. Implications and potential future improvements are further discussed. 

Due to the large storage space necessary for the data storage of the unigram-bigram classification algorithm, these files have been compressed in a file called "large_models.zip" for storage. Before use, these files should be extracted to the same folder as the rest of the project.
