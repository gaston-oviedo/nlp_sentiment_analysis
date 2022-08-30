## Sentiment Analysis on Movies Reviews

This project is related to NLP.

Regarding the data, can be downloaded from [AI Stanford Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). When you download them you will notice that their format is text files, so you will have to work a little there to be able to use and process them. You can use the **prepare_data.py** script to prepare the data. I suggest you name the folder as *movie_reviews_dataset* in order to not have to change the notebook later. 

Basically a basic sentiment analysis problem, as in this case, consists of a classification problem, where the possible output labels are: `positive` and `negative`. Which indicates, if the review of a movie speaks positively or negatively.

## Install

You can use `Docker` to easily install all the needed packages and libraries:

```bash
$ docker build -t s06_project .
```

### Run Docker

```bash
$ docker run --rm -it \
    -p 8888:8888 \
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    s06_project \
    bash
```

## Run Project

It doesn't matter if you are inside or outside a Docker container, in order to execute the project you need to launch a Jupyter notebook server running:

```bash
$ jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

Then, inside the file `Sentiment_Analysis_NLP.ipynb`, you can see the entire project.

## Tests

There are some basic tests in `Sentiment_Analysis_NLP.ipynb` that you must be able to run without errors.