#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


def main():
    data = ["I love machine learning. Its awesome.",
            "I love coding in python",
            "I love building chatbots",
            "they chat amagingly well",
            "kite surfing my passion"]

    tagged_data = [
        TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for
        i, _d in enumerate(data)]

    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Model Saved")


if __name__ == '__main__':
    main()
