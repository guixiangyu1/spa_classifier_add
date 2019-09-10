from model.data_utils import CoNLLDataset, CoNLLdata4classifier, get_processing_word
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config，这里的config实现了load data的作用
    #拥有词表、glove训练好的embeddings矩阵、str->id的function
    config = Config()
    config.nepochs          = 200
    config.dropout          = 0.5
    config.batch_size       = 60
    config.lr_method        = "adam"
    config.lr               = 0.0005
    config.lr_decay         = 1.0
    config.clip             = -2.0 # if negative, no clipping
    config.nepoch_no_imprv  = 8

    config.dir_model = config.dir_output + "model.finetuning.weights/"
    
    # build model
    model = NERModel(config)
    model.build("fine_tuning")
    model.restore_session("results/test/model.weights/", indicate="fine_tuning")

    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets [(char_ids), word_id]
    # processing_word = get_processing_word(lowercase=True)
    dev = CoNLLDataset(config.filename_dev)
    train = CoNLLDataset(config.filename_train)
    test = CoNLLDataset(config.filename_test)

    # train model

    train4cl = CoNLLdata4classifier(train, processing_word=config.processing_word,
                                    processing_tag=config.processing_tag, context_length=config.context_length)
    dev4cl = CoNLLdata4classifier(dev, processing_word=config.processing_word,
                                  processing_tag=config.processing_tag, context_length=config.context_length)
    test4cl = CoNLLdata4classifier(test, processing_word=config.processing_word,
                                   processing_tag=config.processing_tag, context_length=config.context_length)

    model.train(train4cl, dev4cl, test4cl)

if __name__ == "__main__":
    main()
