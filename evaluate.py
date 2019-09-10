import sys

from model.data_utils import CoNLLDataset, get_processing_word, CoNLLdata4classifier
from model.ner_model import NERModel
from model.config import Config



def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build("train")
    model.restore_session(config.dir_model)

    # create dataset

    # processing_word = get_processing_word(lowercase=True)

    if len(sys.argv) == 2:
        if sys.argv[1] == 'test':
            test = CoNLLDataset(config.filename_test)

        elif sys.argv[1] == 'dev':
            test = CoNLLDataset(config.filename_dev)

    else:
        assert len(sys.argv) == 1
        test = CoNLLDataset(config.filename_test)

    test4cl = CoNLLdata4classifier(test, processing_word=config.processing_word,
                                   processing_tag=config.processing_tag)

    # evaluate and interact
    model.evaluate(test4cl)
    # interactive_shell(model)


if __name__ == "__main__":
    main()
