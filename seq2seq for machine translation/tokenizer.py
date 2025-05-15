import os 
import glob
import argparse 
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece 
from tokenizers import normalizers
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing


special_token_dict = {"unknown_token": "[UNK]",
                      "pad_token": "[PAD]", 
                      "start_token": "[BOS]",
                      "end_token": "[EOS]"}


def train_tokenizers(path_to_root_data):
    """
    We need to train a WordPiece tokenizer on our french data (as our regular tokenizers are mostly for English!)
    I set all the special tokens we need above:
        unkown_token: Most important incase tokenizer sees a token not a part of our original token set
        pad_token: Padding for the french text
        start_token: Prepend all french text with start token so the decoder has an input to start generating from
        end_token: Append all french text with end token so decoder knowsn when to stop generating anymore. 

    The only thing in here to keep in mind is the normalizers. There are some issues with how the same letter can 
    be represented in Unicode, so we have to do unicode normalization.

    For example: 

    "é" can be written as either (\u00E9) as a single unicode
    "é" can also be written as "e" + ' where we break the accents off of the e and write as a sequence of 2 unicode characters \u0065\u0301

    We want all our data to be in one or the either for some consistency, so we will be using NMC which tries to represent these characters
    with just a single unicode
    """
    ### Prepare Tokenizer Definition ###
    tokenizer_model = WordPiece(unk_token=special_token_dict["unknown_token"])
    tokenizer = Tokenizer(tokenizer_model)
    pass
    