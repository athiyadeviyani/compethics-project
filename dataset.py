import torch
from torch.utils.data import Dataset
import re 
from nltk.tokenize import TweetTokenizer


class CausalLMDataset(Dataset):

    def __init__(
        self,
        token_list = None,
        tokenizer = None,
        df = None,
        text_col = None,
    ):

        """
        Provide either `token_list` and `tokenizer` together, or `df` and `text_col` together.

        :param df: dataframe of (prompt, response) pairs
        :param text_col: column of texts you want to model

        This doesn't do much; __getitem__ just returns the raw text of a single example.
        It's the collate function of the dataloader that does the heavy lifting.
        """

        self.from_text = True
        self.tokenizer = tokenizer

        if token_list is None:
            # preprocess
            texts = df[text_col].apply(CausalLMDataset.preprocess_text).tolist()

            # save
            self.texts = texts
            self.from_text = True
        else:
            self.texts = token_list
            self.from_text = False


    @staticmethod
    def preprocess_text(txt):
        txt = str(txt).lower()

        # url and tag
        words = []
        for word in txt.split():
            if word[0] == '#': # don't allow tag
                continue
            i = word.lower().find('http')
            if i >= 0:
                word = word[:i] + ' ' + '__url__'
            words.append(word.strip())
        txt = ' '.join(words)

        # remove illegal char
        txt = txt.replace(chr(92),'') # chr(92) = '\'. as twitter has 'b\/c' rather than 'b/c'
        txt = txt.replace("b/c","because").replace('j/k','just kidding').replace('w/o','without').replace('w/','with')
        txt = re.sub('__mention__','MENTION',txt)
        txt = re.sub('__url__','URL',txt)
        txt = re.sub(r"[^A-Za-z0-9()\[\]:,.!?'“” ]", " ", txt)
        txt = re.sub('MENTION','__mention__',txt)
        txt = re.sub('URL','__url__',txt)

        # tokenizer = TweetTokenizer(preserve_case=True)
        # txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '

        # remove un-necessary space
        return ' '.join(txt.split())
        
        
    def __len__(self):
        return len(self.texts)
    
    
    def __getitem__(self, index):
        '''
        __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalized source and
        target values using the vocabulary objects we created in __init__
        '''


        if self.from_text:
            return {'text': self.texts[index]}
        else:
            tokens = self.texts[index]
            tokens = [int(token) for token in tokens.split(' ')]
            text = self.tokenizer.decode(tokens)
            return {'text': text}