import re
import emoji
import string
import random
import inflect
import pandas as pd
from typing import Union
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer


def retrieve_lm(model: str='bert', token_length: Union[int, str]='avg', case_sensitive: bool=False, return_dict: bool=False) -> dict:
    """
    Retrieves language model to use for tokenization.
        Input(s):
            - model: str :: name of the model to use for tokenization -> 'bert' | 'roberta' | 'roberta-large'
            - token_length: int or str :: token length to use for tokenization -> 50 | 'avg' | 'max' | 'min'
            - case_sensitive: bool :: whether to use case sensitive tokenization
            - return_dict: bool :: whether to return a dictionary containing the model and tokenizer
        Output(s):
            - dict :: dictionary containing the model, tokenizer and token length
    """
    if model == 'bert':
        case  = 'bert-base-uncased' if case_sensitive==False else 'bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(case, return_dict=return_dict)
        model = BertModel.from_pretrained(case, return_dict=return_dict)

    elif model == 'roberta':
        case  = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(case, return_dict=return_dict)
        model = RobertaModel.from_pretrained(case, return_dict=return_dict)

    elif model == 'roberta-large':
        case  = 'roberta-large'
        tokenizer = RobertaTokenizer.from_pretrained(case, return_dict=return_dict)
        model = RobertaModel.from_pretrained(case, return_dict=return_dict)

    else:
        raise ValueError("Wrong model provided, please check model parameter.")

    if (not isinstance(token_length, int) and not isinstance(token_length, str)) or \
       (not isinstance(token_length, str) and not isinstance(token_length, int)):
        raise ValueError("Wrong token length provided, please check token_length parameter.")
    if isinstance(token_length, str) and token_length not in ['avg', 'max', 'min']:
        raise ValueError("Wrong token length provided, please check token_length parameter.\n\
                          It must be one of 'avg', 'max' or 'min' values.")

    return {
        'model':          model,
        'tokenizer':      tokenizer,
        'token_length':   token_length,
        'embedding_size': model.config.hidden_size
    }


def preprocess_tweet(tweet: str, options: dict) -> str:
    """
    This function preprocesses a tweet and returns cleaned version.
        Input(s):
            - tweet: str :: tweet to be preprocessed
            - options: dict :: dictionary containing preprocessing options
        Output(s):
            - str :: preprocessed tweet
    """
    if options['mentions']['enabled']:
        if options['mentions']['keep_after']: tweet = re.sub('(@)(\S+)', r' \2', tweet)
        else: tweet = re.sub('@\S+', '', tweet)
    if options['hashtags']['enabled']:
        if options['hashtags']['keep_after']: tweet = re.sub('(#)(\S+)', r' \2', tweet)
        else: tweet = re.sub('#\S+', '', tweet)
    if options['brackets']['enabled']:
        if options['brackets']['keep_inside']: tweet = re.sub('(\[)(.*?)(\])', r' \2', tweet)
        else: tweet = re.sub('\[.*?\]', '', tweet)
    if options['emojis']:
        if emoji.emoji_count(tweet) > 0:
            if options['emojis']['replace']:
                delimiters = ('﴾﴾﴾', '﴿﴿﴿')
                tweet = emoji.demojize(tweet, delimiters=delimiters)
                tweet = re.sub(f'({delimiters[0]})(\S+)({delimiters[1]})',
                                    lambda x: str.replace(re.sub(f'({delimiters[0]})(\S+)({delimiters[1]})', r' (\2)', x.group()), '_', ' ').lower() if options['lowercase']
                                    else str.replace(re.sub(f'({delimiters[0]})(\S+)({delimiters[1]})', r' (\2)', x.group()), '_', ' ').upper(),
                                    tweet)
            else:
                tweet = emoji.replace_emoji(tweet, '')
    if options['hyperlinks']:
        tweet = re.sub('https?://\S+|www\.\S+', '', tweet) # remove hyperlinks
        tweet = re.sub('bit.ly/\S+', '', tweet)            # remove bitly links
        tweet = re.sub('pic.twitter\S+', '', tweet)        # remove pic.twitter links
    if options['lowercase']:
        tweet = tweet.lower()
    if options['punctuation']:
        tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
    if options['numbers']['enabled']:
        if options['numbers']['replace']:
            tweet = re.sub('(\d+)', lambda x: f' {inflect.engine().number_to_words(x.group())}', tweet)
        else: tweet = re.sub('\d+', '', tweet)
    if options['replacements']['enabled']:
        for key, value in options['replacements']['dict'].items():
            tweet = re.sub(key, value, tweet)

    tweet = re.sub(' +', ' ', tweet)  # remove extra spaces
    tweet = tweet.strip()  # strip leading and trailing spaces
    return tweet


def data_sampler(df: pd.DataFrame, sample_size: Union[int, float],
                 distribution_column: pd.Series = None, target_column_name: str = None,
                 shuffle: bool = True, random_state: int = None, reset_index: bool = True) -> pd.DataFrame:
    """
    Samples a dataframe based on value distribution of a another column.
        Input(s):
            - df: pandas dataframe :: dataframe that we want to sample
            - sample_size: int or float :: size of the sample, int for absolute size (1 to inf), float for fraction of the total size (0 to 1.0)
            - distribution_column: pandas series :: column that we want to use for calculation of the target class distributions
            - target_column_name: str :: name of the target column which we want to sample based on
            - shuffle: bool :: wheather or not shuffle the dataframe before sampling
            - random_state: int :: seed value for random state (used for reproducibility purposes)
        Output(s):
            - pandas dataframe: sampled dataframe
    """
    if random_state == None:
        random_state = random.randint(0, 10000)
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if sample_size <= 0:
        raise ValueError("Sample size must be greater than 0!")

    if sample_size <= 1:
        if distribution_column is None:
            sample_df = df.sample(frac=sample_size, random_state=random_state)
            return sample_df.reset_index(drop=True) if reset_index else sample_df
        else:
            sample_size = int(sample_size * len(df))
    
    if sample_size > 1:
        if type(sample_size) != int:
            raise ValueError("Sample size larger that 1 must be an integer!")
        if distribution_column is None:
            sample_df = df.sample(frac=sample_size/len(df), random_state=random_state)
            return sample_df.reset_index(drop=True) if reset_index else sample_df
    
    if sample_size < len(distribution_column.unique().tolist()):
        raise ValueError("Sample size must be larger than the number of unique values in the distribution column!")

    if (distribution_column is not None) and (target_column_name is None):
        raise ValueError('target_column_name must be specified if distribution_column exists.')

    if distribution_column is not None:
        if sorted(distribution_column.unique().tolist()) != sorted(df[target_column_name].unique().tolist()):
            raise ValueError('distribution_column and target_column_name values don\'t match.')


    # distributions = distribution_column.value_counts().to_dict()
    # distributions_ratio = {k: v / sum(distributions.values()) for k, v in distributions.items()}
    distributions_ratio = distribution_column.value_counts(normalize=True).to_dict()
    sample_size_per_class = {k: int(sample_size * v) for k, v in distributions_ratio.items()}
    for target_class, count_needed in sample_size_per_class.items():
        count_in_df = len(df[df[target_column_name] == target_class])
        if count_needed > count_in_df:
            raise ValueError(f'sample_size is too large for the target column "{target_column_name}", try again with a lower sample_size value.\n\
                             "{target_class}" is represented with only {count_in_df} samples in sampling dataframe, which is lower than required minimum samples ({count_needed}) to fullfill distribution ratio goal.')

    candidate_samples_index_list = []
    for target_class, count in sample_size_per_class.items():
        candidate_samples_index_list.append(df[df[target_column_name] == target_class].sample(count, random_state=random_state).index.tolist())

    candidate_samples_index = [item for sublist in candidate_samples_index_list for item in sublist]
    sample_df = df.loc[candidate_samples_index]
    return sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True) if reset_index else sample_df
