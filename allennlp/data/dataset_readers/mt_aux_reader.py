from typing import Dict
import logging, json, os

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("mt_aux_reader")
class MTAuxDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 use_pieces: bool = False,
                 use_language_specific_pos: bool = False) -> None:
        super().__init__(lazy)
        source_add_start_token: bool = False
        """
        Notes from jkasai
        Do not add start or end token in source to be contingent with the parsing encoder
        Question. Alternatives? Is this an okay thing to do?
        """
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        #self._target_token_indexers = target_token_indexers or {"pieces": SingleIdTokenIndexer()} 
        #self._target_token_indexers = target_token_indexers or {"tokens": SingleIdTokenIndexer()} 
        if use_pieces:
            self._target_token_indexers = {"tokens": SingleIdTokenIndexer(namespace='pieces')} 
        else:
            self._target_token_indexers = {"tokens": SingleIdTokenIndexer(namespace='tokens')} 
        self._source_add_start_token = source_add_start_token
        self._use_pieces = use_pieces
        self.use_language_specific_pos = use_language_specific_pos

    @overrides
    def _read(self, file_path):
        """
        In this auxiliary framework, we consider both directions.
        jkasai
        """
        for root, _, files in list(os.walk(file_path)):
            for data_file in files:
                file_path = os.path.join(root, data_file)
                with open(cached_path(file_path), "r") as data_file:
                    logger.info("Reading bitext instances from jsonl dataset at: %s", file_path)
                    for line in data_file:
                        example = json.loads(line)
                        lang1_text = example["lang1"]['text']
                        lang2_text = example["lang2"]['text']
                        if self._use_pieces:
                            lang1_piece = example["lang1"]['piece']
                            lang2_piece = example["lang2"]['piece']
                        else:
                            lang1_piece = example["lang1"]['text']
                            lang2_piece = example["lang2"]['text']
                        if self.use_language_specific_pos:
                            lang1_pos = example["lang1"]['xpos']
                            lang2_pos = example["lang2"]['xpos']
                        else:
                            lang1_pos = example["lang1"]['upos']
                            lang2_pos = example["lang2"]['upos']
                        lang1 = example["lang1"]['lang']
                        lang2 = example["lang2"]['lang']
                        data = [(lang1_text, lang1_pos, lang1, lang2_piece, lang2), (lang2_text, lang2_pos, lang2, lang1_piece, lang1)]
                        #data = [(lang1_text, lang1_pos, lang1, lang2_piece, lang2), (lang2_text, lang2_pos, lang2, lang1_piece, lang1), (lang1_text, lang1_pos, lang1, lang1_piece, lang1), (lang2_text, lang2_pos, lang2, lang2_piece, lang2)]
                        for source, source_pos, source_lang, target, target_lang in data:
                            yield self.text_to_instance(source, source_pos, source_lang, target, target_lang)

    @overrides
    def text_to_instance(self, source_string: str, source_pos: str, source_lang: str, target_string: str = None, target_lang: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        #tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source = source_string.split()
        pos_tags = source_pos.split()
        if self._source_add_start_token:
            tokenized_source.insert(0, START_SYMBOL)
            tokenized_source.append(END_SYMBOL)
            pos_tags.insert(0, START_SYMBOL)
            pos_tags.append(END_SYMBOL)
        #source_field = TextField(tokenized_source, self._source_token_indexers)
        #source_field = TextField([Token(source_lang+':'+word) for word in tokenized_source], self._source_token_indexers)
        source_field = TextField([Token(word) for word in tokenized_source], self._source_token_indexers)
        ## Yes, reapplying Token is stupid. Alternative? jkasai
        fields['words'] = source_field
        fields["pos_tags"] = SequenceLabelField(pos_tags, source_field, label_namespace="pos")
        """
        'words' is really 'source_tokens' but we want the same key name as the primary task. So 'words'
        """
        if target_string is not None:
            tokenized_target = target_string.split()
            tokenized_target.insert(0, target_lang + ':' + START_SYMBOL)
            """
            Notes from jkasai
            Add language prefix after adding start tokens to flag which language the MT module is translating into.
            Probably, the end symbol can be language-specific?
            """
            target_field = TextField([Token(word) for word in tokenized_target]+[Token(END_SYMBOL)], self._target_token_indexers)
            fields['target_tokens'] = target_field
            fields["metadata"] = MetadataField({"source_words": tokenized_source, "target_words": tokenized_target, "source_lang": source_lang, "target_lang": target_lang, "pos_tags": pos_tags})
        return Instance(fields)
    def _get_num_samples(self, file_path):
        num_batches = 0
        for root, _, files in list(os.walk(file_path)):
            for data_file in files:
                file_path = os.path.join(root, data_file)
                with open(cached_path(file_path), "r") as data_file:
                    logger.info("Reading bitext instances from jsonl dataset at: %s", file_path)
                    for line in data_file:
                        if '}' in line:
                            num_batches += 1
        #num_batches = num_batches*4
        num_batches = num_batches*2
        ## bidirectional
        return num_batches
