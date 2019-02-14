from typing import Dict, Optional, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary, Instance
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.models.parser_aux import BiaffineDependencyParser
from allennlp.models.mt_aux import MTAux
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.training.metrics import AttachmentScores

## MT tools
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {'``', "''", ':', ',', '.', 'PU', 'PUNCT', 'SYM'}

@Model.register("parser_mt")
class ParserMT(Model):
    """
    This dependency parser follows the model of
    ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ .

    Word representations are generated using a bidirectional LSTM,
    followed by separate biaffine classifiers for pairs of words,
    predicting whether a directed arc exists between the two words
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimial Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : ``int``, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for head arc prediction.
    tag_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 tag_representation_dim: int,
                 arc_representation_dim: int,
                 encoder_mt: Seq2SeqEncoder,
                 encoder_additional: Seq2SeqEncoder = None,
                 max_decoding_steps: int = 30,
                 languages: List[str] = ['eng', 'cmn'],
                 langid_dim: int = 0,
                 tag_feedforward: FeedForward = None,
                 arc_feedforward: FeedForward = None,
                 pos_tag_embedding: Embedding = None,
                 use_mst_decoding_for_validation: bool = True,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 projection_dim: int = 200,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 beam_size: int = None,
                 #target_namespace: str = "pieces",
                 target_namespace: str = "tokens",
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True) -> None:
        super(ParserMT, self).__init__(vocab, regularizer)


        ## Biaffine Parser
        self.parser = BiaffineDependencyParser(vocab,
                 text_field_embedder, encoder, tag_representation_dim,
                 arc_representation_dim, languages, langid_dim, 
                 tag_feedforward, arc_feedforward, pos_tag_embedding,
                 use_mst_decoding_for_validation, dropout, input_dropout, initializer, regularizer)
        self.encoder_additional = encoder_additional

        ## MT
        target_embedding_dim = self.parser.encoder.get_output_dim()
        self.machine_trans = MTAux(vocab, encoder_mt, target_embedding_dim, max_decoding_steps, projection_dim,
                                    attention, attention_function, beam_size, target_namespace, scheduled_sampling_ratio)


    """ 
    Notes from jkasai
    We get rid of POS tag dependency in this multitasking to facilitate non-parsing input (technically, we can add predicted POS tags from Udpipe, but leave it for future.
    We add kwargs for everything not shared between two tasks. Kwargs also indicates which type of batch we are feeding.
    """

    @overrides
    def forward(self,  # type: ignore
                words: Dict[str, torch.LongTensor],
                pos_tags: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                **kwargs) -> Dict[str, torch.Tensor]:
                #langid: torch.LongTensor,
                #pos_tags: torch.LongTensor,
                #head_tags: torch.LongTensor = None,
                #head_indices: torch.LongTensor = None
                #target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, sequence_length)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pos_tags : ``torch.LongTensor``, required.
            The output of a ``SequenceLabelField`` containing POS tags.
            POS tags are required regardless of whether they are used in the model,
            because they are used to filter the evaluation metric to only consider
            heads of words which are not punctuation.
        head_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape ``(batch_size, sequence_length)``.
        head_indices : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length)``.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        arc_loss : ``torch.FloatTensor``
            The loss contribution from the unlabeled arcs.
        loss : ``torch.FloatTensor``, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : ``torch.FloatTensor``
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : ``torch.FloatTensor``
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.
        """

        encoded_text, mask = self.parser.get_encoded_text(words, pos_tags)
        #output_dict = self.parser.forward(words, kwargs['langid'], kwargs['pos_tags'], metadata, kwargs['head_tags'], kwargs['head_indices'])
        # first find which type of batch
        parsing = False
        if 'head_indices' in kwargs.keys():
            parsing = True
        if parsing:
            if self.encoder_additional:
                encoded_text = self.encoder_additional(encoded_text, mask)
            output_dict = self.parser.get_output_dict(encoded_text, mask, metadata, pos_tags, kwargs['head_tags'], kwargs['head_indices'])
        else:
            output_dict = self.machine_trans.forward(encoded_text, mask, kwargs['target_tokens'])

        return output_dict

    @overrides
    def forward_on_instances(self,
                             instances: List[Instance]) -> List[Dict[str, numpy.ndarray]]:
        return self.parser.forward_on_instances(instances)

    """
    Notes from jkasai
    beam_search and bleu metric are ready. Need to implement it in the master model.
    """

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.parser.get_metrics(reset)

